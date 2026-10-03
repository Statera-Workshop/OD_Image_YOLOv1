"""
YOLOv1 builder (Image task only) - the Legacy YOLOv1 model.

Reference: Redmon et al., "You Only Look Once: Unified, Real-Time Object Detection", CVPR 2016.
Code     : Legacy YOLOv1 project (RichardMinsooGo-ML/Bible_4_Part_F_01_Pytorch_Yolov1, yjh0410 style),
           the implementation the maintainer verified to converge (VOC2007 test mAP50 0.379, VOC07 metric).

Decision (2026-10-01): the paper network (24 Darknet conv + FC 4096 + FC 7x7x30, ImageNet-pretrained
on the Extraction model) is replaced by this Legacy design, which keeps the YOLOv1 idea - one
prediction per grid cell of a single stride-32 map, the cell containing the box centre is
responsible - with a modern body:

    backbone : ResNet-18 (C5, stride 32)                        yolov1_backbone.py
    neck     : SPPF                                             yolov1_neck.py
    head     : decoupled head (cls branch / reg branch)         yolov1_head.py
    pred     : 1x1 obj (from the cls branch) | cls | reg (4)
    box      : centre = (sigmoid(t_xy) + cell) * 32, size = exp(t_wh) * 32
    targets  : centre cell only, one box per cell                matcher.py
    loss     : obj BCE + cls BCE + 5 x GIoU loss, normalised by the positives   loss.py

Pretraining: the Legacy run started from the ImageNet ResNet-18 (torchvision weights). Under the
common protocol the model trains from scratch; pass  --pretrained imagenet  to reproduce the
Legacy setting (the weights are loaded after the common initialisation).
Score: [Unified] objectness x class (common rule); the Legacy demo used sqrt(obj x cls) (same ranking).
"""

import torch
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

from detectors.anchor_free.common import HeadStats, split_targets, to_raw
from detectors.yolov1.loss import Criterion
from detectors.yolov1.yolov1_backbone import build_backbone, model_urls
from detectors.yolov1.yolov1_head import build_head
from detectors.yolov1.yolov1_neck import build_neck

# Legacy config/model_config/yolov1_config.py (multi_scale / trans_type belong to the Legacy trainer)
YOLOV1_CFG = {
    "backbone": "resnet18", "stride": 32, "max_stride": 32,
    "neck": "sppf", "expand_ratio": 0.5, "pooling_size": 5, "neck_act": "lrelu", "neck_norm": "BN",
    "neck_depthwise": False,
    "head": "decoupled_head", "head_act": "lrelu", "head_norm": "BN", "num_cls_head": 2, "num_reg_head": 2,
    "head_depthwise": False,
    "loss_obj_weight": 1.0, "loss_cls_weight": 1.0, "loss_box_weight": 5.0,
}


class YOLOv1(nn.Module):
    def __init__(self, num_classes, cfg=YOLOV1_CFG):
        super().__init__()
        self.cfg, self.num_classes, self.stride = cfg, num_classes, cfg["stride"]
        self.backbone, feat_dim = build_backbone(cfg["backbone"], False)
        self.neck = build_neck(cfg, feat_dim, out_dim=512)
        head_dim = self.neck.out_dim
        self.head = build_head(cfg, head_dim, head_dim, num_classes)
        self.obj_pred = nn.Conv2d(head_dim, 1, kernel_size=1)
        self.cls_pred = nn.Conv2d(head_dim, num_classes, kernel_size=1)
        self.reg_pred = nn.Conv2d(head_dim, 4, kernel_size=1)
        self.criterion = Criterion(cfg, None, num_classes)
        self.stats = HeadStats()
        self.yolo_layers = [self.stats]

    def decode_boxes(self, pred, fmp_size):
        """Legacy decode_boxes: [B, M, 4] (t_x, t_y, t_w, t_h) -> xyxy pixels."""
        h, w = fmp_size
        gy, gx = torch.meshgrid(torch.arange(h, device=pred.device), torch.arange(w, device=pred.device), indexing="ij")
        grid = torch.stack([gx, gy], -1).float().view(-1, 2)
        ctr = (torch.sigmoid(pred[..., :2]) + grid) * self.stride
        wh = torch.exp(pred[..., 2:]) * self.stride
        return torch.cat([ctr - wh * 0.5, ctr + wh * 0.5], -1)

    def forward(self, x, targets=None):
        feat = self.neck(self.backbone(x))
        cls_feat, reg_feat = self.head(feat)
        obj = self.obj_pred(cls_feat)
        cls = self.cls_pred(cls_feat)
        reg = self.reg_pred(reg_feat)
        fmp_size = obj.shape[-2:]
        obj, cls, reg = (t.permute(0, 2, 3, 1).contiguous().flatten(1, 2) for t in (obj, cls, reg))
        box = self.decode_boxes(reg, fmp_size)
        raw = to_raw(box, cls.sigmoid(), obj.sigmoid())
        if targets is None:
            return raw
        tgts = [{k: v.cpu() for k, v in t.items()} for t in split_targets(targets, x.shape[0], x.shape[2])]
        outputs = {"pred_obj": obj, "pred_cls": cls, "pred_box": box, "stride": self.stride, "fmp_size": fmp_size}
        loss_dict = self.criterion(outputs, tgts)
        with torch.no_grad():                                      # logging only
            gt_obj, gt_cls, _ = self.criterion.matcher(fmp_size, self.stride, tgts)
            fg = gt_obj.view(-1).to(x.device) > 0
            self.stats.update(loss_dict, obj.sigmoid().view(-1), cls.reshape(-1, self.num_classes), fg,
                              gt_cls.view(-1, self.num_classes).to(x.device)[fg].argmax(-1))
        return loss_dict["losses"], raw

    def load_darknet_weights(self, path):
        """--pretrained imagenet : torchvision ImageNet ResNet-18 (the Legacy initialisation)."""
        if path != "imagenet":
            raise ValueError("YOLOv1 accepts --pretrained imagenet (ResNet-18 ImageNet weights) only")
        state = model_zoo.load_url(model_urls[self.cfg["backbone"]])
        missing, _ = self.backbone.load_state_dict(state, strict=False)
        print(f"ImageNet {self.cfg['backbone']} loaded ({len(missing)} missing keys)")


def build_model(task, num_classes):
    return YOLOv1(num_classes)
