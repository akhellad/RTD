import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from config_v2 import NUM_CLASSES, CELL_SIZES, GRID_SIZES


class ConvBNSiLU(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class BackBoneResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2   # → C3  512ch  stride 8
        self.layer3 = resnet.layer3   # → C4 1024ch  stride 16
        self.layer4 = resnet.layer4   # → C5 2048ch  stride 32

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class PANet(nn.Module):
    def __init__(self):
        super().__init__()
        # Lateral 1×1
        self.lateral_c5 = nn.Conv2d(2048, 256, 1, bias=False)
        self.lateral_c4 = nn.Conv2d(1024, 256, 1, bias=False)
        self.lateral_c3 = nn.Conv2d(512, 256, 1, bias=False)

        # Top-down smoothing
        self.smooth_p5 = ConvBNSiLU(256, 256, 3, padding=1)
        self.smooth_p4 = ConvBNSiLU(256, 256, 3, padding=1)
        self.smooth_p3 = ConvBNSiLU(256, 256, 3, padding=1)

        # Bottom-up downsample (stride-2) + smoothing
        self.down_3to4 = ConvBNSiLU(256, 256, 3, stride=2, padding=1)
        self.smooth_n4 = ConvBNSiLU(256, 256, 3, padding=1)
        self.down_4to5 = ConvBNSiLU(256, 256, 3, stride=2, padding=1)
        self.smooth_n5 = ConvBNSiLU(256, 256, 3, padding=1)

    def forward(self, c3, c4, c5):
        # Top-down
        p5 = self.lateral_c5(c5)
        p4 = F.interpolate(p5, scale_factor=2, mode='nearest') + self.lateral_c4(c4)
        p3 = F.interpolate(p4, scale_factor=2, mode='nearest') + self.lateral_c3(c3)
        p5 = self.smooth_p5(p5)
        p4 = self.smooth_p4(p4)
        p3 = self.smooth_p3(p3)

        # Bottom-up
        n3 = p3
        n4 = self.smooth_n4(self.down_3to4(n3) + p4)
        n5 = self.smooth_n5(self.down_4to5(n4) + p5)
        return n3, n4, n5


class DetectionHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.shared = nn.Sequential(
            ConvBNSiLU(256, 256, 3, padding=1),
            ConvBNSiLU(256, 256, 3, padding=1),
        )
        self.obj_conv = nn.Conv2d(256, 1, 1)
        self.cls_conv = nn.Conv2d(256, num_classes, 1)
        self.box_conv = nn.Conv2d(256, 4, 1)

    def forward(self, x):
        feat = self.shared(x)
        obj = self.obj_conv(feat)      # [B, 1, H, W]  logits
        cls = self.cls_conv(feat)      # [B, C, H, W]  logits
        box = self.box_conv(feat)      # [B, 4, H, W]  raw (avant exp)
        return obj, cls, box


class ObjectDetectorFPN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = BackBoneResNet()
        self.neck = PANet()
        self.head_p3 = DetectionHead(num_classes)
        self.head_p4 = DetectionHead(num_classes)
        self.head_p5 = DetectionHead(num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        n3, n4, n5 = self.neck(c3, c4, c5)
        return self.head_p3(n3), self.head_p4(n4), self.head_p5(n5)


# ======================================================================
#  Losses
# ======================================================================

def focal_loss(logits, targets, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
    prob = torch.sigmoid(logits)
    pt = torch.where(targets == 1, prob, 1 - prob)
    at = torch.where(targets == 1, alpha, 1 - alpha)
    return at * (1 - pt) ** gamma * bce


def _make_grid(grid_size, stride, device):
    gy, gx = torch.meshgrid(
        torch.arange(grid_size, device=device, dtype=torch.float32),
        torch.arange(grid_size, device=device, dtype=torch.float32),
        indexing='ij',
    )
    cx = (gx + 0.5) * stride   # [H, W]
    cy = (gy + 0.5) * stride
    return cx.unsqueeze(0), cy.unsqueeze(0)   # [1, H, W]


def _decode_distances(raw, cx, cy, apply_exp=True):
    """raw : [B, 4, H, W] → x1, y1, x2, y2  chacun [B, H, W]"""
    if apply_exp:
        d = torch.exp(raw.float().clamp(-16, 16))
    else:
        d = raw.float()
    left, top, right, bottom = d[:, 0], d[:, 1], d[:, 2], d[:, 3]
    x1 = cx - left
    y1 = cy - top
    x2 = cx + right
    y2 = cy + bottom
    return x1, y1, x2, y2


def ciou_loss_flat(px1, py1, px2, py2, gx1, gy1, gx2, gy2):
    """CIoU element-wise, retourne (1 - CIoU) de shape identique."""
    inter_x1 = torch.max(px1, gx1)
    inter_y1 = torch.max(py1, gy1)
    inter_x2 = torch.min(px2, gx2)
    inter_y2 = torch.min(py2, gy2)
    inter = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)

    area_p = (px2 - px1).clamp(0) * (py2 - py1).clamp(0)
    area_g = (gx2 - gx1).clamp(0) * (gy2 - gy1).clamp(0)
    union = area_p + area_g - inter + 1e-7
    iou = inter / union

    cp_x, cp_y = (px1 + px2) / 2, (py1 + py2) / 2
    cg_x, cg_y = (gx1 + gx2) / 2, (gy1 + gy2) / 2
    rho2 = (cp_x - cg_x) ** 2 + (cp_y - cg_y) ** 2

    ex1 = torch.min(px1, gx1)
    ey1 = torch.min(py1, gy1)
    ex2 = torch.max(px2, gx2)
    ey2 = torch.max(py2, gy2)
    c2 = (ex2 - ex1) ** 2 + (ey2 - ey1) ** 2 + 1e-7

    pw = (px2 - px1).clamp(min=1e-7)
    ph = (py2 - py1).clamp(min=1e-7)
    gw = (gx2 - gx1).clamp(min=1e-7)
    gh = (gy2 - gy1).clamp(min=1e-7)
    v = (4 / (math.pi ** 2)) * (torch.atan(gw / gh) - torch.atan(pw / ph)) ** 2
    with torch.no_grad():
        alpha = v / (1 - iou + v + 1e-7)

    return 1 - (iou - rho2 / c2 - alpha * v)


class DetectionLossFPN(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.ls = label_smoothing

    def _single_scale(self, pred, target, level):
        obj_pred, cls_pred, box_pred = pred   # [B,1,H,W], [B,C,H,W], [B,4,H,W]
        obj_target, cls_target, box_target = target  # [B,H,W], [B,H,W], [B,4,H,W]

        device = obj_pred.device
        stride = CELL_SIZES[level]
        gs = GRID_SIZES[level]

        # --- objectness (focal, unreduced sum + count) ---
        fl = focal_loss(obj_pred.squeeze(1), obj_target.float())
        obj_sum = fl.sum()
        num_cells = fl.numel()

        mask = (obj_target > 0)            # [B, H, W]
        num_pos = mask.sum()

        # --- classification (BCE logits, masked) ---
        cls_pred_p = cls_pred.permute(0, 2, 3, 1)     # [B, H, W, C]
        cls_oh = F.one_hot(cls_target.long(), self.num_classes).float()
        if self.ls > 0:
            cls_oh = cls_oh * (1 - self.ls) + self.ls / self.num_classes
        cls_bce = F.binary_cross_entropy_with_logits(cls_pred_p, cls_oh, reduction='none')
        cls_sum = (cls_bce.sum(dim=-1) * mask.float()).sum()

        # --- box (CIoU sur boxes décodées, forcé float32) ---
        cx, cy = _make_grid(gs, stride, device)
        px1, py1, px2, py2 = _decode_distances(box_pred, cx, cy, apply_exp=True)
        gx1, gy1, gx2, gy2 = _decode_distances(box_target, cx, cy, apply_exp=False)
        ciou = ciou_loss_flat(px1, py1, px2, py2, gx1, gy1, gx2, gy2)
        box_sum = (ciou * mask.float()).sum()

        return obj_sum, num_cells, cls_sum, box_sum, num_pos

    def forward(self, predictions, targets):
        total_obj, total_cells = torch.tensor(0.0, device=predictions[0][0].device), 0
        total_cls = torch.tensor(0.0, device=predictions[0][0].device)
        total_box = torch.tensor(0.0, device=predictions[0][0].device)
        total_pos = torch.tensor(0.0, device=predictions[0][0].device)

        for level in range(3):
            o_s, nc, c_s, b_s, np_ = self._single_scale(
                predictions[level], targets[level], level,
            )
            total_obj = total_obj + o_s
            total_cells += nc
            total_cls = total_cls + c_s
            total_box = total_box + b_s
            total_pos = total_pos + np_

        obj_loss = total_obj / max(total_cells, 1)
        total_pos = total_pos.clamp(min=1)
        cls_loss = total_cls / total_pos
        box_loss = total_box / total_pos

        total_loss = 0.5 * obj_loss + 0.5 * cls_loss + 7.5 * box_loss
        return obj_loss, cls_loss, box_loss, total_loss
