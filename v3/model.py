import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
import math


class BackBoneResNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        self.conv1   = resnet.conv1
        self.bn1     = resnet.bn1
        self.relu    = resnet.relu
        self.maxpool = resnet.maxpool
        self.layer1  = resnet.layer1
        self.layer2  = resnet.layer2
        self.layer3  = resnet.layer3
        self.layer4  = resnet.layer4

    def forward(self, x):
        x  = self.conv1(x)
        x  = self.bn1(x)
        x  = self.relu(x)
        x  = self.maxpool(x)
        x  = self.layer1(x)
        c3 = self.layer2(x)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        return c3, c4, c5


class FPN_ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.lateral_c5 = nn.Conv2d(2048, 256, 1)
        self.lateral_c4 = nn.Conv2d(1024, 256, 1)
        self.lateral_c3 = nn.Conv2d(512,  256, 1)
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')

    def forward(self, c3, c4, c5):
        p5 = self.lateral_c5(c5)
        p4 = self.upsamp(p5) + self.lateral_c4(c4)
        p3 = self.upsamp(p4) + self.lateral_c3(c3)
        return p3, p4, p5


class DetectionHead(nn.Module):
    def __init__(self, num_classes, num_anchors=3):
        super().__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors

        self.shared = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )

        self.objectness = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_anchors, 1),
        )

        self.classificator = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_anchors * num_classes, 1),
        )

        self.boxregressor = nn.Sequential(
            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, num_anchors * 4, 1),
        )

    def forward(self, x):
        x = self.shared(x)
        B, _, H, W = x.shape

        objectness = self.objectness(x)

        cls_raw = self.classificator(x)
        classification = cls_raw.reshape(B, self.num_anchors, self.num_classes, H, W)

        box_raw = self.boxregressor(x).reshape(B, self.num_anchors, 4, H, W)
        boxes = torch.cat([
            torch.sigmoid(box_raw[:, :, :2, :, :]),
            box_raw[:, :, 2:, :, :],
        ], dim=2)

        return objectness, classification, boxes


class ObjectDetectorFPN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = BackBoneResNet()
        self.FPN      = FPN_ResNet()
        self.head_p3  = DetectionHead(num_classes)
        self.head_p4  = DetectionHead(num_classes)
        self.head_p5  = DetectionHead(num_classes)

    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.FPN(c3, c4, c5)
        return (self.head_p3(p3), self.head_p4(p4), self.head_p5(p5))


def focal_loss(logits, target, alpha=0.75, gamma=2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
    prob = torch.sigmoid(logits)
    pt = torch.where(target == 1, prob, 1 - prob)
    alpha_t = torch.where(target == 1, alpha, 1 - alpha)
    return alpha_t * (1 - pt) ** gamma * bce


def ciou_loss(pred_box, target_box, anchors_wh, cell_xy, cell_size, mask, reduce=True):
    if mask.sum() == 0:
        return torch.tensor(0.0, device=pred_box.device)

    pred_box = pred_box.float()
    target_box = target_box.float()
    mask = mask.float()

    anchors_w = anchors_wh[:, 0].view(1, -1, 1, 1)
    anchors_h = anchors_wh[:, 1].view(1, -1, 1, 1)
    cx_grid, cy_grid = cell_xy

    pred_cx = (cx_grid + pred_box[:, :, 0]) * cell_size
    pred_cy = (cy_grid + pred_box[:, :, 1]) * cell_size
    pred_w = anchors_w * torch.exp(torch.clamp(pred_box[:, :, 2], -4, 4))
    pred_h = anchors_h * torch.exp(torch.clamp(pred_box[:, :, 3], -4, 4))

    gt_cx = (cx_grid + target_box[:, :, 0]) * cell_size
    gt_cy = (cy_grid + target_box[:, :, 1]) * cell_size
    gt_w = anchors_w * torch.exp(torch.clamp(target_box[:, :, 2], -4, 4))
    gt_h = anchors_h * torch.exp(torch.clamp(target_box[:, :, 3], -4, 4))

    pred_x1 = pred_cx - pred_w / 2
    pred_y1 = pred_cy - pred_h / 2
    pred_x2 = pred_cx + pred_w / 2
    pred_y2 = pred_cy + pred_h / 2

    gt_x1 = gt_cx - gt_w / 2
    gt_y1 = gt_cy - gt_h / 2
    gt_x2 = gt_cx + gt_w / 2
    gt_y2 = gt_cy + gt_h / 2

    inter_x1 = torch.max(pred_x1, gt_x1)
    inter_y1 = torch.max(pred_y1, gt_y1)
    inter_x2 = torch.min(pred_x2, gt_x2)
    inter_y2 = torch.min(pred_y2, gt_y2)
    inter = torch.clamp(inter_x2 - inter_x1, min=0) * torch.clamp(inter_y2 - inter_y1, min=0)

    area_pred = pred_w * pred_h
    area_gt = gt_w * gt_h
    union = area_pred + area_gt - inter + 1e-7
    iou = inter / union

    enclose_x1 = torch.min(pred_x1, gt_x1)
    enclose_y1 = torch.min(pred_y1, gt_y1)
    enclose_x2 = torch.max(pred_x2, gt_x2)
    enclose_y2 = torch.max(pred_y2, gt_y2)
    c2 = (enclose_x2 - enclose_x1) ** 2 + (enclose_y2 - enclose_y1) ** 2 + 1e-7
    rho2 = (pred_cx - gt_cx) ** 2 + (pred_cy - gt_cy) ** 2

    v = (4 / (math.pi ** 2)) * (torch.atan(gt_w / (gt_h + 1e-7)) - torch.atan(pred_w / (pred_h + 1e-7))) ** 2
    with torch.no_grad():
        alpha_ciou = v / (1 - iou + v + 1e-7)

    ciou = iou - rho2 / c2 - alpha_ciou * v
    loss = (1 - ciou) * mask
    if reduce:
        return loss.sum() / (mask.sum() + 1e-6)
    return loss.sum()


class DetectionLossFPN(nn.Module):
    def __init__(self, num_classes, anchors, label_smoothing=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.anchors = anchors
        self.label_smoothing = label_smoothing
        self.cell_sizes = [8, 16, 32]
        self.grid_sizes = [52, 26, 13]

    def _make_grid(self, grid_size, device):
        gy, gx = torch.meshgrid(
            torch.arange(grid_size, device=device, dtype=torch.float32),
            torch.arange(grid_size, device=device, dtype=torch.float32),
            indexing='ij',
        )
        cx = gx.view(1, 1, grid_size, grid_size)
        cy = gy.view(1, 1, grid_size, grid_size)
        return cx, cy

    def _compute_single_scale_loss(self, predictions, target, level):
        obj_pred, class_pred, box_pred = predictions
        obj_target, class_target, box_target = target

        device = obj_pred.device
        grid_size = self.grid_sizes[level]
        cell_size = self.cell_sizes[level]
        anchors_wh = torch.tensor(self.anchors[level], device=device, dtype=torch.float32)

        mask = (obj_target > 0).float()
        num_pos = mask.sum()

        object_loss = focal_loss(obj_pred, obj_target.float()).mean()

        class_pred_perm = class_pred.permute(0, 1, 3, 4, 2)
        class_target_onehot = F.one_hot(class_target.long(), num_classes=self.num_classes).float()
        if self.label_smoothing > 0:
            class_target_onehot = class_target_onehot * (1 - self.label_smoothing) + \
                                  self.label_smoothing / self.num_classes
        class_loss_raw = F.binary_cross_entropy_with_logits(class_pred_perm, class_target_onehot, reduction='none')
        class_loss = (class_loss_raw.sum(dim=-1) * mask).sum()

        cx_grid, cy_grid = self._make_grid(grid_size, device)
        box_loss = ciou_loss(box_pred, box_target, anchors_wh, (cx_grid, cy_grid), cell_size, mask, reduce=False)

        return object_loss, class_loss, box_loss, num_pos

    def forward(self, predictions, targets):
        obj_p3, cls_p3, box_p3, npos_p3 = self._compute_single_scale_loss(predictions[0], targets[0], 0)
        obj_p4, cls_p4, box_p4, npos_p4 = self._compute_single_scale_loss(predictions[1], targets[1], 1)
        obj_p5, cls_p5, box_p5, npos_p5 = self._compute_single_scale_loss(predictions[2], targets[2], 2)

        total_num_pos = (npos_p3 + npos_p4 + npos_p5).clamp(min=1)

        obj_loss = obj_p3 + obj_p4 + obj_p5
        cls_loss = (cls_p3 + cls_p4 + cls_p5) / total_num_pos
        box_loss = (box_p3 + box_p4 + box_p5) / total_num_pos

        total_loss = 0.5 * obj_loss + 0.5 * cls_loss + 7.5 * box_loss
        return obj_loss, cls_loss, box_loss, total_loss
