import torch.nn as nn
import torch.nn.functional as F 

class BackBoneFPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bloc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.bloc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.bloc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.bloc4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2)
        )
        self.bloc5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.MaxPool2d(2)
        )

    def forward(self, x):
        x = self.bloc1(x)
        x = self.bloc2(x)
        c3 = self.bloc3(x)
        c4 = self.bloc4(c3) 
        c5 = self.bloc5(c4)
        return c3, c4, c5 

class FPN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(512, 256, 1)
        self.conv2 = nn.Conv2d(512, 256, 1)
        self.conv3 = nn.Conv2d(256, 256, 1)
        self.upsamp = nn.Upsample(scale_factor=2, mode='nearest')
    
    def forward(self, c3, c4, c5):
        p5 = self.conv1(c5)
        p4 = self.upsamp(p5) + self.conv2(c4)
        p3 = self.upsamp(p4) + self.conv3(c3)
        return p3, p4, p5

class DetectionHead(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.objectness = nn.Sequential(
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )
        self.classificator = nn.Sequential(
            nn.Conv2d(256, num_classes, 1),
            nn.Sigmoid()
        )
        self.boxregressor = nn.Conv2d(256, 4, 1)

    def forward(self, x):
        objectness = self.objectness(x)
        classification = self.classificator(x)
        boxes = self.boxregressor(x)
        return (objectness, classification, boxes)

class ObjectDetectorFPN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.backbone = BackBoneFPN()
        self.FPN = FPN()
        self.head_p3 = DetectionHead(num_classes)
        self.head_p4 = DetectionHead(num_classes)
        self.head_p5 = DetectionHead(num_classes)
    
    def forward(self, x):
        c3, c4, c5 = self.backbone(x)
        p3, p4, p5 = self.FPN(c3, c4, c5)
        obj3, cls3, box3 = self.head_p3(p3) 
        obj4, cls4, box4 = self.head_p4(p4)
        obj5, cls5, box5 = self.head_p5(p5)  
        return ((obj3, cls3, box3), (obj4, cls4, box4), (obj5, cls5, box5)) 

class DetectionLossFPN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
    
    def _compute_single_scale_loss(self, predictions, target):
        obj_pred, class_pred, box_pred = predictions
        obj_target, class_target, box_target = target
        obj_pred = obj_pred.squeeze(1)
        object_loss = F.binary_cross_entropy(obj_pred, obj_target.float())
        mask = (obj_target > 0).float()
        num_pos = mask.sum() + 1e-6
        class_pred = class_pred.permute(0, 2, 3, 1) 
        box_pred = box_pred.permute(0, 2, 3, 1)
        box_target = box_target.permute(0, 2, 3, 1)
        class_target_onehot = F.one_hot(class_target.long(), num_classes=self.num_classes).float()
        class_loss_all = F.binary_cross_entropy(class_pred, class_target_onehot, reduction='none')
        class_loss = (class_loss_all.sum(dim=-1) * mask).sum() / num_pos
        box_loss_all = F.smooth_l1_loss(box_pred, box_target, reduction='none')
        box_loss = (box_loss_all.sum(dim=-1) * mask).sum() / num_pos
        
        total_loss = object_loss + class_loss + 5.0 * box_loss
        return object_loss, class_loss, box_loss, total_loss

    def forward(self, predictions, targets):
        obj_p3, cls_p3, box_p3, total_p3 = self._compute_single_scale_loss(predictions[0], targets[0])
        obj_p4, cls_p4, box_p4, total_p4 = self._compute_single_scale_loss(predictions[1], targets[1])
        obj_p5, cls_p5, box_p5, total_p5 = self._compute_single_scale_loss(predictions[2], targets[2])
        obj_loss = obj_p3 + obj_p4 + obj_p5
        class_loss = cls_p3 + cls_p4 + cls_p5
        box_loss = box_p3 + box_p4 + box_p5
        total_loss = total_p3 + total_p4 + total_p5
        
        return obj_loss, class_loss, box_loss, total_loss