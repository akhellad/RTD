import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as F 
import torch
from torch.utils.data import DataLoader
from coco_dataset import COCODataset

class BackBone(nn.Module):
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
        x = self.bloc3(x)
        x = self.bloc4(x)
        x = self.bloc5(x)
        return x 
         
class ObjectDetector(nn.Module):
    def __init__(self, numclasses):
        super().__init__()
        self.backbone = BackBone()
        self.objectness = nn.Sequential(
            nn.Conv2d(512, 1, 1),
            nn.Sigmoid()
        )
        self.classificator = nn.Sequential(
            nn.Conv2d(512, numclasses, 1),
            nn.Sigmoid()
        )
        self.boxregressor = nn.Conv2d(512, 4, 1)

    def forward(self, x):
        x = self.backbone(x)
        objectness = self.objectness(x)
        classification = self.classificator(x)
        boxes = self.boxregressor(x)
        return (objectness, classification, boxes)

class DetectionLoss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, predictions, target):
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