import os
import json
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import albumentations as A

class COCODatasetFPN(Dataset):
    def __init__(self, annotation_file, image_dir, anchors, train=True):
        with open(annotation_file) as f:
            self.labels_file = json.load(f)
        self.data = []
        self.image_dir = image_dir
        self.category_id_to_label = {}
        self.label_to_category_id = {}
        for i, category in enumerate(self.labels_file['categories']):
            self.category_id_to_label[category['id']] = i
            self.label_to_category_id[i] = category['id']
        annotations = {}
        for annotation in self.labels_file['annotations']:
            if annotation['image_id'] not in annotations:
                annotations[annotation['image_id']] = []
            annotations[annotation['image_id']].append((annotation['bbox'], self.category_id_to_label[annotation['category_id']]))
        for image in self.labels_file['images']:
            image_id = image['id']
            bbox_label = annotations.get(image_id, [])
            if bbox_label:
                self.data.append({'file_name': image['file_name'], 'annotations': bbox_label})
        self.anchors = anchors
        self.all_anchors = [(w, h) for level in anchors for w, h in level]

        if train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.GaussianBlur(p=0.2)
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
        else:
            self.transform = None

    def _anchor_iou(self, gt_w, gt_h):
        """
        IoU entre une boite GT (w, h) et chacune des 9 anchors,
        en centrant tout à (0,0) — on compare uniquement les formes.
        Retourne une liste de 9 valeurs IoU.
        """
        ious = []
        for anchor_w, anchor_h in self.all_anchors:
            inter = min(gt_w, anchor_w) * min(gt_h, anchor_h)
            union = gt_w * gt_h + anchor_w * anchor_h - inter
            ious.append(inter / (union + 1e-6))
        return ious

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        objectness_grid_P5 = torch.zeros(3, 7, 7)
        class_grid_P5     = torch.zeros(3, 7, 7)
        box_grid_P5       = torch.zeros(3, 4, 7, 7)

        objectness_grid_P4 = torch.zeros(3, 14, 14)
        class_grid_P4     = torch.zeros(3, 14, 14)
        box_grid_P4       = torch.zeros(3, 4, 14, 14)

        objectness_grid_P3 = torch.zeros(3, 28, 28)
        class_grid_P3     = torch.zeros(3, 28, 28)
        box_grid_P3       = torch.zeros(3, 4, 28, 28)

        target_resize = 416
        img_path = os.path.join(self.image_dir, self.data[index]['file_name'])
        bbox  = [b for b, _ in self.data[index]['annotations']]
        label = [l for _, l in self.data[index]['annotations']]

        img = plt.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2)
        elif img.shape[2] == 4:
            img = img[:, :, :3]

        y_ = img.shape[0]
        x_ = img.shape[1]
        y_scale = target_resize / y_
        x_scale = target_resize / x_
        scale_tensor = torch.tensor([x_scale, y_scale, x_scale, y_scale])

        valid_boxes, valid_labels = [], []
        for i, box in enumerate(bbox):
            _, _, w, h = box
            if w > 1 and h > 1:
                valid_boxes.append(box)
                valid_labels.append(label[i])
        bbox  = valid_boxes
        label = valid_labels

        if self.transform:
            transformed = self.transform(image=img, bboxes=bbox, labels=label)
            img   = transformed['image']
            bbox  = transformed['bboxes']
            label = transformed['labels']

        if len(bbox) == 0:
            img = cv2.resize(img, (target_resize, target_resize))
            img = torch.from_numpy(img).permute(2, 0, 1) / 255
            return img, (objectness_grid_P3, class_grid_P3, box_grid_P3), \
                        (objectness_grid_P4, class_grid_P4, box_grid_P4), \
                        (objectness_grid_P5, class_grid_P5, box_grid_P5)

        label = torch.tensor(label)
        img   = cv2.resize(img, (target_resize, target_resize))
        bbox  = torch.tensor(bbox, dtype=torch.float32) * scale_tensor

        cell_sizes = [8,  16, 32]
        grid_sizes = [28, 14, 7]
        obj_grids  = [objectness_grid_P3, objectness_grid_P4, objectness_grid_P5]
        cls_grids  = [class_grid_P3,      class_grid_P4,      class_grid_P5]
        box_grids  = [box_grid_P3,        box_grid_P4,        box_grid_P5]

        for i, box in enumerate(bbox):
            x, y, w, h = box[0].item(), box[1].item(), box[2].item(), box[3].item()
            if w <= 0 or h <= 0:
                continue

            center_x = x + w / 2
            center_y = y + h / 2

            ious = self._anchor_iou(w, h)
            best_anchor_idx  = int(np.argmax(ious))
            level            = best_anchor_idx // 3
            anchor_in_level  = best_anchor_idx % 3

            cell_size = cell_sizes[level]
            grid_size = grid_sizes[level]
            anchor_w, anchor_h = self.anchors[level][anchor_in_level]

            cell_x = min(int(center_x // cell_size), grid_size - 1)
            cell_y = min(int(center_y // cell_size), grid_size - 1)

            tx = (center_x / cell_size) - cell_x
            ty = (center_y / cell_size) - cell_y
            tw = np.log(w / anchor_w + 1e-6)
            th = np.log(h / anchor_h + 1e-6)

            obj_grids[level][anchor_in_level, cell_y, cell_x] = 1
            cls_grids[level][anchor_in_level, cell_y, cell_x] = label[i]
            box_grids[level][anchor_in_level, :, cell_y, cell_x] = torch.tensor(
                [tx, ty, tw, th], dtype=torch.float32
            )

        img = torch.from_numpy(img).permute(2, 0, 1) / 255
        return img, (objectness_grid_P3, class_grid_P3, box_grid_P3), \
                    (objectness_grid_P4, class_grid_P4, box_grid_P4), \
                    (objectness_grid_P5, class_grid_P5, box_grid_P5)
