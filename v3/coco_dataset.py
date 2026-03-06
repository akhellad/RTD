import os
import json
import random
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import albumentations as A


class MosaicTransform:
    def __init__(self, dataset, target_size=416, p=0.5):
        self.dataset = dataset
        self.target_size = target_size
        self.p = p

    def __call__(self, index):
        if random.random() > self.p:
            return None

        indices = [index] + random.choices(range(len(self.dataset.data)), k=3)
        half = self.target_size // 2
        mosaic_img = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        all_boxes = []
        all_labels = []

        placements = [
            (0, 0, half, half),
            (half, 0, self.target_size, half),
            (0, half, half, self.target_size),
            (half, half, self.target_size, self.target_size),
        ]

        for i, idx in enumerate(indices):
            entry = self.dataset.data[idx]
            img_path = os.path.join(self.dataset.image_dir, entry['file_name'])
            img = plt.imread(img_path)
            if len(img.shape) == 2:
                img = np.stack([img, img, img], axis=2)
            elif img.shape[2] == 4:
                img = img[:, :, :3]

            oh, ow = img.shape[:2]
            img_resized = cv2.resize(img, (half, half))
            sx, sy = half / ow, half / oh

            x1p, y1p, x2p, y2p = placements[i]
            mosaic_img[y1p:y2p, x1p:x2p] = img_resized

            for bbox, lbl in entry['annotations']:
                bx, by, bw, bh = bbox
                if bw <= 1 or bh <= 1:
                    continue
                nx = bx * sx + x1p
                ny = by * sy + y1p
                nw = bw * sx
                nh = bh * sy

                nx = max(0, nx)
                ny = max(0, ny)
                if nx + nw > x2p:
                    nw = x2p - nx
                if ny + nh > y2p:
                    nh = y2p - ny

                if nw > 2 and nh > 2:
                    all_boxes.append([nx, ny, nw, nh])
                    all_labels.append(lbl)

        return mosaic_img, all_boxes, all_labels


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
            annotations[annotation['image_id']].append(
                (annotation['bbox'], self.category_id_to_label[annotation['category_id']])
            )
        for image in self.labels_file['images']:
            image_id = image['id']
            bbox_label = annotations.get(image_id, [])
            if bbox_label:
                self.data.append({'file_name': image['file_name'], 'annotations': bbox_label})
        self.anchors = anchors
        self.all_anchors = [(w, h) for level in anchors for w, h in level]
        self.train = train
        self.mosaic = MosaicTransform(self, target_size=416, p=0.5) if train else None

        if train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.GaussianBlur(p=0.2),
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels'], min_area=1, min_visibility=0.1))
        else:
            self.transform = None

    def _anchor_iou(self, gt_w, gt_h):
        ious = []
        for anchor_w, anchor_h in self.all_anchors:
            inter = min(gt_w, anchor_w) * min(gt_h, anchor_h)
            union = gt_w * gt_h + anchor_w * anchor_h - inter
            ious.append(inter / (union + 1e-6))
        return ious

    def __len__(self):
        return len(self.data)

    def _build_targets(self, bbox, label):
        target_resize = 416

        objectness_grid_P5 = torch.zeros(3, 13, 13)
        class_grid_P5 = torch.zeros(3, 13, 13)
        box_grid_P5 = torch.zeros(3, 4, 13, 13)

        objectness_grid_P4 = torch.zeros(3, 26, 26)
        class_grid_P4 = torch.zeros(3, 26, 26)
        box_grid_P4 = torch.zeros(3, 4, 26, 26)

        objectness_grid_P3 = torch.zeros(3, 52, 52)
        class_grid_P3 = torch.zeros(3, 52, 52)
        box_grid_P3 = torch.zeros(3, 4, 52, 52)

        cell_sizes = [8, 16, 32]
        grid_sizes = [52, 26, 13]
        obj_grids = [objectness_grid_P3, objectness_grid_P4, objectness_grid_P5]
        cls_grids = [class_grid_P3, class_grid_P4, class_grid_P5]
        box_grids = [box_grid_P3, box_grid_P4, box_grid_P5]

        for i, box in enumerate(bbox):
            x, y, w, h = float(box[0]), float(box[1]), float(box[2]), float(box[3])
            if w <= 0 or h <= 0:
                continue

            center_x = x + w / 2
            center_y = y + h / 2

            ious = self._anchor_iou(w, h)
            best_anchor_idx = int(np.argmax(ious))
            assigned = set()
            assigned.add(best_anchor_idx)
            for ai, iou_val in enumerate(ious):
                if iou_val > 0.5:
                    assigned.add(ai)

            for anchor_idx in assigned:
                level = anchor_idx // 3
                anchor_in_level = anchor_idx % 3

                cell_size = cell_sizes[level]
                grid_size = grid_sizes[level]
                anchor_w, anchor_h = self.anchors[level][anchor_in_level]

                cell_x = min(int(center_x // cell_size), grid_size - 1)
                cell_y = min(int(center_y // cell_size), grid_size - 1)

                if obj_grids[level][anchor_in_level, cell_y, cell_x] == 1:
                    continue

                tx = (center_x / cell_size) - cell_x
                ty = (center_y / cell_size) - cell_y
                tw = np.log(w / anchor_w + 1e-6)
                th = np.log(h / anchor_h + 1e-6)

                obj_grids[level][anchor_in_level, cell_y, cell_x] = 1
                cls_grids[level][anchor_in_level, cell_y, cell_x] = label[i]
                box_grids[level][anchor_in_level, :, cell_y, cell_x] = torch.tensor(
                    [tx, ty, tw, th], dtype=torch.float32
                )

        return (objectness_grid_P3, class_grid_P3, box_grid_P3), \
               (objectness_grid_P4, class_grid_P4, box_grid_P4), \
               (objectness_grid_P5, class_grid_P5, box_grid_P5)

    def __getitem__(self, index):
        target_resize = 416

        mosaic_result = self.mosaic(index) if self.mosaic else None

        if mosaic_result is not None:
            img, bbox, label = mosaic_result
            if self.transform and len(bbox) > 0:
                transformed = self.transform(image=img, bboxes=bbox, labels=label)
                img = transformed['image']
                bbox = transformed['bboxes']
                label = transformed['labels']
        else:
            img_path = os.path.join(self.image_dir, self.data[index]['file_name'])
            bbox = [b for b, _ in self.data[index]['annotations']]
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

            valid_boxes, valid_labels = [], []
            for i, box in enumerate(bbox):
                _, _, w, h = box
                if w > 1 and h > 1:
                    valid_boxes.append(box)
                    valid_labels.append(label[i])
            bbox = valid_boxes
            label = valid_labels

            if self.transform and len(bbox) > 0:
                transformed = self.transform(image=img, bboxes=bbox, labels=label)
                img = transformed['image']
                bbox = transformed['bboxes']
                label = transformed['labels']

            img = cv2.resize(img, (target_resize, target_resize))
            scale_tensor = [x_scale, y_scale, x_scale, y_scale]
            bbox = [[b[0] * scale_tensor[0], b[1] * scale_tensor[1],
                     b[2] * scale_tensor[2], b[3] * scale_tensor[3]] for b in bbox]

        if mosaic_result is None:
            pass
        else:
            img = cv2.resize(img, (target_resize, target_resize)) if img.shape[:2] != (target_resize, target_resize) else img

        empty_p3 = (torch.zeros(3, 52, 52), torch.zeros(3, 52, 52), torch.zeros(3, 4, 52, 52))
        empty_p4 = (torch.zeros(3, 26, 26), torch.zeros(3, 26, 26), torch.zeros(3, 4, 26, 26))
        empty_p5 = (torch.zeros(3, 13, 13), torch.zeros(3, 13, 13), torch.zeros(3, 4, 13, 13))

        if len(bbox) == 0:
            img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255
            return img, empty_p3, empty_p4, empty_p5

        label_tensor = [int(l) for l in label]
        targets_p3, targets_p4, targets_p5 = self._build_targets(bbox, label_tensor)

        img = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255
        return img, targets_p3, targets_p4, targets_p5
