import os 
import json
from torch.utils.data import Dataset
import torch
import matplotlib.pyplot as plt
import cv2
import numpy as np
import albumentations as A

class COCODatasetFPN(Dataset):
    def __init__(self, annotation_file, image_dir, train=True):
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
        if train:
            self.transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.HueSaturationValue(p=0.3),
                A.GaussianBlur(p=0.2)
                # A.Affine(
                #     translate_percent={"x": (-0.0625, 0.0625), "y": (-0.0625, 0.0625)},
                #     scale=(0.9, 1.1),
                #     rotate=(-15, 15),
                #     p=0.3
                # )
            ], bbox_params=A.BboxParams(format='coco', label_fields=['labels']))
        else:
            self.transform = None

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        objectness_grid_P5, class_grid_P5, box_grid_P5 = torch.zeros(7, 7), torch.zeros(7, 7), torch.zeros(4, 7, 7)
        objectness_grid_P4, class_grid_P4, box_grid_P4 = torch.zeros(14, 14), torch.zeros(14, 14), torch.zeros(4, 14, 14)
        objectness_grid_P3, class_grid_P3, box_grid_P3 = torch.zeros(28, 28), torch.zeros(28, 28), torch.zeros(4, 28, 28)
        target_resize = 224
        img_path = os.path.join(self.image_dir, self.data[index]['file_name'])
        bbox = [bbox for bbox, _ in self.data[index]['annotations']]
        label =  [label for _, label in self.data[index]['annotations']]
        img = plt.imread(img_path)
        if len(img.shape) == 2:
            img = np.stack([img, img, img], axis=2) 
        y_ = img.shape[0]
        x_ = img.shape[1]
        y_scale = target_resize / y_
        x_scale = target_resize / x_
        scale_tensor = torch.tensor([x_scale, y_scale, x_scale, y_scale])
        valid_boxes = []
        valid_labels = []
        for i, box in enumerate(bbox):
            x, y, w, h = box
            if w > 1 and h > 1:
                valid_boxes.append(box)
                valid_labels.append(label[i])
        bbox = valid_boxes
        label = valid_labels
        if self.transform:
            transformed = self.transform(image=img, bboxes=bbox, labels=label)
            img = transformed['image']
            bbox = transformed['bboxes']
            label = transformed['labels']
        if len(bbox) == 0:
            img = cv2.resize(img, (target_resize, target_resize))
            img = torch.from_numpy(img).permute(2, 0, 1) / 255
            return img, (objectness_grid_P3, class_grid_P3, box_grid_P3), (objectness_grid_P4, class_grid_P4, box_grid_P4), (objectness_grid_P5, class_grid_P5, box_grid_P5)
        label = torch.tensor(label)
        img = cv2.resize(img, (target_resize, target_resize))
        bbox = torch.tensor(bbox) * scale_tensor
        for i, box in enumerate(bbox):
            area = box[2] * box[3]
            if area < 64 * 64:
                cell_size = 8
            elif area < 128 * 128:
                cell_size = 16
            else:
                cell_size = 32
            center_x = box[0] + (box[2] / 2)
            center_y = box[1] + (box[3] / 2)
            cell_x = int(center_x // cell_size)
            cell_y = int(center_y // cell_size)
            top_left_x = cell_x * cell_size
            top_left_y = cell_y * cell_size
            dx = (center_x - top_left_x) / cell_size
            dy = (center_y - top_left_y) / cell_size
            if cell_size == 8:
                objectness_grid_P3[cell_y][cell_x] = 1
                class_grid_P3[cell_y][cell_x] = label[i]
                box_grid_P3[:, cell_y, cell_x] = torch.tensor([dx, dy, box[2] / cell_size, box[3] / cell_size])
            elif cell_size == 16:
                objectness_grid_P4[cell_y][cell_x] = 1
                class_grid_P4[cell_y][cell_x] = label[i]
                box_grid_P4[:, cell_y, cell_x] = torch.tensor([dx, dy, box[2] / cell_size, box[3] / cell_size])
            else:
                objectness_grid_P5[cell_y][cell_x] = 1
                class_grid_P5[cell_y][cell_x] = label[i]
                box_grid_P5[:, cell_y, cell_x] = torch.tensor([dx, dy, box[2] / cell_size, box[3] / cell_size])
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1) / 255
        return img, (objectness_grid_P3, class_grid_P3, box_grid_P3), (objectness_grid_P4, class_grid_P4, box_grid_P4), (objectness_grid_P5, class_grid_P5, box_grid_P5)
