import os 
import json
from torch.utils.data import Dataset
import torch
import matplotlib 
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
import numpy as np

class COCODataset(Dataset):
    def __init__(self, annotation_file, image_dir):
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

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        target_resize = 224
        img_path = os.path.join(self.image_dir, self.data[index]['file_name'])
        bbox = [bbox for bbox, _ in self.data[index]['annotations']]
        label =  torch.tensor([label for _, label in self.data[index]['annotations']])
        img = plt.imread(img_path)
        y_ = img.shape[0]
        x_ = img.shape[1]
        y_scale = target_resize / y_
        x_scale = target_resize / x_
        scale_tensor = torch.tensor([x_scale, y_scale, x_scale, y_scale])
        img = cv2.resize(img, (target_resize, target_resize))
        bbox = torch.tensor(bbox) * scale_tensor
        img = torch.from_numpy(img)
        img = img.permute(2, 0, 1) / 255
        return img, bbox, label

dataset = COCODataset('labels.json', 'images')
img, bbox, label = dataset[1458]
print(img.shape, img.dtype)
print(bbox.shape, bbox.dtype)
print(label.shape, label.dtype)
img = img.permute(1, 2, 0)
img = img.numpy()
fig, ax = plt.subplots(1, 1, figsize=(5, 5))
ax.imshow(img)
for box in bbox:
    x, y, width, height = box[0], box[1], box[2], box[3]
    rect = patches.Rectangle((x, y), width, height, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
ax.axis('off')
plt.show()