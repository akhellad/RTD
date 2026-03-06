import os
import json
import random
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A

from config_v2 import IMAGE_SIZE, CELL_SIZES, GRID_SIZES, SCALE_RANGES


def _load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


class COCODatasetFPNv2(Dataset):
    def __init__(self, annotation_file, image_dir, train=True):
        with open(annotation_file) as f:
            raw = json.load(f)

        self.image_dir = image_dir
        self.train = train

        self.category_id_to_label = {}
        self.label_to_category_id = {}
        for i, cat in enumerate(raw['categories']):
            self.category_id_to_label[cat['id']] = i
            self.label_to_category_id[i] = cat['id']

        annotations = {}
        for ann in raw['annotations']:
            if ann['bbox'][2] > 0 and ann['bbox'][3] > 0:
                annotations.setdefault(ann['image_id'], []).append(
                    (ann['bbox'], self.category_id_to_label[ann['category_id']])
                )

        self.data = []
        for img_info in raw['images']:
            anns = annotations.get(img_info['id'], [])
            if anns:
                self.data.append({
                    'file_name': img_info['file_name'],
                    'width': img_info['width'],
                    'height': img_info['height'],
                    'annotations': anns,
                })

        if train:
            self.spatial_transform = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.Affine(scale=(0.8, 1.2), translate_percent=(-0.1, 0.1), p=0.4),
            ], bbox_params=A.BboxParams(
                format='coco', label_fields=['labels'], min_area=1, min_visibility=0.1,
            ))
            self.color_transform = A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3, p=0.5),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=40, val_shift_limit=30, p=0.5),
                A.GaussianBlur(p=0.2),
                A.CoarseDropout(
                    max_holes=4, max_height=40, max_width=40,
                    fill_value=114, p=0.3,
                ),
            ])
        else:
            self.spatial_transform = None
            self.color_transform = None

    def __len__(self):
        return len(self.data)

    # ------------------------------------------------------------------
    #  Chargement d'une image + boxes
    # ------------------------------------------------------------------
    def _load_entry(self, index):
        entry = self.data[index]
        img = _load_image(os.path.join(self.image_dir, entry['file_name']))
        oh, ow = img.shape[:2]
        bboxes, labels = [], []
        for (x, y, w, h), lbl in entry['annotations']:
            if w > 1 and h > 1:
                bboxes.append([x, y, w, h])
                labels.append(lbl)
        return img, bboxes, labels, ow, oh

    # ------------------------------------------------------------------
    #  Mosaic 2×2
    # ------------------------------------------------------------------
    def _mosaic(self, index):
        s = IMAGE_SIZE
        half = s // 2
        indices = [index] + random.choices(range(len(self.data)), k=3)
        mosaic = np.full((s, s, 3), 114, dtype=np.uint8)
        all_boxes, all_labels = [], []

        placements = [
            (0, 0, half, half),
            (half, 0, s, half),
            (0, half, half, s),
            (half, half, s, s),
        ]
        for i, idx in enumerate(indices):
            img, bboxes, labels, ow, oh = self._load_entry(idx)
            img_r = cv2.resize(img, (half, half))
            sx, sy = half / ow, half / oh
            x1p, y1p, x2p, y2p = placements[i]
            mosaic[y1p:y2p, x1p:x2p] = img_r

            for (bx, by, bw, bh), lbl in zip(bboxes, labels):
                nx, ny = bx * sx + x1p, by * sy + y1p
                nw, nh = bw * sx, bh * sy
                # clip au quadrant
                clip_x2 = min(nx + nw, x2p)
                clip_y2 = min(ny + nh, y2p)
                nx = max(nx, x1p)
                ny = max(ny, y1p)
                nw = clip_x2 - nx
                nh = clip_y2 - ny
                if nw > 2 and nh > 2:
                    all_boxes.append([nx, ny, nw, nh])
                    all_labels.append(lbl)

        return mosaic, all_boxes, all_labels

    # ------------------------------------------------------------------
    #  MixUp
    # ------------------------------------------------------------------
    def _mixup(self, img, bboxes, labels):
        idx2 = random.randint(0, len(self.data) - 1)
        img2, bboxes2, labels2, ow2, oh2 = self._load_entry(idx2)
        img2 = cv2.resize(img2, (IMAGE_SIZE, IMAGE_SIZE))
        sx, sy = IMAGE_SIZE / ow2, IMAGE_SIZE / oh2
        bboxes2 = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in bboxes2]

        ratio = float(np.random.beta(8.0, 8.0))
        img = (img.astype(np.float32) * ratio + img2.astype(np.float32) * (1 - ratio))
        img = np.clip(img, 0, 255).astype(np.uint8)
        bboxes = bboxes + bboxes2
        labels = labels + labels2
        return img, bboxes, labels

    # ------------------------------------------------------------------
    #  Target encoding (anchor-free l/t/r/b)
    # ------------------------------------------------------------------
    def _build_targets(self, bboxes, labels):
        targets = []
        for gs in GRID_SIZES:
            targets.append((
                torch.zeros(gs, gs),
                torch.zeros(gs, gs),
                torch.zeros(4, gs, gs),
            ))

        # Trier par aire décroissante : les plus petits sont traités en dernier
        # et écrasent les plus grands si conflit de cellule → on garde le plus petit
        order = sorted(range(len(bboxes)),
                       key=lambda k: bboxes[k][2] * bboxes[k][3], reverse=True)

        for k in order:
            x, y, w, h = [float(v) for v in bboxes[k]]
            lbl = int(labels[k])
            if w <= 0 or h <= 0:
                continue
            cx = x + w / 2
            cy = y + h / 2
            max_side = max(w, h)

            level = None
            for l_idx, (smin, smax) in enumerate(SCALE_RANGES):
                if smin <= max_side < smax:
                    level = l_idx
                    break
            if level is None:
                continue

            stride = CELL_SIZES[level]
            gs = GRID_SIZES[level]
            cell_x = min(int(cx / stride), gs - 1)
            cell_y = min(int(cy / stride), gs - 1)

            cell_cx = (cell_x + 0.5) * stride
            cell_cy = (cell_y + 0.5) * stride

            left = cell_cx - x
            top = cell_cy - y
            right = (x + w) - cell_cx
            bottom = (y + h) - cell_cy

            if left <= 0 or top <= 0 or right <= 0 or bottom <= 0:
                continue

            obj_g, cls_g, box_g = targets[level]
            obj_g[cell_y, cell_x] = 1.0
            cls_g[cell_y, cell_x] = lbl
            box_g[:, cell_y, cell_x] = torch.tensor([left, top, right, bottom], dtype=torch.float32)

        return tuple(targets)

    # ------------------------------------------------------------------
    #  __getitem__
    # ------------------------------------------------------------------
    def __getitem__(self, index):
        s = IMAGE_SIZE

        use_mosaic = self.train and random.random() < 0.8

        if use_mosaic:
            img, bboxes, labels = self._mosaic(index)
        else:
            img, bboxes, labels, ow, oh = self._load_entry(index)
            sx, sy = s / ow, s / oh
            img = cv2.resize(img, (s, s))
            bboxes = [[b[0]*sx, b[1]*sy, b[2]*sx, b[3]*sy] for b in bboxes]

        if self.train and random.random() < 0.15 and len(bboxes) > 0:
            img, bboxes, labels = self._mixup(img, bboxes, labels)

        if self.spatial_transform and len(bboxes) > 0:
            t = self.spatial_transform(image=img, bboxes=bboxes, labels=labels)
            img, bboxes, labels = t['image'], list(t['bboxes']), list(t['labels'])

        if self.color_transform and self.train:
            img = self.color_transform(image=img)['image']

        # Taille finale
        if img.shape[0] != s or img.shape[1] != s:
            img = cv2.resize(img, (s, s))

        empty = tuple(
            (torch.zeros(gs, gs), torch.zeros(gs, gs), torch.zeros(4, gs, gs))
            for gs in GRID_SIZES
        )

        if len(bboxes) == 0:
            img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
            return img_t, *empty

        targets = self._build_targets(bboxes, labels)
        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float() / 255.0
        return img_t, *targets
