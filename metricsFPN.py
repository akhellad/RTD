import numpy as np

class DetectionMetricsFPN:
    def __init__(self, num_classes, anchors, iou_threshold=0.5):
        self.num_classes   = num_classes
        self.iou_threshold = iou_threshold
        self.anchors       = anchors   # [[3 anchors P3], [3 anchors P4], [3 anchors P5]]

    def _decode_single_scale(self, scale_pred, grid_size, cell_size, level_anchors):
        obj_pred, class_pred, box_pred = scale_pred
        # obj_pred   : (B, 3, H, W)
        # class_pred : (B, 3, num_classes, H, W)
        # box_pred   : (B, 3, 4, H, W)  — tx/ty déjà sigmoïdés, tw/th bruts

        batch     = obj_pred.shape[0]
        all_boxes = []

        for i in range(batch):
            boxes = []
            for a, (anchor_w, anchor_h) in enumerate(level_anchors):
                for cell_y in range(grid_size):       # dimension 0 de la grille = y (lignes)
                    for cell_x in range(grid_size):   # dimension 1 de la grille = x (colonnes)
                        objectness = obj_pred[i, a, cell_y, cell_x].item()
                        if objectness < 0.2:
                            continue

                        class_scores = class_pred[i, a, :, cell_y, cell_x].cpu().numpy()
                        class_id     = int(np.argmax(class_scores))
                        class_prob   = float(class_scores[class_id])

                        tx = box_pred[i, a, 0, cell_y, cell_x].item()   # déjà sigmoïdé
                        ty = box_pred[i, a, 1, cell_y, cell_x].item()
                        tw = box_pred[i, a, 2, cell_y, cell_x].item()   # brut (log-space)
                        th = box_pred[i, a, 3, cell_y, cell_x].item()

                        # Décodage inverse de l'encodage du dataset
                        center_x = (cell_x + tx) * cell_size
                        center_y = (cell_y + ty) * cell_size
                        width    = anchor_w * np.exp(np.clip(tw, -4, 4))
                        height   = anchor_h * np.exp(np.clip(th, -4, 4))

                        x1 = center_x - width  / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width  / 2
                        y2 = center_y + height / 2

                        confidence = objectness * class_prob
                        boxes.append([x1, y1, x2, y2, confidence, class_id])

            all_boxes.append(boxes)
        return all_boxes

    def decode_prediction_fpn(self, predictions):
        boxes_p3   = self._decode_single_scale(predictions[0], 28, 8,  self.anchors[0])
        boxes_p4   = self._decode_single_scale(predictions[1], 14, 16, self.anchors[1])
        boxes_p5   = self._decode_single_scale(predictions[2], 7,  32, self.anchors[2])
        batch_size = predictions[0][0].shape[0]
        return [boxes_p3[i] + boxes_p4[i] + boxes_p5[i] for i in range(batch_size)]

    def _extract_single_scale(self, scale_target, grid_size, cell_size, level_anchors):
        obj_grid, class_grid, box_grid = scale_target
        # obj_grid   : (B, 3, H, W)
        # class_grid : (B, 3, H, W)
        # box_grid   : (B, 3, 4, H, W)

        batch     = obj_grid.shape[0]
        all_boxes = []

        for i in range(batch):
            boxes = []
            for a, (anchor_w, anchor_h) in enumerate(level_anchors):
                for cell_y in range(grid_size):
                    for cell_x in range(grid_size):
                        if obj_grid[i, a, cell_y, cell_x].item() != 1:
                            continue

                        class_id = int(class_grid[i, a, cell_y, cell_x].item())
                        tx = box_grid[i, a, 0, cell_y, cell_x].item()
                        ty = box_grid[i, a, 1, cell_y, cell_x].item()
                        tw = box_grid[i, a, 2, cell_y, cell_x].item()
                        th = box_grid[i, a, 3, cell_y, cell_x].item()

                        center_x = (cell_x + tx) * cell_size
                        center_y = (cell_y + ty) * cell_size
                        width    = anchor_w * np.exp(np.clip(tw, -4, 4))
                        height   = anchor_h * np.exp(np.clip(th, -4, 4))

                        x1 = center_x - width  / 2
                        y1 = center_y - height / 2
                        x2 = center_x + width  / 2
                        y2 = center_y + height / 2

                        boxes.append([x1, y1, x2, y2, class_id])

            all_boxes.append(boxes)
        return all_boxes

    def extract_gt_boxes_fpn(self, targets):
        boxes_p3   = self._extract_single_scale(targets[0], 28, 8,  self.anchors[0])
        boxes_p4   = self._extract_single_scale(targets[1], 14, 16, self.anchors[1])
        boxes_p5   = self._extract_single_scale(targets[2], 7,  32, self.anchors[2])
        batch_size = targets[0][0].shape[0]
        return [boxes_p3[i] + boxes_p4[i] + boxes_p5[i] for i in range(batch_size)]

    def calculate_iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        inter  = max(0, xB - xA) * max(0, yB - yA)
        areaA  = (box1[2] - box1[0]) * (box1[3] - box1[1])
        areaB  = (box2[2] - box2[0]) * (box2[3] - box2[1])
        return inter / (areaA + areaB - inter + 1e-6)

    def compute_nms(self, boxes, iou_threshold=0.5):
        if len(boxes) == 0:
            return []
        box_by_class = {}
        for box in boxes:
            box_by_class.setdefault(box[5], []).append(box)
        result = []
        for cls_boxes in box_by_class.values():
            cls_boxes = sorted(cls_boxes, key=lambda x: x[4], reverse=True)
            while cls_boxes:
                best = cls_boxes.pop(0)
                result.append(best)
                cls_boxes = [b for b in cls_boxes if self.calculate_iou(best, b) <= iou_threshold]
        return result

    def compute_map(self, pred_boxes, gt_boxes):
        pred_boxes = self.decode_prediction_fpn(pred_boxes)
        gt_boxes   = self.extract_gt_boxes_fpn(gt_boxes)

        results_by_class = {c: {'preds': [], 'num_gts': 0} for c in range(self.num_classes)}

        for preds, gts in zip(pred_boxes, gt_boxes):
            matched_gts = set()
            for pred in preds:
                pred_class = pred[5]
                confidence = pred[4]
                best_iou, best_gt_idx = 0, -1
                for gt_idx, gt in enumerate(gts):
                    if gt[4] == pred_class:
                        iou = self.calculate_iou(pred, gt)
                        if iou > best_iou:
                            best_iou, best_gt_idx = iou, gt_idx
                is_tp = best_iou >= 0.5 and best_gt_idx not in matched_gts
                if is_tp:
                    matched_gts.add(best_gt_idx)
                results_by_class[pred_class]['preds'].append((confidence, is_tp))
            for gt in gts:
                results_by_class[gt[4]]['num_gts'] += 1

        aps = []
        for class_id in range(self.num_classes):
            preds    = results_by_class[class_id]['preds']
            num_gts  = results_by_class[class_id]['num_gts']
            if num_gts == 0:
                continue
            if not preds:
                aps.append(0.0)
                continue
            preds        = sorted(preds, key=lambda x: x[0], reverse=True)
            tp_cum = fp_cum = 0
            precisions, recalls = [], []
            for _, is_tp in preds:
                if is_tp:
                    tp_cum += 1
                else:
                    fp_cum += 1
                precisions.append(tp_cum / (tp_cum + fp_cum))
                recalls.append(tp_cum / num_gts)
            ap = sum(
                max((p for p, r in zip(precisions, recalls) if r >= thr), default=0)
                for thr in np.arange(0, 1.1, 0.1)
            ) / 11
            aps.append(ap)

        return sum(aps) / len(aps) if aps else 0
