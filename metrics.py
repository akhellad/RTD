import numpy as np

class DetectionMetrics:
    def __init__(self, num_classes, iou_threshold=0.5):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
    
    def decode_prediction(self, obj_pred, class_pred, box_pred):
        batch = obj_pred.shape[0]
        all_boxes = []
        for i in range(batch):
            boxes = []
            obj_grid = obj_pred[i]
            class_grid = class_pred[i]
            box_grid = box_pred[i]
            for j in range(7):
                for k in range(7):
                    objectness = obj_grid[0, j, k]
                    if objectness > 0.2:
                        class_id = np.argmax(class_grid[:, j, k])
                        class_prob = max(class_grid[:, j, k])
                        box = box_grid[:, j, k]
                        center_x_pixels = (j * 32) + (box[0] * 32)
                        center_y_pixels = (k * 32) + (box[1] * 32)
                        width_pixels = box[2] * 32
                        height_pixels = box[3] * 32
                        x1 = center_x_pixels - width_pixels / 2
                        y1 = center_y_pixels - height_pixels / 2
                        x2 = center_x_pixels + width_pixels / 2
                        y2 = center_y_pixels + height_pixels / 2
                        confidence = objectness * class_prob
                        boxes.append([x1.item(), y1.item(), x2.item(), y2.item(), confidence.item(), class_id.item()])
            all_boxes.append(boxes)
        return all_boxes
    
    def extract_gt_boxes(self, obj_grid, class_grid, box_grid):
        batch = obj_grid.shape[0]
        all_boxes = []
        for i in range(batch):
            boxes = []
            obj_g = obj_grid[i]
            class_g = class_grid[i]
            box_g = box_grid[i]
            for j in range(7):
                for k in range(7):
                    objectness = obj_g[j, k]
                    if objectness == 1:
                        class_id = class_g[j, k]
                        box = box_g[:, j, k]
                        center_x_pixels = (j * 32) + (box[0] * 32)
                        center_y_pixels = (k * 32) + (box[1] * 32)
                        width_pixels = box[2] * 32
                        height_pixels = box[3] * 32
                        x1 = center_x_pixels - width_pixels / 2
                        y1 = center_y_pixels - height_pixels / 2
                        x2 = center_x_pixels + width_pixels / 2
                        y2 = center_y_pixels + height_pixels / 2
                        boxes.append([x1.item(), y1.item(), x2.item(), y2.item(), class_id.item()])
            all_boxes.append(boxes)
        return all_boxes
    
    def calculate_iou(self, box1, box2):
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)

        boxAArea = (box1[2] - box1[0]) * (box1[3] - box1[1])
        boxBArea = (box2[2] - box2[0]) * (box2[3] - box2[1])

        union = boxAArea + boxBArea - interArea

        return interArea / union 

    def compute_nms(self, boxes, iou_threshold=0.5):
        if len(boxes) == 0:
            return []
        box_by_class = {}
        for box in boxes:
            if box[5] not in box_by_class:
                box_by_class[box[5]] = []
            box_by_class[box[5]].append(box)
        result = []
        for boxes in box_by_class.values():
            boxes = sorted(boxes, key=lambda x: x[4], reverse=True)
            while len(boxes) > 0:
                best_box = boxes[0]
                result.append(best_box)
                boxes = [box for box in boxes[1:] if self.calculate_iou(best_box, box) <= iou_threshold]
        return result

    def compute_map(self, pred_boxes, gt_boxes):
        obj_preds, class_preds, box_preds = pred_boxes
        pred_boxes = self.decode_prediction(obj_preds, class_preds, box_preds)
        obj_grid, class_grid, box_grid = gt_boxes
        gt_boxes = self.extract_gt_boxes(obj_grid, class_grid, box_grid)
        results_by_class = {}
        for class_id in range(self.num_classes):
            results_by_class[class_id] = {'preds': [], 'num_gts': 0}
        for i in range(len(pred_boxes)):
            preds = pred_boxes[i]
            gts = gt_boxes[i]
            matched_gts = set()
            for pred in preds:
                pred_class = pred[5]
                confidence = pred[4]
                best_iou = 0
                best_gt_idx = -1
                for gt_idx, gt in enumerate(gts):
                    if gt[4] == pred_class:
                        iou = self.calculate_iou(pred, gt)
                        if iou > best_iou:
                            best_iou = iou 
                            best_gt_idx = gt_idx
                is_tp = (best_iou >= 0.5 and best_gt_idx not in matched_gts)
                if is_tp:
                    matched_gts.add(best_gt_idx)
                results_by_class[pred_class]['preds'].append((confidence, is_tp))
            for gt in gts:
                results_by_class[gt[4]]['num_gts'] += 1
        aps = []
        for class_id in range(self.num_classes):
            preds = results_by_class[class_id]['preds']
            num_gts = results_by_class[class_id]['num_gts']
            if num_gts == 0:
                continue
            if len(preds) == 0: 
                aps.append(0.0)
                continue
            preds = sorted(preds, key=lambda x: x[0], reverse=True)
            tp_cumsum = 0
            fp_cumsum = 0
            precisions = []
            recalls = []
            for conf, is_tp in preds:
                if is_tp:
                    tp_cumsum += 1
                else:
                    fp_cumsum += 1
                precision = tp_cumsum / (tp_cumsum + fp_cumsum)
                recall = tp_cumsum / num_gts
                precisions.append(precision)
                recalls.append(recall)
            ap = 0
            for recall_threshold in np.arange(0, 1.1, 0.1):
                precisions_above = [p for p, r in zip(precisions, recalls) if r >= recall_threshold]
                if len(precisions_above) > 0:
                    ap += max(precisions_above)

            ap = ap / 11
            aps.append(ap)
        map_score = sum(aps) / len(aps) if len(aps) > 0 else 0
        return map_score



