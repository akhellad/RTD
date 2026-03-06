import numpy as np
import torch
import torchvision


class DetectionMetricsFPN:
    def __init__(self, num_classes, anchors, iou_threshold=0.5, conf_threshold=0.001, max_det=300, max_nms=30000):
        self.num_classes    = num_classes
        self.iou_threshold  = iou_threshold
        self.conf_threshold = conf_threshold
        self.max_det        = max_det
        self.max_nms        = max_nms
        self.anchors        = anchors

    def _decode_single_scale(self, scale_pred, grid_size, cell_size, level_anchors):
        obj_pred, class_pred, box_pred = scale_pred
        B, A, H, W = obj_pred.shape

        obj_scores = torch.sigmoid(obj_pred)
        cls_scores = torch.sigmoid(class_pred)

        anchors_t = torch.tensor(level_anchors, device=obj_pred.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(
            torch.arange(H, device=obj_pred.device, dtype=torch.float32),
            torch.arange(W, device=obj_pred.device, dtype=torch.float32),
            indexing="ij",
        )
        gx = gx.view(1, 1, H, W)
        gy = gy.view(1, 1, H, W)

        tx = box_pred[:, :, 0]
        ty = box_pred[:, :, 1]
        tw = box_pred[:, :, 2].clamp(-4, 4)
        th = box_pred[:, :, 3].clamp(-4, 4)

        cx = (gx + tx) * cell_size
        cy = (gy + ty) * cell_size
        w = anchors_t[:, 0].view(1, A, 1, 1) * torch.exp(tw)
        h = anchors_t[:, 1].view(1, A, 1, 1) * torch.exp(th)

        x1 = (cx - w / 2).reshape(B, -1)
        y1 = (cy - h / 2).reshape(B, -1)
        x2 = (cx + w / 2).reshape(B, -1)
        y2 = (cy + h / 2).reshape(B, -1)

        cls_max_scores, cls_ids = cls_scores.max(dim=2)
        confidence = (obj_scores * cls_max_scores).reshape(B, -1)
        cls_ids = cls_ids.reshape(B, -1)

        boxes = torch.stack([x1, y1, x2, y2], dim=2)
        return boxes, confidence, cls_ids

    def decode_prediction_fpn(self, predictions):
        cell_sizes = [8, 16, 32]
        B = predictions[0][0].shape[0]

        all_boxes, all_confs, all_cls = [], [], []
        for level in range(3):
            grid_size = predictions[level][0].shape[-1]
            boxes, confs, cls_ids = self._decode_single_scale(
                predictions[level], grid_size, cell_sizes[level], self.anchors[level]
            )
            all_boxes.append(boxes)
            all_confs.append(confs)
            all_cls.append(cls_ids)

        all_boxes = torch.cat(all_boxes, dim=1)
        all_confs = torch.cat(all_confs, dim=1)
        all_cls = torch.cat(all_cls, dim=1)

        results = []
        for i in range(B):
            mask = all_confs[i] > self.conf_threshold
            if mask.sum() == 0:
                results.append(torch.zeros(0, 6))
                continue

            boxes_i = all_boxes[i][mask].float()
            confs_i = all_confs[i][mask].float()
            cls_i = all_cls[i][mask]

            if confs_i.shape[0] > self.max_nms:
                topk = confs_i.topk(self.max_nms).indices
                boxes_i = boxes_i[topk]
                confs_i = confs_i[topk]
                cls_i = cls_i[topk]

            max_coordinate = boxes_i.max()
            offsets = cls_i.float() * (max_coordinate + 1)
            nms_boxes = boxes_i + offsets.unsqueeze(1)
            keep = torchvision.ops.nms(nms_boxes, confs_i, 0.45)

            if len(keep) > self.max_det:
                keep = keep[:self.max_det]

            results.append(torch.cat([
                boxes_i[keep], confs_i[keep].unsqueeze(1), cls_i[keep].float().unsqueeze(1)
            ], dim=1))

        return results

    def _extract_single_scale(self, scale_target, grid_size, cell_size, level_anchors):
        obj_grid, class_grid, box_grid = scale_target
        B, A, H, W = obj_grid.shape

        anchors_t = torch.tensor(level_anchors, device=obj_grid.device, dtype=torch.float32)
        gy, gx = torch.meshgrid(
            torch.arange(H, device=obj_grid.device, dtype=torch.float32),
            torch.arange(W, device=obj_grid.device, dtype=torch.float32),
            indexing="ij",
        )
        gx = gx.view(1, 1, H, W)
        gy = gy.view(1, 1, H, W)

        tx = box_grid[:, :, 0]
        ty = box_grid[:, :, 1]
        tw = box_grid[:, :, 2].clamp(-4, 4)
        th = box_grid[:, :, 3].clamp(-4, 4)

        cx = (gx + tx) * cell_size
        cy = (gy + ty) * cell_size
        w = anchors_t[:, 0].view(1, A, 1, 1) * torch.exp(tw)
        h = anchors_t[:, 1].view(1, A, 1, 1) * torch.exp(th)

        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2

        all_boxes = []
        for i in range(B):
            mask = obj_grid[i] == 1
            if mask.sum() == 0:
                all_boxes.append(torch.zeros(0, 5))
                continue
            boxes_i = torch.stack([
                x1[i][mask], y1[i][mask], x2[i][mask], y2[i][mask],
                class_grid[i][mask].float(),
            ], dim=1)
            all_boxes.append(boxes_i)
        return all_boxes

    def extract_gt_boxes_fpn(self, targets):
        cell_sizes = [8, 16, 32]
        all_scale_boxes = []
        for level in range(3):
            grid_size = targets[level][0].shape[-1]
            boxes = self._extract_single_scale(
                targets[level], grid_size, cell_sizes[level], self.anchors[level]
            )
            all_scale_boxes.append(boxes)
        batch_size = targets[0][0].shape[0]
        results = []
        for i in range(batch_size):
            combined = [all_scale_boxes[l][i] for l in range(3)]
            non_empty = [c for c in combined if c.shape[0] > 0]
            if non_empty:
                results.append(torch.cat(non_empty, dim=0))
            else:
                results.append(torch.zeros(0, 5))
        return results

    def compute_map(self, pred_boxes, gt_boxes):
        pred_boxes = self.decode_prediction_fpn(pred_boxes)
        gt_boxes   = self.extract_gt_boxes_fpn(gt_boxes)
        return self.compute_map_from_decoded(pred_boxes, gt_boxes)

    def compute_map_from_decoded(self, pred_boxes, gt_boxes):
        results_by_class = {c: {"preds": [], "num_gts": 0} for c in range(self.num_classes)}

        for preds, gts in zip(pred_boxes, gt_boxes):
            if gts.shape[0] == 0:
                if preds.shape[0] > 0:
                    for j in range(preds.shape[0]):
                        cls_id = int(preds[j, 5].item())
                        results_by_class[cls_id]["preds"].append((preds[j, 4].item(), False))
                continue

            gt_np = gts.cpu().numpy()
            matched_gts = set()

            if preds.shape[0] > 0:
                pred_np = preds.cpu().numpy()
                iou_mat = self._iou_matrix(pred_np[:, :4], gt_np[:, :4])

                order = np.argsort(-pred_np[:, 4])
                for pi in order:
                    pred_class = int(pred_np[pi, 5])
                    confidence = pred_np[pi, 4]
                    gt_class_mask = gt_np[:, 4].astype(int) == pred_class
                    best_iou, best_gt_idx = 0.0, -1
                    if gt_class_mask.any():
                        masked_ious = iou_mat[pi] * gt_class_mask
                        best_gt_idx = int(np.argmax(masked_ious))
                        best_iou = masked_ious[best_gt_idx]
                    is_tp = best_iou >= 0.5 and best_gt_idx not in matched_gts
                    if is_tp:
                        matched_gts.add(best_gt_idx)
                    results_by_class[pred_class]["preds"].append((confidence, is_tp))

            for gi in range(gt_np.shape[0]):
                results_by_class[int(gt_np[gi, 4])]["num_gts"] += 1

        aps = []
        recall_thresholds = np.arange(0, 1.1, 0.1)
        for class_id in range(self.num_classes):
            preds   = results_by_class[class_id]["preds"]
            num_gts = results_by_class[class_id]["num_gts"]
            if num_gts == 0:
                continue
            if not preds:
                aps.append(0.0)
                continue
            preds = sorted(preds, key=lambda x: x[0], reverse=True)
            tp = np.array([int(p[1]) for p in preds])
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(1 - tp)
            precisions = tp_cum / (tp_cum + fp_cum)
            recalls = tp_cum / num_gts
            ap = 0.0
            for thr in recall_thresholds:
                mask = recalls >= thr
                if mask.any():
                    ap += precisions[mask].max()
            aps.append(ap / 11)

        return sum(aps) / len(aps) if aps else 0

    def _iou_matrix(self, boxes_a, boxes_b):
        xa = np.maximum(boxes_a[:, 0:1], boxes_b[:, 0])
        ya = np.maximum(boxes_a[:, 1:2], boxes_b[:, 1])
        xb = np.minimum(boxes_a[:, 2:3], boxes_b[:, 2])
        yb = np.minimum(boxes_a[:, 3:4], boxes_b[:, 3])
        inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
        area_a = (boxes_a[:, 2] - boxes_a[:, 0]) * (boxes_a[:, 3] - boxes_a[:, 1])
        area_b = (boxes_b[:, 2] - boxes_b[:, 0]) * (boxes_b[:, 3] - boxes_b[:, 1])
        union = area_a[:, None] + area_b[None, :] - inter
        return inter / (union + 1e-6)
