import numpy as np
import torch
import torchvision

from config_v2 import NUM_CLASSES, CELL_SIZES, GRID_SIZES, IMAGE_SIZE


class DetectionMetricsFPNv2:
    def __init__(self, num_classes=NUM_CLASSES, iou_threshold=0.5,
                 conf_threshold=0.001, max_det=300, max_nms=30000):
        self.num_classes = num_classes
        self.iou_threshold = iou_threshold
        self.conf_threshold = conf_threshold
        self.max_det = max_det
        self.max_nms = max_nms

    # ------------------------------------------------------------------
    #  Décodage prédictions anchor-free
    # ------------------------------------------------------------------
    def _decode_single_scale(self, pred, level):
        obj_logits, cls_logits, box_raw = pred  # [B,1,H,W], [B,C,H,W], [B,4,H,W]
        B, _, H, W = obj_logits.shape
        stride = CELL_SIZES[level]

        obj_scores = torch.sigmoid(obj_logits.squeeze(1))   # [B, H, W]
        cls_scores = torch.sigmoid(cls_logits)               # [B, C, H, W]

        gy, gx = torch.meshgrid(
            torch.arange(H, device=obj_logits.device, dtype=torch.float32),
            torch.arange(W, device=obj_logits.device, dtype=torch.float32),
            indexing='ij',
        )
        cx = (gx + 0.5) * stride   # [H, W]
        cy = (gy + 0.5) * stride

        distances = torch.exp(box_raw.float().clamp(-16, 16))
        left   = distances[:, 0]
        top    = distances[:, 1]
        right  = distances[:, 2]
        bottom = distances[:, 3]

        x1 = (cx.unsqueeze(0) - left).reshape(B, -1)
        y1 = (cy.unsqueeze(0) - top).reshape(B, -1)
        x2 = (cx.unsqueeze(0) + right).reshape(B, -1)
        y2 = (cy.unsqueeze(0) + bottom).reshape(B, -1)

        cls_max, cls_ids = cls_scores.max(dim=1)       # [B, H, W]
        confidence = (obj_scores * cls_max).reshape(B, -1)
        cls_ids = cls_ids.reshape(B, -1)

        boxes = torch.stack([x1, y1, x2, y2], dim=2)   # [B, N, 4]
        return boxes, confidence, cls_ids

    def decode_prediction_fpn(self, predictions):
        B = predictions[0][0].shape[0]
        all_boxes, all_confs, all_cls = [], [], []
        for level in range(3):
            boxes, confs, cls_ids = self._decode_single_scale(predictions[level], level)
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
                boxes_i, confs_i, cls_i = boxes_i[topk], confs_i[topk], cls_i[topk]

            max_coord = boxes_i.max() + 1
            offsets = cls_i.float() * max_coord
            nms_boxes = boxes_i + offsets.unsqueeze(1)
            keep = torchvision.ops.nms(nms_boxes, confs_i, 0.45)

            if len(keep) > self.max_det:
                keep = keep[:self.max_det]

            results.append(torch.cat([
                boxes_i[keep],
                confs_i[keep].unsqueeze(1),
                cls_i[keep].float().unsqueeze(1),
            ], dim=1))

        return results

    # ------------------------------------------------------------------
    #  Extraction GT anchor-free
    # ------------------------------------------------------------------
    def _extract_single_scale(self, target, level):
        obj_grid, cls_grid, box_grid = target  # [B,H,W], [B,H,W], [B,4,H,W]
        B, H, W = obj_grid.shape
        stride = CELL_SIZES[level]

        gy, gx = torch.meshgrid(
            torch.arange(H, device=obj_grid.device, dtype=torch.float32),
            torch.arange(W, device=obj_grid.device, dtype=torch.float32),
            indexing='ij',
        )
        cx = (gx + 0.5) * stride
        cy = (gy + 0.5) * stride

        left   = box_grid[:, 0]
        top    = box_grid[:, 1]
        right  = box_grid[:, 2]
        bottom = box_grid[:, 3]

        x1 = cx.unsqueeze(0) - left
        y1 = cy.unsqueeze(0) - top
        x2 = cx.unsqueeze(0) + right
        y2 = cy.unsqueeze(0) + bottom

        per_image = []
        for i in range(B):
            mask = obj_grid[i] == 1
            if mask.sum() == 0:
                per_image.append(torch.zeros(0, 5))
                continue
            per_image.append(torch.stack([
                x1[i][mask], y1[i][mask], x2[i][mask], y2[i][mask],
                cls_grid[i][mask].float(),
            ], dim=1))
        return per_image

    def extract_gt_boxes_fpn(self, targets):
        all_scale = []
        for level in range(3):
            all_scale.append(self._extract_single_scale(targets[level], level))
        B = targets[0][0].shape[0]
        results = []
        for i in range(B):
            parts = [all_scale[l][i] for l in range(3) if all_scale[l][i].shape[0] > 0]
            results.append(torch.cat(parts, dim=0) if parts else torch.zeros(0, 5))
        return results

    # ------------------------------------------------------------------
    #  mAP@0.5  (11-point interpolation)
    # ------------------------------------------------------------------
    def compute_map(self, predictions, targets):
        pred_decoded = self.decode_prediction_fpn(predictions)
        gt_decoded = self.extract_gt_boxes_fpn(targets)
        return self.compute_map_from_decoded(pred_decoded, gt_decoded)

    def compute_map_from_decoded(self, pred_boxes, gt_boxes):
        by_class = {c: {'preds': [], 'num_gts': 0} for c in range(self.num_classes)}

        for preds, gts in zip(pred_boxes, gt_boxes):
            if gts.shape[0] == 0:
                if preds.shape[0] > 0:
                    for j in range(preds.shape[0]):
                        cid = int(preds[j, 5].item())
                        by_class[cid]['preds'].append((preds[j, 4].item(), False))
                continue

            gt_np = gts.cpu().numpy()
            matched = set()

            if preds.shape[0] > 0:
                pred_np = preds.cpu().numpy()
                iou_mat = self._iou_matrix(pred_np[:, :4], gt_np[:, :4])
                order = np.argsort(-pred_np[:, 4])
                for pi in order:
                    pcls = int(pred_np[pi, 5])
                    conf = pred_np[pi, 4]
                    gt_mask = gt_np[:, 4].astype(int) == pcls
                    best_iou, best_gi = 0.0, -1
                    if gt_mask.any():
                        masked = iou_mat[pi] * gt_mask
                        best_gi = int(np.argmax(masked))
                        best_iou = masked[best_gi]
                    tp = best_iou >= 0.5 and best_gi not in matched
                    if tp:
                        matched.add(best_gi)
                    by_class[pcls]['preds'].append((conf, tp))

            for gi in range(gt_np.shape[0]):
                by_class[int(gt_np[gi, 4])]['num_gts'] += 1

        recall_thr = np.arange(0, 1.1, 0.1)
        aps = []
        for cid in range(self.num_classes):
            preds = by_class[cid]['preds']
            ng = by_class[cid]['num_gts']
            if ng == 0:
                continue
            if not preds:
                aps.append(0.0)
                continue
            preds.sort(key=lambda x: x[0], reverse=True)
            tp = np.array([int(p[1]) for p in preds])
            tp_cum = np.cumsum(tp)
            fp_cum = np.cumsum(1 - tp)
            prec = tp_cum / (tp_cum + fp_cum)
            rec = tp_cum / ng
            ap = sum(prec[rec >= t].max() if (rec >= t).any() else 0.0 for t in recall_thr) / 11
            aps.append(ap)

        return sum(aps) / len(aps) if aps else 0.0

    @staticmethod
    def _iou_matrix(a, b):
        xa = np.maximum(a[:, 0:1], b[:, 0])
        ya = np.maximum(a[:, 1:2], b[:, 1])
        xb = np.minimum(a[:, 2:3], b[:, 2])
        yb = np.minimum(a[:, 3:4], b[:, 3])
        inter = np.maximum(0, xb - xa) * np.maximum(0, yb - ya)
        aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        return inter / (aa[:, None] + ab[None, :] - inter + 1e-6)
