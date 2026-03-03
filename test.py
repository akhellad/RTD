import torch
from coco_dataset_FPN import COCODatasetFPN
from modelFPN_ResNet import ObjectDetectorFPN, DetectionLossFPN
from metricsFPN import DetectionMetricsFPN
from torch.utils.data import DataLoader
from config import ANCHORS

B = 2  # batch size réduit pour le test

print("=" * 55)
print("1. Dataset & DataLoader")
print("=" * 55)
dataset = COCODatasetFPN('instances_train2017.json', 'images', ANCHORS, train=True)
loader  = DataLoader(dataset, batch_size=B, shuffle=False)

images, targets_p3, targets_p4, targets_p5 = next(iter(loader))

print(f"Images            : {images.shape}")           # (B, 3, 416, 416)
print(f"Targets P3 obj    : {targets_p3[0].shape}")    # (B, 3, 28, 28)
print(f"Targets P3 cls    : {targets_p3[1].shape}")    # (B, 3, 28, 28)
print(f"Targets P3 box    : {targets_p3[2].shape}")    # (B, 3, 4, 28, 28)
print(f"Targets P4 obj    : {targets_p4[0].shape}")    # (B, 3, 14, 14)
print(f"Targets P5 obj    : {targets_p5[0].shape}")    # (B, 3, 7, 7)

# Vérifier qu'il y a bien des objets encodés dans le batch
total_objects = targets_p3[0].sum() + targets_p4[0].sum() + targets_p5[0].sum()
print(f"Objets GT encodés : {int(total_objects.item())}")
assert total_objects > 0, "ERREUR : aucun objet GT dans le batch !"

print("\n" + "=" * 55)
print("2. Modele — forward pass")
print("=" * 55)
model     = ObjectDetectorFPN(num_classes=80)
criterion = DetectionLossFPN(num_classes=80)

predictions = model(images)

obj3, cls3, box3 = predictions[0]
print(f"Pred P3 obj : {obj3.shape}")    # (B, 3, 28, 28)
print(f"Pred P3 cls : {cls3.shape}")    # (B, 3, 80, 28, 28)
print(f"Pred P3 box : {box3.shape}")    # (B, 3, 4, 28, 28)
print(f"Pred P4 obj : {predictions[1][0].shape}")
print(f"Pred P5 obj : {predictions[2][0].shape}")

# tx, ty doivent être en [0, 1] car sigmoid appliqué dans la tête
tx_min, tx_max = box3[:, :, 0].min().item(), box3[:, :, 0].max().item()
print(f"tx range    : [{tx_min:.3f}, {tx_max:.3f}]  (attendu: [0, 1])")
assert 0 <= tx_min and tx_max <= 1, "ERREUR : tx/ty hors de [0, 1] !"

print("\n" + "=" * 55)
print("3. Loss")
print("=" * 55)
targets = (targets_p3, targets_p4, targets_p5)
obj_loss, cls_loss, box_loss, total_loss = criterion(predictions, targets)

print(f"Obj loss   : {obj_loss.item():.4f}")
print(f"Class loss : {cls_loss.item():.4f}")
print(f"Box loss   : {box_loss.item():.4f}")
print(f"Total loss : {total_loss.item():.4f}")
assert not torch.isnan(total_loss), "ERREUR : loss est NaN !"
assert total_loss.item() > 0,       "ERREUR : loss est nulle !"

print("\n" + "=" * 55)
print("4. Backward pass")
print("=" * 55)
total_loss.backward()
# Vérifier que les gradients se propagent bien jusqu'au backbone
grad = model.backbone.layer4[-1].conv3.weight.grad
print(f"Gradient backbone non nul : {grad is not None and grad.abs().sum().item() > 0}")
assert grad is not None, "ERREUR : pas de gradient dans le backbone !"

print("\n" + "=" * 55)
print("5. Decode predictions (metricsFPN)")
print("=" * 55)
metrics = DetectionMetricsFPN(80, ANCHORS)
with torch.no_grad():
    decoded_preds = metrics.decode_prediction_fpn(predictions)
    decoded_gts   = metrics.extract_gt_boxes_fpn(targets)

print(f"Preds décodées batch[0] : {len(decoded_preds[0])} boites")
print(f"GT décodés   batch[0]   : {len(decoded_gts[0])} boites")

if decoded_gts[0]:
    gt = decoded_gts[0][0]
    print(f"Exemple GT              : x1={gt[0]:.1f} y1={gt[1]:.1f} x2={gt[2]:.1f} y2={gt[3]:.1f} cls={int(gt[4])}")

print("\n" + "=" * 55)
print("OK Tout fonctionne correctement !")
print("=" * 55)
