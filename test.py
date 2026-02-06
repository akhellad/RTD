import torch
from coco_dataset_FPN import COCODatasetFPN
from modelFPN import ObjectDetectorFPN, DetectionLossFPN
from torch.utils.data import DataLoader

# 1. Crée le dataset et dataloader
dataset = COCODatasetFPN('train_labels.json', 'images', train=True)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

# 2. Crée le modèle et la loss
model = ObjectDetectorFPN(num_classes=80)
criterion = DetectionLossFPN(num_classes=80)

# 3. Prends un batch
images, targets_p3, targets_p4, targets_p5 = next(iter(loader))

print(f"Images shape: {images.shape}")
print(f"Targets P3: {[t.shape for t in targets_p3]}")
print(f"Targets P4: {[t.shape for t in targets_p4]}")
print(f"Targets P5: {[t.shape for t in targets_p5]}")

# 4. Forward pass
predictions = model(images)
print(f"\nPredictions P3: {[p.shape for p in predictions[0]]}")
print(f"Predictions P4: {[p.shape for p in predictions[1]]}")
print(f"Predictions P5: {[p.shape for p in predictions[2]]}")

# 5. Calcule la loss
targets = (targets_p3, targets_p4, targets_p5)
obj_loss, cls_loss, box_loss, total_loss = criterion(predictions, targets)

print(f"\nObj loss: {obj_loss.item():.4f}")
print(f"Class loss: {cls_loss.item():.4f}")
print(f"Box loss: {box_loss.item():.4f}")
print(f"Total loss: {total_loss.item():.4f}")

print("\n✅ Tout fonctionne !")