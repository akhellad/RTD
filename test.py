import json
from coco_dataset import COCODataset

with open('coco2017/annotations/instances_train2017.json') as f:
    file = json.load(f)

with open('coco2017/annotations/instances_val2017.json') as f:
    file2 = json.load(f)

print(len(file['images']))
print(len(file2['images']))

train_dataset = COCODataset('coco2017/annotations/instances_train2017.json', 'coco2017/train2017')
val_dataset = COCODataset('coco2017/annotations/instances_val2017.json', 'coco2017/val2017', train=False)

print(f"Nombre d'images train: {len(train_dataset)}")
print(f"Nombre d'images val: {len(val_dataset)}")