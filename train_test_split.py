import json 
import os
with open('labels.json') as f:
    labels = json.load(f)

train_len = int(len(labels['images']) - (len(labels['images']) * 0.2))
val_len = int(len(labels['images']) - train_len)
print(train_len, val_len)

train_labels = {
     'images': [],
     'annotations': [],
     'categories': []
}
val_labels = {
     'images': [],
     'annotations': [],
     'categories': []
}
checker_train = set()
checker_val = set()
images = set(os.listdir('images'))
for i, image in enumerate(labels['images']):
        train_labels['images'].append(image)
        checker_train.add(image['id'])
        if i == train_len:
              break


while train_len <= len(labels['images']) - 1:
      val_labels['images'].append(labels['images'][train_len])
      checker_val.add(labels['images'][train_len][('id')])
      train_len += 1

for annotation in labels['annotations']:
     if annotation['image_id'] in checker_train:
          train_labels['annotations'].append(annotation)
     if annotation['image_id'] in checker_val:
           val_labels['annotations'].append(annotation)
           

train_labels['categories'] = labels['categories']
val_labels['categories'] = labels['categories']

with open('train_labels.json', 'w') as f:
     json.dump(train_labels, f)

with open('val_labels.json', 'w') as f:
     json.dump(val_labels, f)