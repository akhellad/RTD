import json 
import os

with open("instances_minitrain2017.json") as f:
    d = json.load(f)

labels = {
     'images': [],
     'annotations': [],
     'categories': []
}
checker = set()
images = set(os.listdir('images'))
for image in d['images']:
     if image['file_name'] in images:
          labels['images'].append(image)
          checker.add(image['id'])

for annotation in d['annotations']:
     if annotation['image_id'] in checker:
          labels['annotations'].append(annotation)

labels['categories'] = d['categories']

with open('labels.json', 'w') as f:
     json.dump(labels, f)