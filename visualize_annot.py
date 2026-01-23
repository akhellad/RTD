import json
import os
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

with open('labels.json') as f:
    labels_file = json.load(f)

fig, axes = plt.subplots(1, 5, figsize=(25, 6))
indices = np.random.randint(0, len(labels_file['images']), 5)
for i, indice in enumerate(indices):
    image_obj = labels_file['images'][indice]
    image_id = image_obj['id']
    annotations = [annotation for annotation in labels_file['annotations'] if annotation['image_id'] == image_id]
    image = plt.imread(os.path.join('images', image_obj['file_name']))
    axes[i].imshow(image)
    for annotation in annotations:
        x, y, width, height = annotation['bbox'][0], annotation['bbox'][1], annotation['bbox'][2], annotation['bbox'][3]
        rect = patches.Rectangle((x, y), width, height, edgecolor='r', facecolor='none')
        axes[i].add_patch(rect)
        axes[i].axis('off')

plt.show()

