import numpy as np
import json

def load_annots(json_filepath):
    with open(json_filepath)as f:
        annot_file = json.load(f)
    sizes = {image['id'] : (image['width'], image['height']) for image in annot_file['images']}
    boxes = []
    for annotation in annot_file['annotations']:
        if annotation['bbox'][2] > 0 and annotation['bbox'][3] > 0:
            boxes.append((annotation['bbox'][2] / sizes[annotation['image_id']][0], annotation['bbox'][3] / sizes[annotation['image_id']][1]))
    boxes = np.array(boxes)
    return boxes

def kmean(boxes):
    assignements = np.array([-1] * len(boxes))
    indices = np.random.choice(len(boxes), 9, replace=False)
    centroids = boxes[indices]
    w_boxes = boxes[:, 0]
    w_boxes = w_boxes[:, np.newaxis]
    h_boxes = boxes[:, 1]
    h_boxes = h_boxes[:, np.newaxis]
    while True:
        w_centroids = centroids[:, 0]
        w_centroids = w_centroids[np.newaxis, :]
        h_centroids = centroids[:, 1]
        h_centroids = h_centroids[np.newaxis, :]
        inter_w = np.minimum(w_boxes, w_centroids)
        inter_h = np.minimum(h_boxes, h_centroids)
        intersections = inter_h * inter_w
        boxes_area = boxes[:, 0] * boxes[:, 1]
        boxes_area = boxes_area[:, np.newaxis]
        centroids_area = centroids[:, 0] * centroids[:, 1]
        centroids_area = centroids_area[np.newaxis, :]
        union = boxes_area + centroids_area - intersections
        iou = intersections / union
        new_assignements = np.argmax(iou, 1)
        if np.array_equal(assignements, new_assignements):    
            break
        assignements = new_assignements
        for k in range(9):
            centroids[k] = np.median(boxes[assignements == k], axis=0)        

    centroids_pixel = centroids * 416
    centroids_pixel_area = centroids_pixel[:, 0] * centroids_pixel[:, 1]
    centroids_pixel = centroids_pixel[np.argsort(centroids_pixel_area)]
    return [centroids_pixel[:3], centroids_pixel[3:6], centroids_pixel[6:]]

boxes = load_annots('instances_train2017.json')
anchors = kmean(boxes)
print(anchors)
