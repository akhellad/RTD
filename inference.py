import argparse
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from v3.model import ObjectDetectorFPN
from v3.metrics import DetectionMetricsFPN
from v3.config import ANCHORS

COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
    "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack",
    "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
    "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket",
    "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
    "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
]


def run(image_path, checkpoint_path, output_path, conf_threshold):
    model = ObjectDetectorFPN(80)
    metrics = DetectionMetricsFPN(80, ANCHORS)
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint['ema_state_dict'])
    model.eval()

    img_originale = cv2.imread(image_path)
    img_originale = cv2.cvtColor(img_originale, cv2.COLOR_BGR2RGB)
    img_h, img_w = img_originale.shape[:2]
    scale_h = img_h / 416
    scale_w = img_w / 416

    img = cv2.resize(img_originale, (416, 416))
    img = np.expand_dims(img, axis=0)
    img = torch.from_numpy(np.ascontiguousarray(img)).permute(0, 3, 1, 2).float() / 255

    with torch.no_grad():
        predictions = model(img)
    predictions = metrics.decode_prediction_fpn(predictions)

    for box in predictions[0]:
        if box[4] > conf_threshold:
            start_point = (int(box[0] * scale_w), int(box[1] * scale_h))
            end_point = (int(box[2] * scale_w), int(box[3] * scale_h))
            box_h = end_point[1] - start_point[1]
            font_scale = max(0.3, min(box_h / 300, 1.5))
            cv2.rectangle(img_originale, start_point, end_point, (255, 0, 0))
            cv2.putText(img_originale, f"{COCO_CLASSES[int(box[5])]} {round(float(box[4]), 2)}",
                        (start_point[0], start_point[1] - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1)
    plt.imsave(output_path, img_originale)
    plt.imshow(img_originale)
    plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, default="best_model.pt")
    parser.add_argument("--output", type=str, default="prediction.jpg")
    parser.add_argument("--conf", type=float, default=0.4)
    args = parser.parse_args()
    run(args.image, args.checkpoint, args.output, args.conf)