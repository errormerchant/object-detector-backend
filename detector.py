# import torch
# from torchvision.models.detection import ssdlite320_mobilenet_v3_large, SSDLite320_MobileNet_V3_Large_Weights
from ultralytics import YOLO
from PIL import Image
import io

# Pretrained model
# weights = SSDLite320_MobileNet_V3_Large_Weights.DEFAULT
# transform = weights.transforms()
model = YOLO("yolov8n.pt")
# model.eval()

# COCO class labels
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
    'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
    'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
    'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]

def detect_objects(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    results = model(image, conf=0.5)
    detections = []
    names = results[0].names

    for box in results[0].boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        label = names[int(box.cls)]
        score = float(box.conf)
        detections.append({
            "label": label,
            "score": round(score, 2),
            "bbox": [x1, y1, x2, y2]
        })
    
    if not detections:
        detections.append({
            "label": 'Hold on, let me put on my glasses… nope, still nothing.',
            "score": 0,
            "bbox": [0, 0, 0, 0]
        })

    return detections