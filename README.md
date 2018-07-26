# Aeerial Object Detector
An object detection system for aerial data (esp. for DOTA dataset)

# Competition
- ODAI: https://captain-whu.github.io/ODAI/index.html

# Dataset
- DOTA: https://captain-whu.github.io/DOTA/dataset.html
    - Paper: https://arxiv.org/pdf/1711.10398.pdf

# Faster-RCNN approach
- reference code: 
    - https://github.com/msracver/Deformable-ConvNets
    - https://github.com/rbgirshick/py-faster-rcnn

# YOLO-based approach 
- ref: https://github.com/ringringyi/DOTA_YOLOv2

## Convert DOTA to YOLO (Darknet) format
In DOTA, the annotation format is:
```
    x1 y1 x2 y2 x3 y3 x4 y4 category difficult
```
In YOLO (Darknet), the below annotation format is required
```
    category-id x y width height
```

To do so, run the script below
```
python convert_to_darknet.py
```

## How to Train
