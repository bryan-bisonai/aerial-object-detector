# Aerial Object Detector
An object detection system for aerial data (esp. for DOTA dataset)

## using Tensorflow Object Detection API
- Blog: https://towardsdatascience.com/how-to-train-your-own-object-detector-with-tensorflows-object-detector-api-bec72ecfe1d9
- Explains about how to train using TOD api: https://blog.algorithmia.com/deep-dive-into-object-detection-with-open-images-using-tensorflow/

## Competition
- ODAI: https://captain-whu.github.io/ODAI/index.html

## Applications
- OpenStreetMap: https://github.com/jremillard/images-to-osm

## Dataset
- DOTA: https://captain-whu.github.io/DOTA/dataset.html
    - Paper: https://arxiv.org/pdf/1711.10398.pdf
- COWC: https://gdo152.llnl.gov/cowc/

## Object Detection
- Blogs: 
    - https://medium.com/comet-app/review-of-deep-learning-algorithms-for-object-detection-c1f3d437b852
    - http://cv-tricks.com/object-detection/faster-r-cnn-yolo-ssd/

## Mask-RCNN approach
- reference code:
    - https://github.com/matterport/Mask_RCNN
    - https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN

## Faster-RCNN approach
- reference code: 
    - https://github.com/tensorpack/tensorpack/tree/master/examples/FasterRCNN
    - https://github.com/msracver/Deformable-ConvNets
    - https://github.com/rbgirshick/py-faster-rcnn

## YOLO-based approach 
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

## Train
```
python /data/private/models/research/object_detection/model_main.py --pipeline_config_path='./configs/faster_rcnn_resnet101_dota.config' --train_dir=./checkpoints/faster_rcnn_dota --num_train_steps=1000 --alsologtostderr
```
Download model train weights: 
```
wget https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md
tar -xzvf faster_rcnn_resnet101_coco_2018_01_28.tar.gz
```