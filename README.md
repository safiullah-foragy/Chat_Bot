# Object Detection Model

A comprehensive object detection system trained on the COCO dataset to detect 80+ object categories.

## Features
- Pre-trained Faster R-CNN with ResNet-50 backbone
- Trained on COCO dataset (330K+ images, 1.5M+ object instances)
- Detects 80 different object categories
- High accuracy and real-time inference capability

## Installation

```bash
pip install -r requirements.txt
```

## Dataset
The model uses the COCO (Common Objects in Context) dataset:
- **Training images**: ~118K images
- **Validation images**: ~5K images
- **Object categories**: 80 classes
- **Total instances**: 1.5M+ labeled objects

## Usage

### Training
```bash
python train.py --epochs 10 --batch-size 4 --lr 0.001
```

### Inference
```bash
python predict.py --image path/to/image.jpg --threshold 0.5
```

### Batch Prediction
```bash
python predict.py --folder path/to/images/ --threshold 0.5
```

## Object Categories
The model can detect 80 categories including:
- People, vehicles, animals
- Household items, furniture
- Food items, sports equipment
- And many more!

## Model Architecture
- **Base**: Faster R-CNN
- **Backbone**: ResNet-50 with FPN
- **Pre-trained**: ImageNet + COCO weights
- **Output**: Bounding boxes, class labels, confidence scores
