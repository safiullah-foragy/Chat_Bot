# Quick Start Guide

## Installation

1. **Run the setup script:**
```bash
chmod +x setup.sh
./setup.sh
```

2. **Activate the environment:**
```bash
source venv/bin/activate
```

## Quick Demo (No Training Needed!)

Run the demo with a pretrained model:
```bash
python demo.py
```

This will download a sample image and detect objects using the pretrained COCO model.

## Using the Pretrained Model

Detect objects in your images immediately:

```bash
# Single image
python predict.py --image path/to/your/image.jpg --output result.jpg

# Folder of images
python predict.py --folder path/to/images/ --output-dir predictions/

# Webcam (real-time detection)
python predict.py --webcam
```

The model is already trained on COCO dataset (330K+ images) and can detect **80 different object categories**!

## Training on Full COCO Dataset

If you want to fine-tune or train from scratch:

### Step 1: Download COCO Dataset (~20GB)
```bash
python dataset.py --download
```

This downloads:
- 118K training images
- 5K validation images
- 1.5M+ labeled object instances
- 80 object categories

### Step 2: Train the Model
```bash
# Quick training (10 epochs)
python train.py --epochs 10 --batch-size 4

# Full training (50 epochs)
python train.py --epochs 50 --batch-size 8 --lr 0.005

# Resume from checkpoint
python train.py --resume checkpoints/checkpoint_epoch_10.pth
```

### Training Options:
- `--epochs`: Number of training epochs
- `--batch-size`: Batch size (reduce if out of memory)
- `--lr`: Learning rate
- `--pretrained`: Use pretrained weights (recommended)
- `--trainable-layers`: Number of backbone layers to train (0-5)

### Step 3: Use Your Trained Model
```bash
python predict.py --image test.jpg --checkpoint checkpoints/best_model.pth
```

## Object Categories

The model can detect these 80 categories:

**People & Animals:**
- person, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe

**Vehicles:**
- bicycle, car, motorcycle, airplane, bus, train, truck, boat

**Outdoor Objects:**
- traffic light, fire hydrant, stop sign, parking meter, bench

**Sports:**
- frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket

**Kitchen:**
- bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake

**Furniture:**
- chair, couch, potted plant, bed, dining table, toilet

**Electronics:**
- tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator

**Accessories:**
- backpack, umbrella, handbag, tie, suitcase

**Other:**
- book, clock, vase, scissors, teddy bear, hair drier, toothbrush

## Model Performance

- **Architecture:** Faster R-CNN with ResNet-50 + FPN
- **Parameters:** ~42 million
- **Pretrained on:** ImageNet (backbone) + COCO dataset
- **Dataset size:** 330K+ images with 1.5M+ instances
- **Classes:** 80 object categories
- **Inference speed:** ~50ms per image (GPU), ~500ms (CPU)

## Advanced Usage

### Custom Confidence Threshold
```bash
python predict.py --image test.jpg --threshold 0.7
```

### GPU Training
```bash
# The script automatically uses GPU if available
python train.py --batch-size 8  # Increase batch size for GPU
```

### Monitor Training
```bash
# View training progress in tensorboard
tensorboard --logdir checkpoints/logs
```

## Tips for Best Results

1. **Start with pretrained model** - It's already trained on 330K images!
2. **Adjust threshold** - Lower for more detections, higher for more precise
3. **Good lighting** - Model works best with clear, well-lit images
4. **Scale matters** - Very small or very large objects may be harder to detect
5. **GPU recommended** - Training is much faster with a GPU

## Memory Requirements

- **Inference:** 2-4 GB RAM, 2 GB GPU (optional)
- **Training:** 8+ GB RAM, 6+ GB GPU (recommended)
- **Dataset:** 20 GB disk space

## Troubleshooting

**Out of memory during training?**
- Reduce batch size: `--batch-size 2`
- Reduce trainable layers: `--trainable-layers 1`

**Slow inference?**
- Use GPU if available
- Reduce image size before prediction

**Low detection accuracy?**
- Lower confidence threshold: `--threshold 0.3`
- Use better quality images
- Fine-tune the model on your specific dataset

## Example Commands

```bash
# Quick test with demo
python demo.py

# Predict on single image
python predict.py --image cat.jpg --threshold 0.6 --output detected_cat.jpg

# Batch process folder
python predict.py --folder vacation_photos/ --output-dir results/ --threshold 0.5

# Real-time webcam detection
python predict.py --webcam --threshold 0.7

# Train with custom settings
python train.py --epochs 20 --batch-size 4 --lr 0.001 --output-dir my_model/

# Use your trained model
python predict.py --image test.jpg --checkpoint my_model/best_model.pth
```

## Next Steps

1. Try the demo: `python demo.py`
2. Test on your images: `python predict.py --image your_image.jpg`
3. If you need custom training, download COCO and train
4. Share your results!

For more information, see the main README.md
