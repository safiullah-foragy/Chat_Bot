"""
Test script to verify model predictions
"""
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
from model import get_model

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Use pretrained model directly
from torchvision.models.detection import fasterrcnn_resnet50_fpn
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.to(device)
model.eval()

print("Model loaded successfully!")

# Load a real test image
test_image = Image.open('test_dog.jpg').convert('RGB')
print(f"\nTest image size: {test_image.size}")

# Preprocess
transform = transforms.ToTensor()
image_tensor = transform(test_image).to(device)
print(f"Tensor shape: {image_tensor.shape}")

# Make prediction
print("\nMaking prediction...")
with torch.no_grad():
    predictions = model([image_tensor])

# Extract results
pred = predictions[0]
boxes = pred['boxes'].cpu().numpy()
labels = pred['labels'].cpu().numpy()
scores = pred['scores'].cpu().numpy()

print(f"\nRaw predictions:")
print(f"Number of boxes: {len(boxes)}")
print(f"Number of labels: {len(labels)}")
print(f"Number of scores: {len(scores)}")

if len(scores) > 0:
    print(f"\nScore range: {scores.min():.4f} to {scores.max():.4f}")
    print(f"\nTop 10 predictions:")
    for i in range(min(10, len(scores))):
        print(f"  {i+1}. Label {labels[i]}, Score: {scores[i]:.4f}, Box: {boxes[i]}")
    
    # Test different thresholds
    for threshold in [0.1, 0.3, 0.5, 0.7, 0.9]:
        mask = scores >= threshold
        count = mask.sum()
        print(f"\nDetections with threshold {threshold}: {count}")
else:
    print("\n⚠️ WARNING: No predictions returned by model!")
