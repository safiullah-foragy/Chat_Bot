"""
Quick Start Demo
Demonstrates object detection on a sample image using pretrained model
"""

import torch
from model import get_model
from dataset import COCO_CLASSES
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import requests
from io import BytesIO
import numpy as np


def download_sample_image():
    """Download a sample image for demo"""
    url = "https://images.pexels.com/photos/1108099/pexels-photo-1108099.jpeg?auto=compress&cs=tinysrgb&w=1260"
    
    print("Downloading sample image...")
    response = requests.get(url)
    image = Image.open(BytesIO(response.content)).convert('RGB')
    
    # Resize if too large
    max_size = 800
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.LANCZOS)
    
    return image


def run_demo():
    """Run a quick demo of object detection"""
    
    print("="*60)
    print("OBJECT DETECTION DEMO")
    print("="*60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    # Load pretrained model
    print("\nLoading pretrained model...")
    model = get_model(num_classes=91, pretrained=True)
    model.to(device)
    model.eval()
    print("Model loaded!")
    
    # Download sample image
    try:
        image = download_sample_image()
        print("Sample image loaded!")
    except:
        print("\nCouldn't download sample image.")
        print("Please run: python predict.py --image your_image.jpg")
        return
    
    # Preprocess
    print("\nRunning inference...")
    transform = transforms.ToTensor()
    image_tensor = transform(image).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model([image_tensor])
    
    # Extract results
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filter by threshold
    threshold = 0.5
    mask = scores >= threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    print(f"\nFound {len(boxes)} objects with confidence > {threshold}:")
    print("-" * 60)
    
    # Display results
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        
        if class_name != "N/A":
            x1, y1, x2, y2 = box
            print(f"{i+1}. {class_name}: {score:.2%} confidence")
            print(f"   Location: [{int(x1)}, {int(y1)}, {int(x2)}, {int(y2)}]")
    
    # Draw boxes on image
    print("\nDrawing bounding boxes...")
    draw = ImageDraw.Draw(image)
    
    # Generate colors
    np.random.seed(42)
    colors = [(np.random.randint(0, 255), np.random.randint(0, 255), 
               np.random.randint(0, 255)) for _ in range(len(COCO_CLASSES))]
    
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 20)
    except:
        font = ImageFont.load_default()
    
    for box, label, score in zip(boxes, labels, scores):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        
        if class_name == "N/A":
            continue
        
        color = colors[label % len(colors)]
        x1, y1, x2, y2 = box
        
        # Draw box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
        
        # Draw label
        label_text = f"{class_name}: {score:.2f}"
        bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        draw.rectangle([x1, y1 - text_height - 8, x1 + text_width + 8, y1], 
                      fill=color)
        draw.text((x1 + 4, y1 - text_height - 4), label_text, 
                 fill='white', font=font)
    
    # Save result
    output_path = 'demo_result.jpg'
    image.save(output_path)
    
    print(f"\n✓ Result saved to: {output_path}")
    print("\n" + "="*60)
    print("DEMO COMPLETE!")
    print("="*60)
    print("\nTo detect objects in your own images:")
    print("  python predict.py --image path/to/your/image.jpg")
    print("\nTo train on COCO dataset:")
    print("  1. Download dataset: python dataset.py --download")
    print("  2. Train model: python train.py --epochs 10")
    print("\nTo use webcam:")
    print("  python predict.py --webcam")


if __name__ == '__main__':
    run_demo()
