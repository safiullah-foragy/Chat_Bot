"""
Inference Script for Object Detection
Predict objects in images using trained model
"""

import torch
import torchvision.transforms as transforms
from PIL import Image, ImageDraw, ImageFont
import argparse
import os
import numpy as np
import cv2

from model import get_model
from dataset import COCO_CLASSES


# Color palette for different classes
COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)


def load_model(checkpoint_path=None, device='cpu'):
    """
    Load trained model or use pretrained model
    """
    model = get_model(num_classes=91, pretrained=True)
    
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading model from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("Using pretrained COCO model")
    
    model.to(device)
    model.eval()
    
    return model


def preprocess_image(image_path):
    """
    Preprocess image for inference
    """
    image = Image.open(image_path).convert('RGB')
    transform = transforms.ToTensor()
    image_tensor = transform(image)
    
    return image, image_tensor


def predict(model, image_tensor, device, threshold=0.5):
    """
    Make predictions on an image
    
    Args:
        model: Detection model
        image_tensor: Preprocessed image tensor
        device: Device to run inference on
        threshold: Confidence threshold for detections
    
    Returns:
        boxes, labels, scores
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(device)
        predictions = model([image_tensor])
    
    # Extract predictions
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filter by threshold
    mask = scores >= threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    return boxes, labels, scores


def draw_boxes(image, boxes, labels, scores):
    """
    Draw bounding boxes on image
    
    Args:
        image: PIL Image
        boxes: Bounding boxes
        labels: Class labels
        scores: Confidence scores
    
    Returns:
        Image with drawn boxes
    """
    draw = ImageDraw.Draw(image)
    
    # Try to use a nice font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
    except:
        font = ImageFont.load_default()
    
    for box, label, score in zip(boxes, labels, scores):
        # Get class name
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        
        # Skip N/A classes
        if class_name == "N/A":
            continue
        
        # Get color for this class
        color = tuple(int(c) for c in COLORS[label % len(COLORS)])
        
        # Draw box
        x1, y1, x2, y2 = box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw label background
        label_text = f"{class_name}: {score:.2f}"
        
        # Get text size
        bbox = draw.textbbox((x1, y1), label_text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        
        # Draw background rectangle for text
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill=color
        )
        
        # Draw text
        draw.text((x1 + 2, y1 - text_height - 2), label_text, fill='white', font=font)
    
    return image


def predict_image(model, image_path, device, threshold=0.5, output_path=None):
    """
    Predict objects in a single image
    """
    # Load and preprocess image
    image, image_tensor = preprocess_image(image_path)
    
    # Make predictions
    boxes, labels, scores = predict(model, image_tensor, device, threshold)
    
    # Print detections
    print(f"\nDetections in {os.path.basename(image_path)}:")
    print(f"Found {len(boxes)} objects:")
    
    for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        if class_name != "N/A":
            print(f"  {i+1}. {class_name}: {score:.2f} at {box}")
    
    # Draw boxes
    result_image = draw_boxes(image.copy(), boxes, labels, scores)
    
    # Save or display
    if output_path:
        result_image.save(output_path)
        print(f"\nSaved result to: {output_path}")
    
    return result_image, boxes, labels, scores


def predict_folder(model, folder_path, device, threshold=0.5, output_dir=None):
    """
    Predict objects in all images in a folder
    """
    # Create output directory
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_files = []
    
    for file in os.listdir(folder_path):
        if any(file.lower().endswith(ext) for ext in image_extensions):
            image_files.append(file)
    
    print(f"Found {len(image_files)} images in {folder_path}")
    
    # Process each image
    all_detections = []
    
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        
        # Set output path
        if output_dir:
            output_path = os.path.join(output_dir, f"detected_{image_file}")
        else:
            output_path = None
        
        # Predict
        try:
            result_image, boxes, labels, scores = predict_image(
                model, image_path, device, threshold, output_path
            )
            
            all_detections.append({
                'filename': image_file,
                'boxes': boxes,
                'labels': labels,
                'scores': scores
            })
        except Exception as e:
            print(f"Error processing {image_file}: {e}")
    
    return all_detections


def predict_webcam(model, device, threshold=0.5):
    """
    Real-time object detection from webcam
    """
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return
    
    print("Starting webcam... Press 'q' to quit")
    
    transform = transforms.ToTensor()
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            break
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(frame_rgb)
        
        # Preprocess
        image_tensor = transform(pil_image)
        
        # Predict
        boxes, labels, scores = predict(model, image_tensor, device, threshold)
        
        # Draw on frame
        for box, label, score in zip(boxes, labels, scores):
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
            
            if class_name == "N/A":
                continue
            
            x1, y1, x2, y2 = map(int, box)
            color = tuple(int(c) for c in COLORS[label % len(COLORS)])
            
            # Draw rectangle
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label_text = f"{class_name}: {score:.2f}"
            cv2.putText(frame, label_text, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Display
        cv2.imshow('Object Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description='Object Detection Inference')
    
    # Input options
    parser.add_argument('--image', type=str, help='Path to input image')
    parser.add_argument('--folder', type=str, help='Path to folder with images')
    parser.add_argument('--webcam', action='store_true', help='Use webcam')
    
    # Model options
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Detection confidence threshold')
    
    # Output options
    parser.add_argument('--output', type=str, help='Output path for single image')
    parser.add_argument('--output-dir', type=str, default='./predictions',
                        help='Output directory for batch processing')
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    print("Loading model...")
    model = load_model(args.checkpoint, device)
    print("Model loaded successfully!")
    
    # Run inference
    if args.webcam:
        predict_webcam(model, device, args.threshold)
    elif args.image:
        predict_image(model, args.image, device, args.threshold, args.output)
    elif args.folder:
        predict_folder(model, args.folder, device, args.threshold, args.output_dir)
    else:
        print("Please specify --image, --folder, or --webcam")


if __name__ == '__main__':
    main()
