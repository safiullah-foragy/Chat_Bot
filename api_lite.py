"""
Lightweight REST API Server for Object Detection
Can be accessed from any app via HTTP requests
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
import json

from model import get_model
from dataset import COCO_CLASSES

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global variables
model = None
device = None
COLORS = None


def initialize_model():
    """Initialize the pretrained model"""
    global model, device, COLORS
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    
    print("Loading Faster R-CNN ResNet50 FPN...")
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    
    # Optimization settings
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.3
    
    # Generate colors for visualization
    np.random.seed(42)
    COLORS = np.random.randint(0, 255, size=(len(COCO_CLASSES), 3), dtype=np.uint8)
    
    print("✅ Model loaded successfully!")


def predict_objects_api(image, confidence_threshold=0.5):
    """
    Predict objects in an image
    
    Args:
        image: PIL Image
        confidence_threshold: Minimum confidence for detection
    
    Returns:
        dict: Detection results
    """
    # Preprocess image
    transform = transforms.ToTensor()
    image_tensor = transform(image).to(device)
    
    # Make prediction
    with torch.no_grad():
        predictions = model([image_tensor])
    
    # Extract results
    pred = predictions[0]
    boxes = pred['boxes'].cpu().numpy()
    labels = pred['labels'].cpu().numpy()
    scores = pred['scores'].cpu().numpy()
    
    # Filter by confidence threshold
    mask = scores >= confidence_threshold
    boxes = boxes[mask]
    labels = labels[mask]
    scores = scores[mask]
    
    # Prepare results
    detections = []
    
    for box, label, score in zip(boxes, labels, scores):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        
        # Skip N/A classes
        if class_name == "N/A":
            continue
        
        x1, y1, x2, y2 = box
        
        detection_info = {
            "object": class_name,
            "confidence": float(score),
            "bbox": {
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2)
            },
            "center": {
                "x": int((x1 + x2) / 2),
                "y": int((y1 + y2) / 2)
            },
            "size": {
                "width": int(x2 - x1),
                "height": int(y2 - y1)
            }
        }
        detections.append(detection_info)
    
    return {
        "success": True,
        "total_detections": len(detections),
        "confidence_threshold": float(confidence_threshold),
        "image_size": {
            "width": int(image.width),
            "height": int(image.height)
        },
        "detections": detections
    }


@app.route('/', methods=['GET'])
def home():
    """API information endpoint"""
    return jsonify({
        "name": "Object Detection API",
        "version": "1.0",
        "status": "running",
        "endpoints": {
            "/": "API information",
            "/health": "Health check",
            "/detect": "Object detection (POST with image)",
            "/detect/url": "Object detection from URL (POST with image_url)",
            "/detect/base64": "Object detection from base64 (POST with image_base64)",
            "/classes": "List of detectable classes"
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    })


@app.route('/classes', methods=['GET'])
def get_classes():
    """Get list of detectable classes"""
    return jsonify({
        "total_classes": len([c for c in COCO_CLASSES if c != "N/A"]),
        "classes": [c for c in COCO_CLASSES if c != "N/A"]
    })


@app.route('/detect', methods=['POST'])
def detect():
    """
    Detect objects in uploaded image
    
    Request:
        - file: image file (multipart/form-data)
        - confidence: optional confidence threshold (default: 0.5)
    
    Response:
        JSON with detection results
    """
    try:
        # Check if image file is present
        if 'file' not in request.files:
            return jsonify({
                "success": False,
                "error": "No image file provided. Use 'file' field."
            }), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({
                "success": False,
                "error": "Empty filename"
            }), 400
        
        # Get confidence threshold
        confidence = float(request.form.get('confidence', 0.5))
        
        # Read and process image
        image_bytes = file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Perform detection
        results = predict_objects_api(image, confidence)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/detect/url', methods=['POST'])
def detect_url():
    """
    Detect objects from image URL
    
    Request JSON:
        - image_url: URL of the image
        - confidence: optional confidence threshold (default: 0.5)
    
    Response:
        JSON with detection results
    """
    try:
        data = request.get_json()
        
        if not data or 'image_url' not in data:
            return jsonify({
                "success": False,
                "error": "No image_url provided in JSON body"
            }), 400
        
        image_url = data['image_url']
        confidence = float(data.get('confidence', 0.5))
        
        # Download and process image
        import requests
        response = requests.get(image_url, timeout=10)
        image = Image.open(io.BytesIO(response.content)).convert('RGB')
        
        # Perform detection
        results = predict_objects_api(image, confidence)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/detect/base64', methods=['POST'])
def detect_base64():
    """
    Detect objects from base64 encoded image
    
    Request JSON:
        - image_base64: base64 encoded image string
        - confidence: optional confidence threshold (default: 0.5)
    
    Response:
        JSON with detection results
    """
    try:
        data = request.get_json()
        
        if not data or 'image_base64' not in data:
            return jsonify({
                "success": False,
                "error": "No image_base64 provided in JSON body"
            }), 400
        
        image_base64 = data['image_base64']
        confidence = float(data.get('confidence', 0.5))
        
        # Remove data URL prefix if present
        if ',' in image_base64:
            image_base64 = image_base64.split(',')[1]
        
        # Decode base64 and process image
        image_bytes = base64.b64decode(image_base64)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        
        # Perform detection
        results = predict_objects_api(image, confidence)
        
        return jsonify(results)
    
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


if __name__ == "__main__":
    print("=" * 70)
    print("OBJECT DETECTION LITE API SERVER")
    print("=" * 70)
    
    # Initialize model
    print("\n🔧 Initializing model...")
    initialize_model()
    
    # Start server
    print("\n🚀 Starting API server...")
    print("📡 API will be available at: http://0.0.0.0:5000")
    print("\n📝 Endpoints:")
    print("  GET  /           - API information")
    print("  GET  /health     - Health check")
    print("  GET  /classes    - List detectable classes")
    print("  POST /detect     - Upload image file")
    print("  POST /detect/url - Detect from image URL")
    print("  POST /detect/base64 - Detect from base64")
    print("\n" + "=" * 70)
    
    app.run(
        host='0.0.0.0',
        port=5000,
        debug=False,
        threaded=True
    )
