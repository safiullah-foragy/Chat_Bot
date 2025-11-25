"""
REST API Server for Object Detection
Deploy this server and call it from your Flutter app
"""

from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import io
import base64
from typing import Optional

from dataset import COCO_CLASSES

# Initialize FastAPI app
app = FastAPI(
    title="Object Detection API",
    description="API for object detection using Faster R-CNN",
    version="1.0.0"
)

# Enable CORS for Flutter app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins (restrict in production)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables
model = None
device = None


def initialize_model():
    """Initialize the pretrained model"""
    global model, device
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔧 Initializing model on {device}...")
    
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    
    weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
    model = fasterrcnn_resnet50_fpn(weights=weights)
    model.to(device)
    model.eval()
    
    # Optimizations
    model.roi_heads.score_thresh = 0.05
    model.roi_heads.nms_thresh = 0.3
    
    print("✅ Model loaded successfully!")


def predict_image(image_bytes: bytes, confidence_threshold: float = 0.2):
    """
    Predict objects in image
    Returns list of detected objects with bounding boxes
    """
    # Load image
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    
    # Transform
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        predictions = model(image_tensor)[0]
    
    # Filter by confidence
    boxes = predictions['boxes'].cpu().numpy()
    labels = predictions['labels'].cpu().numpy()
    scores = predictions['scores'].cpu().numpy()
    
    # Filter results
    detections = []
    for box, label, score in zip(boxes, labels, scores):
        if score >= confidence_threshold:
            detections.append({
                'class': COCO_CLASSES[label],
                'confidence': float(score),
                'bbox': {
                    'x1': float(box[0]),
                    'y1': float(box[1]),
                    'x2': float(box[2]),
                    'y2': float(box[3])
                }
            })
    
    return detections


@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    initialize_model()


@app.get("/")
async def root():
    """API root endpoint"""
    return {
        "message": "Object Detection API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "detect": "/detect",
            "health": "/health",
            "info": "/info"
        }
    }


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": model is not None,
        "device": str(device)
    }


@app.get("/info")
async def model_info():
    """Get model information"""
    return {
        "model": "Faster R-CNN ResNet50 FPN",
        "training_data": "COCO dataset (330,000+ images)",
        "classes": len(COCO_CLASSES),
        "categories": COCO_CLASSES,
        "device": str(device)
    }


@app.post("/detect")
async def detect_objects(
    file: UploadFile = File(...),
    confidence: Optional[float] = Form(0.2)
):
    """
    Detect objects in uploaded image
    
    Parameters:
    - file: Image file (JPG, PNG, etc.)
    - confidence: Confidence threshold (0.0 to 1.0, default 0.2)
    
    Returns:
    - detections: List of detected objects with bounding boxes
    - count: Number of objects detected
    """
    try:
        # Read image
        image_bytes = await file.read()
        
        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            return JSONResponse(
                status_code=400,
                content={"error": "Confidence must be between 0.0 and 1.0"}
            )
        
        # Predict
        detections = predict_image(image_bytes, confidence)
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "confidence_threshold": confidence
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


@app.post("/detect_base64")
async def detect_objects_base64(
    image_base64: str = Form(...),
    confidence: Optional[float] = Form(0.2)
):
    """
    Detect objects in base64 encoded image
    Useful for Flutter apps that send images as base64
    
    Parameters:
    - image_base64: Base64 encoded image string
    - confidence: Confidence threshold (0.0 to 1.0, default 0.2)
    
    Returns:
    - detections: List of detected objects with bounding boxes
    - count: Number of objects detected
    """
    try:
        # Decode base64
        image_bytes = base64.b64decode(image_base64)
        
        # Validate confidence
        if not 0.0 <= confidence <= 1.0:
            return JSONResponse(
                status_code=400,
                content={"error": "Confidence must be between 0.0 and 1.0"}
            )
        
        # Predict
        detections = predict_image(image_bytes, confidence)
        
        return {
            "success": True,
            "count": len(detections),
            "detections": detections,
            "confidence_threshold": confidence
        }
        
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "error": str(e)
            }
        )


if __name__ == "__main__":
    import uvicorn
    
    print("=" * 70)
    print("OBJECT DETECTION API SERVER")
    print("=" * 70)
    print("🚀 Starting API server...")
    print("📡 Server will be available at: http://0.0.0.0:8000")
    print("📚 API documentation: http://0.0.0.0:8000/docs")
    print("=" * 70)
    
    # Run server
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
