"""
Object Detection Model Architecture
Uses Faster R-CNN with ResNet-50 backbone pre-trained on COCO
"""

import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torch.nn as nn


def get_model(num_classes=91, pretrained=True, trainable_backbone_layers=3):
    """
    Get Faster R-CNN model with ResNet-50 backbone
    
    Args:
        num_classes: Number of classes (80 COCO classes + background)
        pretrained: Use pretrained weights on COCO
        trainable_backbone_layers: Number of trainable backbone layers (0-5)
    
    Returns:
        model: Faster R-CNN model
    """
    
    # Load pre-trained Faster R-CNN model
    model = fasterrcnn_resnet50_fpn(
        pretrained=pretrained,
        trainable_backbone_layers=trainable_backbone_layers
    )
    
    # Get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    
    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    
    return model


def get_pretrained_model():
    """
    Get fully pre-trained model on COCO dataset
    Ready for inference without additional training
    """
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    model.eval()
    return model


class ObjectDetectionModel(nn.Module):
    """
    Wrapper class for object detection model with additional utilities
    """
    
    def __init__(self, num_classes=91, pretrained=True):
        super().__init__()
        self.model = get_model(num_classes, pretrained)
        self.num_classes = num_classes
        
    def forward(self, images, targets=None):
        """
        Forward pass
        
        Args:
            images: List of images (each a Tensor)
            targets: List of target dicts (for training)
        
        Returns:
            losses (training mode) or detections (eval mode)
        """
        if self.training and targets is not None:
            return self.model(images, targets)
        else:
            return self.model(images)
    
    def predict(self, images):
        """
        Make predictions on images
        
        Args:
            images: List of image tensors
        
        Returns:
            List of predictions (boxes, labels, scores)
        """
        self.eval()
        with torch.no_grad():
            predictions = self.model(images)
        return predictions
    
    def save(self, path):
        """Save model checkpoint"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes
        }, path)
        print(f"Model saved to {path}")
    
    def load(self, path, device='cpu'):
        """Load model checkpoint"""
        checkpoint = torch.load(path, map_location=device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Model loaded from {path}")


def print_model_info(model):
    """Print model architecture information"""
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print("\n" + "="*60)
    print("MODEL INFORMATION")
    print("="*60)
    print(f"Architecture: Faster R-CNN with ResNet-50 + FPN backbone")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")
    print("="*60 + "\n")


if __name__ == '__main__':
    # Test model creation
    print("Creating model...")
    model = get_model(num_classes=91, pretrained=True)
    print_model_info(model)
    
    # Test forward pass
    print("Testing forward pass...")
    model.eval()
    
    # Create dummy input
    dummy_images = [torch.rand(3, 600, 800) for _ in range(2)]
    
    with torch.no_grad():
        outputs = model(dummy_images)
    
    print(f"Output type: {type(outputs)}")
    print(f"Number of predictions: {len(outputs)}")
    print(f"Keys in prediction: {outputs[0].keys()}")
    print(f"Number of detected boxes: {len(outputs[0]['boxes'])}")
    
    print("\nModel test successful!")
