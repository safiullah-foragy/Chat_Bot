"""
Visualization Utilities for Object Detection
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
import torch

from dataset import COCO_CLASSES


def visualize_predictions(image, boxes, labels, scores, threshold=0.5, figsize=(12, 8)):
    """
    Visualize predictions with matplotlib
    
    Args:
        image: PIL Image or tensor
        boxes: Predicted bounding boxes
        labels: Predicted class labels
        scores: Confidence scores
        threshold: Minimum confidence to display
        figsize: Figure size
    """
    # Convert tensor to PIL if needed
    if torch.is_tensor(image):
        image = transforms.ToPILImage()(image)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=figsize)
    ax.imshow(image)
    
    # Generate colors for each class
    colors = plt.cm.hsv(np.linspace(0, 1, len(COCO_CLASSES)))
    
    # Draw boxes
    for box, label, score in zip(boxes, labels, scores):
        if score < threshold:
            continue
        
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        
        if class_name == "N/A":
            continue
        
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        # Create rectangle
        color = colors[label % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax.add_patch(rect)
        
        # Add label
        label_text = f"{class_name}: {score:.2f}"
        ax.text(
            x1, y1 - 5,
            label_text,
            bbox=dict(facecolor=color, alpha=0.7),
            fontsize=10,
            color='white',
            weight='bold'
        )
    
    ax.axis('off')
    plt.tight_layout()
    
    return fig, ax


def visualize_dataset_samples(dataset, num_samples=4, figsize=(15, 10)):
    """
    Visualize samples from dataset with ground truth boxes
    
    Args:
        dataset: Dataset object
        num_samples: Number of samples to show
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    colors = plt.cm.hsv(np.linspace(0, 1, len(COCO_CLASSES)))
    
    for idx in range(min(num_samples, len(dataset))):
        image, target = dataset[idx]
        
        # Convert tensor to numpy
        if torch.is_tensor(image):
            image_np = image.permute(1, 2, 0).numpy()
        else:
            image_np = np.array(image)
        
        ax = axes[idx]
        ax.imshow(image_np)
        
        # Draw ground truth boxes
        boxes = target['boxes'].numpy()
        labels = target['labels'].numpy()
        
        for box, label in zip(boxes, labels):
            class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
            
            if class_name == "N/A":
                continue
            
            x1, y1, x2, y2 = box
            width = x2 - x1
            height = y2 - y1
            
            color = colors[label % len(colors)]
            rect = patches.Rectangle(
                (x1, y1), width, height,
                linewidth=2,
                edgecolor=color,
                facecolor='none'
            )
            ax.add_patch(rect)
            
            ax.text(
                x1, y1 - 5,
                class_name,
                bbox=dict(facecolor=color, alpha=0.7),
                fontsize=8,
                color='white'
            )
        
        ax.set_title(f"Sample {idx + 1} ({len(boxes)} objects)")
        ax.axis('off')
    
    plt.tight_layout()
    return fig


def plot_training_history(log_file):
    """
    Plot training history from tensorboard logs
    
    Args:
        log_file: Path to tensorboard event file
    """
    from tensorboard.backend.event_processing import event_accumulator
    
    ea = event_accumulator.EventAccumulator(log_file)
    ea.Reload()
    
    # Get scalars
    train_loss = ea.Scalars('Loss/train')
    val_loss = ea.Scalars('Loss/val')
    learning_rate = ea.Scalars('LR')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Plot losses
    train_steps = [x.step for x in train_loss]
    train_values = [x.value for x in train_loss]
    val_steps = [x.step for x in val_loss]
    val_values = [x.value for x in val_loss]
    
    ax1.plot(train_steps, train_values, label='Train Loss', marker='o')
    ax1.plot(val_steps, val_values, label='Val Loss', marker='s')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Plot learning rate
    lr_steps = [x.step for x in learning_rate]
    lr_values = [x.value for x in learning_rate]
    
    ax2.plot(lr_steps, lr_values, label='Learning Rate', marker='o', color='orange')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    return fig


def create_class_distribution_plot(dataset):
    """
    Create bar plot showing distribution of object classes in dataset
    
    Args:
        dataset: Dataset object
    """
    from collections import Counter
    
    all_labels = []
    
    print("Analyzing dataset...")
    for i in range(min(1000, len(dataset))):  # Sample first 1000 images
        _, target = dataset[i]
        labels = target['labels'].numpy()
        all_labels.extend(labels)
    
    # Count occurrences
    label_counts = Counter(all_labels)
    
    # Get top 20 classes
    top_classes = label_counts.most_common(20)
    
    class_names = [COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}" 
                   for label, _ in top_classes]
    counts = [count for _, count in top_classes]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(range(len(class_names)), counts, color='steelblue')
    
    ax.set_xlabel('Object Class')
    ax.set_ylabel('Number of Instances')
    ax.set_title('Top 20 Object Classes in Dataset')
    ax.set_xticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha='right')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}',
                ha='center', va='bottom')
    
    plt.tight_layout()
    return fig


def compare_predictions(image, ground_truth, predictions, threshold=0.5):
    """
    Compare ground truth with predictions side by side
    
    Args:
        image: Input image
        ground_truth: Dictionary with 'boxes' and 'labels'
        predictions: Dictionary with 'boxes', 'labels', 'scores'
        threshold: Confidence threshold
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    
    colors = plt.cm.hsv(np.linspace(0, 1, len(COCO_CLASSES)))
    
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        from torchvision import transforms
        image_np = transforms.ToPILImage()(image)
    else:
        image_np = image
    
    # Ground truth
    ax1.imshow(image_np)
    ax1.set_title('Ground Truth', fontsize=16, weight='bold')
    
    gt_boxes = ground_truth['boxes']
    gt_labels = ground_truth['labels']
    
    for box, label in zip(gt_boxes, gt_labels):
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        
        if class_name == "N/A":
            continue
        
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = colors[label % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax1.add_patch(rect)
        ax1.text(x1, y1 - 5, class_name,
                bbox=dict(facecolor=color, alpha=0.7),
                fontsize=10, color='white', weight='bold')
    
    ax1.axis('off')
    
    # Predictions
    ax2.imshow(image_np)
    ax2.set_title('Predictions', fontsize=16, weight='bold')
    
    pred_boxes = predictions['boxes']
    pred_labels = predictions['labels']
    pred_scores = predictions['scores']
    
    for box, label, score in zip(pred_boxes, pred_labels, pred_scores):
        if score < threshold:
            continue
        
        class_name = COCO_CLASSES[label] if label < len(COCO_CLASSES) else f"Class {label}"
        
        if class_name == "N/A":
            continue
        
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        color = colors[label % len(colors)]
        rect = patches.Rectangle(
            (x1, y1), width, height,
            linewidth=2,
            edgecolor=color,
            facecolor='none'
        )
        ax2.add_patch(rect)
        
        label_text = f"{class_name}: {score:.2f}"
        ax2.text(x1, y1 - 5, label_text,
                bbox=dict(facecolor=color, alpha=0.7),
                fontsize=10, color='white', weight='bold')
    
    ax2.axis('off')
    
    plt.tight_layout()
    return fig


if __name__ == '__main__':
    print("Visualization utilities loaded successfully!")
    print("Available functions:")
    print("  - visualize_predictions()")
    print("  - visualize_dataset_samples()")
    print("  - plot_training_history()")
    print("  - create_class_distribution_plot()")
    print("  - compare_predictions()")
