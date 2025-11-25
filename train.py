"""
Training Script for Object Detection Model
Train Faster R-CNN on COCO dataset
"""

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
from tqdm import tqdm
import numpy as np

from model import get_model, print_model_info
from dataset import get_coco_dataloaders


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50):
    """
    Train for one epoch
    """
    model.train()
    
    epoch_loss = 0
    batch_count = 0
    
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for images, targets in progress_bar:
        # Move to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Check for invalid loss
        if not torch.isfinite(losses):
            print(f"Loss is {losses}, stopping training")
            print(f"Loss dict: {loss_dict}")
            continue
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Track loss
        epoch_loss += losses.item()
        batch_count += 1
        
        # Update progress bar
        progress_bar.set_postfix({
            'loss': losses.item(),
            'avg_loss': epoch_loss / batch_count
        })
    
    return epoch_loss / batch_count


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluate model on validation set
    """
    model.eval()
    
    total_loss = 0
    batch_count = 0
    
    progress_bar = tqdm(data_loader, desc='Validation')
    
    for images, targets in progress_bar:
        # Move to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        total_loss += losses.item()
        batch_count += 1
        
        progress_bar.set_postfix({'loss': losses.item()})
    
    return total_loss / batch_count


def train(args):
    """
    Main training function
    """
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get dataloaders
    print("\nLoading dataset...")
    train_loader, val_loader = get_coco_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    if train_loader is None:
        print("Failed to load dataset. Please download COCO first:")
        print("python dataset.py --download")
        return
    
    # Create model
    print("\nCreating model...")
    model = get_model(
        num_classes=91,  # 80 COCO classes + background
        pretrained=args.pretrained,
        trainable_backbone_layers=args.trainable_layers
    )
    model.to(device)
    
    print_model_info(model)
    
    # Setup optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    lr_scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=args.lr_step_size,
        gamma=args.lr_gamma
    )
    
    # Resume from checkpoint if specified
    start_epoch = 0
    if args.resume:
        if os.path.exists(args.resume):
            print(f"\nResuming from checkpoint: {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            print(f"Resuming from epoch {start_epoch}")
        else:
            print(f"Checkpoint not found: {args.resume}")
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=os.path.join(args.output_dir, 'logs'))
    
    # Training loop
    print("\n" + "="*60)
    print("STARTING TRAINING")
    print("="*60)
    
    best_val_loss = float('inf')
    
    for epoch in range(start_epoch, args.epochs):
        print(f"\nEpoch {epoch+1}/{args.epochs}")
        print("-" * 60)
        
        # Train
        train_loss = train_one_epoch(
            model, optimizer, train_loader, device, epoch
        )
        
        # Update learning rate
        lr_scheduler.step()
        
        # Validate
        if val_loader and (epoch + 1) % args.eval_freq == 0:
            val_loss = evaluate(model, val_loader, device)
            print(f"\nValidation Loss: {val_loss:.4f}")
            
            # Log to tensorboard
            writer.add_scalar('Loss/val', val_loss, epoch)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_path = os.path.join(args.output_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': val_loss,
                }, save_path)
                print(f"Best model saved! (val_loss: {val_loss:.4f})")
        
        # Log to tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            checkpoint_path = os.path.join(
                args.output_dir, f'checkpoint_epoch_{epoch+1}.pth'
            )
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
            }, checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_model.pth')
    torch.save({
        'epoch': args.epochs - 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, final_path)
    
    writer.close()
    
    print("\n" + "="*60)
    print("TRAINING COMPLETED!")
    print("="*60)
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved in: {args.output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Train Object Detection Model')
    
    # Data parameters
    parser.add_argument('--data-dir', default='./data', help='Dataset directory')
    parser.add_argument('--output-dir', default='./checkpoints', help='Output directory')
    
    # Model parameters
    parser.add_argument('--pretrained', action='store_true', default=True,
                        help='Use pretrained model')
    parser.add_argument('--trainable-layers', type=int, default=3,
                        help='Number of trainable backbone layers (0-5)')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.005, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
    parser.add_argument('--weight-decay', type=float, default=0.0005, help='Weight decay')
    parser.add_argument('--lr-step-size', type=int, default=3,
                        help='LR scheduler step size')
    parser.add_argument('--lr-gamma', type=float, default=0.1,
                        help='LR scheduler gamma')
    
    # Other parameters
    parser.add_argument('--num-workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--eval-freq', type=int, default=1,
                        help='Evaluation frequency (epochs)')
    parser.add_argument('--save-freq', type=int, default=5,
                        help='Checkpoint save frequency (epochs)')
    parser.add_argument('--resume', default='', help='Resume from checkpoint')
    
    args = parser.parse_args()
    
    train(args)


if __name__ == '__main__':
    main()
