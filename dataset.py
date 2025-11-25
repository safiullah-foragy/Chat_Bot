"""
COCO Dataset Loader and Preprocessing
Handles downloading, loading, and preprocessing of COCO dataset
"""

import torch
from torch.utils.data import DataLoader
import torchvision
from torchvision.datasets import CocoDetection
from torchvision.transforms import functional as F
import numpy as np
import os
from PIL import Image
try:
    from pycocotools.coco import COCO
except ImportError:
    COCO = None


class COCODataset(torch.utils.data.Dataset):
    """Custom COCO dataset with augmentations"""
    
    def __init__(self, root, annotation_file, transforms=None):
        self.root = root
        self.coco = COCO(annotation_file)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.transforms = transforms
        
    def __getitem__(self, index):
        # Get image ID
        img_id = self.ids[index]
        
        # Load image
        path = self.coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        
        # Get annotations
        ann_ids = self.coco.getAnnIds(imgIds=img_id)
        anns = self.coco.loadAnns(ann_ids)
        
        # Filter out invalid annotations
        anns = [ann for ann in anns if ann.get('iscrowd', 0) == 0 and ann['area'] > 0]
        
        if len(anns) == 0:
            # Return empty tensors if no valid annotations
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
            area = torch.zeros((0,), dtype=torch.float32)
            iscrowd = torch.zeros((0,), dtype=torch.int64)
        else:
            # Extract boxes and labels
            boxes = []
            labels = []
            areas = []
            iscrowds = []
            
            for ann in anns:
                xmin, ymin, width, height = ann['bbox']
                xmax = xmin + width
                ymax = ymin + height
                
                # Ensure valid boxes
                if width > 0 and height > 0:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(ann['category_id'])
                    areas.append(ann['area'])
                    iscrowds.append(ann.get('iscrowd', 0))
            
            if len(boxes) == 0:
                boxes = torch.zeros((0, 4), dtype=torch.float32)
                labels = torch.zeros((0,), dtype=torch.int64)
                area = torch.zeros((0,), dtype=torch.float32)
                iscrowd = torch.zeros((0,), dtype=torch.int64)
            else:
                boxes = torch.as_tensor(boxes, dtype=torch.float32)
                labels = torch.as_tensor(labels, dtype=torch.int64)
                area = torch.as_tensor(areas, dtype=torch.float32)
                iscrowd = torch.as_tensor(iscrowds, dtype=torch.int64)
        
        # Create target dictionary
        target = {}
        target['boxes'] = boxes
        target['labels'] = labels
        target['image_id'] = torch.tensor([img_id])
        target['area'] = area
        target['iscrowd'] = iscrowd
        
        # Apply transforms
        if self.transforms is not None:
            img = self.transforms(img)
        
        return img, target
    
    def __len__(self):
        return len(self.ids)


def get_transform(train=True):
    """Get image transformations"""
    transforms = []
    transforms.append(torchvision.transforms.ToTensor())
    
    if train:
        transforms.append(torchvision.transforms.RandomHorizontalFlip(0.5))
    
    return torchvision.transforms.Compose(transforms)


def collate_fn(batch):
    """Custom collate function for dataloader"""
    return tuple(zip(*batch))


def download_coco_dataset(data_dir='./data'):
    """
    Download COCO dataset
    Note: This will download ~20GB of data
    """
    import urllib.request
    import zipfile
    
    os.makedirs(data_dir, exist_ok=True)
    
    # COCO 2017 URLs
    urls = {
        'train_images': 'http://images.cocodataset.org/zips/train2017.zip',
        'val_images': 'http://images.cocodataset.org/zips/val2017.zip',
        'annotations': 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'
    }
    
    for name, url in urls.items():
        zip_path = os.path.join(data_dir, f'{name}.zip')
        
        if not os.path.exists(zip_path):
            print(f'Downloading {name}...')
            urllib.request.urlretrieve(url, zip_path)
            
            print(f'Extracting {name}...')
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir)
            
            print(f'{name} downloaded and extracted!')
        else:
            print(f'{name} already exists.')


def get_coco_dataloaders(data_dir='./data', batch_size=4, num_workers=4):
    """
    Get COCO train and validation dataloaders
    """
    train_dir = os.path.join(data_dir, 'train2017')
    val_dir = os.path.join(data_dir, 'val2017')
    train_ann = os.path.join(data_dir, 'annotations', 'instances_train2017.json')
    val_ann = os.path.join(data_dir, 'annotations', 'instances_val2017.json')
    
    # Check if dataset exists
    if not os.path.exists(train_ann):
        print("COCO dataset not found. Please download it first.")
        print("Run: python dataset.py --download")
        return None, None
    
    # Create datasets
    train_dataset = COCODataset(
        root=train_dir,
        annotation_file=train_ann,
        transforms=get_transform(train=True)
    )
    
    val_dataset = COCODataset(
        root=val_dir,
        annotation_file=val_ann,
        transforms=get_transform(train=False)
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    
    return train_loader, val_loader


# COCO class names (80 categories)
COCO_CLASSES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='COCO Dataset Manager')
    parser.add_argument('--download', action='store_true', help='Download COCO dataset')
    parser.add_argument('--data-dir', default='./data', help='Data directory')
    
    args = parser.parse_args()
    
    if args.download:
        download_coco_dataset(args.data_dir)
        print("Dataset download complete!")
    else:
        # Test dataloader
        train_loader, val_loader = get_coco_dataloaders(args.data_dir, batch_size=2)
        
        if train_loader:
            print("\nTesting dataloader...")
            images, targets = next(iter(train_loader))
            print(f"Batch size: {len(images)}")
            print(f"Image shape: {images[0].shape}")
            print(f"Number of objects in first image: {len(targets[0]['boxes'])}")
