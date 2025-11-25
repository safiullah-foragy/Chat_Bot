"""
Dataset Downloader for Expanding Training Data
Downloads large-scale object detection datasets to reach 1M+ images
"""

import os
import requests
import zipfile
import tarfile
from tqdm import tqdm
import json
import argparse


class DatasetDownloader:
    """Download and prepare large-scale datasets for training"""
    
    def __init__(self, base_dir="datasets"):
        self.base_dir = base_dir
        os.makedirs(base_dir, exist_ok=True)
        
    def download_file(self, url, dest_path):
        """Download file with progress bar"""
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        
        with open(dest_path, 'wb') as f, tqdm(
            desc=dest_path,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                pbar.update(size)
    
    def download_objects365(self, subset_size=600000):
        """
        Download Objects365 dataset
        - 2 Million images total (365 categories, 30M+ boxes)
        - Can download subset for faster setup
        """
        print("=" * 70)
        print("DOWNLOADING OBJECTS365 DATASET")
        print("=" * 70)
        print(f"📦 Total available: 2,000,000 images")
        print(f"📥 Downloading subset: {subset_size:,} images")
        print(f"💾 Storage needed: ~{subset_size * 0.3 / 1000:.1f} GB")
        print()
        
        objects365_dir = os.path.join(self.base_dir, "objects365")
        os.makedirs(objects365_dir, exist_ok=True)
        
        # Objects365 download URLs (these are example URLs - update with actual ones)
        print("⚠️  Objects365 requires registration at: https://www.objects365.org/")
        print("After registration, download links will be provided.")
        print(f"Download to: {objects365_dir}")
        print()
        
        return objects365_dir
    
    def download_open_images(self, num_images=100000, categories=None):
        """
        Download Open Images V7 dataset
        - 9 Million images total (600 categories)
        - Can filter by specific categories
        - Free and open source
        """
        print("=" * 70)
        print("DOWNLOADING OPEN IMAGES V7 DATASET")
        print("=" * 70)
        print(f"📦 Total available: 9,000,000 images")
        print(f"📥 Downloading: {num_images:,} images")
        print(f"💾 Storage needed: ~{num_images * 0.5 / 1000:.1f} GB")
        print()
        
        open_images_dir = os.path.join(self.base_dir, "open_images")
        os.makedirs(open_images_dir, exist_ok=True)
        
        # Install Open Images downloader
        print("📦 Installing Open Images downloader...")
        os.system("pip install -q openimages")
        
        # Download images
        print(f"📥 Downloading to: {open_images_dir}")
        
        if categories:
            category_str = ",".join(categories)
            cmd = f"oi_download_dataset --base_dir {open_images_dir} --labels {category_str} --limit {num_images}"
        else:
            cmd = f"oi_download_dataset --base_dir {open_images_dir} --limit {num_images}"
        
        print(f"Running: {cmd}")
        os.system(cmd)
        
        return open_images_dir
    
    def download_lvis(self):
        """
        Download LVIS dataset
        - 164,000 images (uses COCO images)
        - 1,203 categories (more detailed than COCO)
        - 2M+ instance annotations
        """
        print("=" * 70)
        print("DOWNLOADING LVIS DATASET")
        print("=" * 70)
        print(f"📦 Images: 164,000 (uses COCO images with more annotations)")
        print(f"📊 Categories: 1,203 (vs COCO's 80)")
        print(f"💾 Storage needed: ~25 GB")
        print()
        
        lvis_dir = os.path.join(self.base_dir, "lvis")
        os.makedirs(lvis_dir, exist_ok=True)
        
        # LVIS annotation URLs
        urls = {
            "train": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_train.json.zip",
            "val": "https://dl.fbaipublicfiles.com/LVIS/lvis_v1_val.json.zip",
        }
        
        for split, url in urls.items():
            print(f"📥 Downloading LVIS {split} annotations...")
            dest = os.path.join(lvis_dir, f"lvis_v1_{split}.json.zip")
            if not os.path.exists(dest):
                self.download_file(url, dest)
                print(f"📦 Extracting {split}...")
                with zipfile.ZipFile(dest, 'r') as zip_ref:
                    zip_ref.extractall(lvis_dir)
        
        print(f"✅ LVIS annotations downloaded to: {lvis_dir}")
        print("ℹ️  LVIS uses COCO images. Download COCO images separately if needed.")
        
        return lvis_dir
    
    def show_dataset_info(self):
        """Show information about available datasets"""
        print("=" * 70)
        print("AVAILABLE DATASETS FOR 1M+ IMAGE TRAINING")
        print("=" * 70)
        print()
        
        datasets = [
            {
                "name": "COCO (Current)",
                "images": 330000,
                "categories": 80,
                "size_gb": 25,
                "status": "✅ Already using"
            },
            {
                "name": "Objects365",
                "images": 2000000,
                "categories": 365,
                "size_gb": 300,
                "status": "📥 Can download"
            },
            {
                "name": "Open Images V7",
                "images": 9000000,
                "categories": 600,
                "size_gb": 500,
                "status": "📥 Can download (subset)"
            },
            {
                "name": "LVIS",
                "images": 164000,
                "categories": 1203,
                "size_gb": 25,
                "status": "📥 Can download"
            },
        ]
        
        for ds in datasets:
            print(f"{ds['status']} {ds['name']}")
            print(f"   Images: {ds['images']:,}")
            print(f"   Categories: {ds['categories']}")
            print(f"   Storage: ~{ds['size_gb']} GB")
            print()
        
        print("💡 RECOMMENDED COMBINATIONS:")
        print("   • COCO (330K) + Objects365 subset (670K) = 1M images")
        print("   • COCO (330K) + Open Images subset (670K) = 1M images")
        print("   • COCO (330K) + Objects365 (600K) + LVIS (164K) = 1.1M images")
        print()
        print("⚠️  Note: Training on 1M images requires:")
        print("   • Storage: 600GB - 1TB")
        print("   • GPU: High-end (RTX 4090 or A100)")
        print("   • Time: 1-2 months for full training")
        print("   • Cost: $500-2000 if using cloud GPUs")
        print()


def main():
    parser = argparse.ArgumentParser(description="Download object detection datasets")
    parser.add_argument("--dataset", choices=["objects365", "open_images", "lvis", "info"], 
                       required=True, help="Dataset to download")
    parser.add_argument("--num_images", type=int, default=100000, 
                       help="Number of images to download (for Open Images)")
    parser.add_argument("--categories", nargs="+", 
                       help="Specific categories to download (for Open Images)")
    parser.add_argument("--base_dir", default="datasets", 
                       help="Base directory for downloads")
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(base_dir=args.base_dir)
    
    if args.dataset == "info":
        downloader.show_dataset_info()
    elif args.dataset == "objects365":
        downloader.download_objects365()
    elif args.dataset == "open_images":
        downloader.download_open_images(
            num_images=args.num_images,
            categories=args.categories
        )
    elif args.dataset == "lvis":
        downloader.download_lvis()


if __name__ == "__main__":
    # If run without arguments, show info
    import sys
    if len(sys.argv) == 1:
        downloader = DatasetDownloader()
        downloader.show_dataset_info()
        print("\n💡 Usage examples:")
        print("   python download_datasets.py --dataset info")
        print("   python download_datasets.py --dataset lvis")
        print("   python download_datasets.py --dataset open_images --num_images 100000")
        print("   python download_datasets.py --dataset open_images --num_images 50000 --categories Person Car")
    else:
        main()
