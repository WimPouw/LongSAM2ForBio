import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import random
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns

class SAM2DataProcessor:
    """
    Utility class to process and validate SAM2-generated COCO format data
    before training Detectron2 models.
    """
    
    def __init__(self):
        self.stats = {}
    
    def validate_coco_format(self, json_path: str, images_dir: str) -> Dict:
        """
        Validate COCO format annotations and return statistics.
        
        Args:
            json_path: Path to COCO annotations JSON
            images_dir: Path to images directory
            
        Returns:
            Dictionary with validation results and statistics
        """
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        stats = {
            'valid': True,
            'errors': [],
            'warnings': [],
            'num_images': len(data.get('images', [])),
            'num_annotations': len(data.get('annotations', [])),
            'num_categories': len(data.get('categories', [])),
            'category_distribution': defaultdict(int),
            'missing_images': [],
            'images_without_annotations': [],
            'bbox_size_stats': {'widths': [], 'heights': [], 'areas': []}
        }
        
        # Check required fields
        required_keys = ['info', 'images', 'annotations', 'categories']
        for key in required_keys:
            if key not in data:
                stats['errors'].append(f"Missing required key: {key}")
                stats['valid'] = False
        
        if not stats['valid']:
            return stats
        
        # Create lookups
        image_ids = {img['id']: img['file_name'] for img in data['images']}
        category_ids = {cat['id']: cat['name'] for cat in data['categories']}
        annotations_by_image = defaultdict(list)
        
        # Process annotations
        for ann in data['annotations']:
            image_id = ann['image_id']
            category_id = ann['category_id']
            
            # Check if image exists
            if image_id not in image_ids:
                stats['errors'].append(f"Annotation refers to non-existent image ID: {image_id}")
                continue
            
            # Check if category exists
            if category_id not in category_ids:
                stats['errors'].append(f"Annotation refers to non-existent category ID: {category_id}")
                continue
            
            # Count category distribution
            stats['category_distribution'][category_ids[category_id]] += 1
            annotations_by_image[image_id].append(ann)
            
            # Collect bbox statistics
            if 'bbox' in ann:
                x, y, w, h = ann['bbox']
                stats['bbox_size_stats']['widths'].append(w)
                stats['bbox_size_stats']['heights'].append(h)
                stats['bbox_size_stats']['areas'].append(w * h)
        
        # Check for missing image files
        for image_id, filename in image_ids.items():
            image_path = os.path.join(images_dir, filename)
            if not os.path.exists(image_path):
                stats['missing_images'].append(filename)
        
        # Check for images without annotations
        for image_id in image_ids:
            if image_id not in annotations_by_image:
                stats['images_without_annotations'].append(image_ids[image_id])
        
        # Set warnings
        if stats['missing_images']:
            stats['warnings'].append(f"{len(stats['missing_images'])} image files not found")
        
        if stats['images_without_annotations']:
            stats['warnings'].append(f"{len(stats['images_without_annotations'])} images have no annotations")
        
        if len(stats['missing_images']) > 0:
            stats['valid'] = False
        
        self.stats = stats
        return stats
    
    def print_validation_report(self, stats: Dict) -> None:
        """Print a detailed validation report."""
        print("=" * 50)
        print("COCO Dataset Validation Report")
        print("=" * 50)
        
        print(f"✓ Dataset Valid: {'Yes' if stats['valid'] else 'No'}")
        print(f"✓ Images: {stats['num_images']}")
        print(f"✓ Annotations: {stats['num_annotations']}")
        print(f"✓ Categories: {stats['num_categories']}")
        
        if stats['errors']:
            print(f"\n❌ Errors ({len(stats['errors'])}):")
            for error in stats['errors'][:10]:  # Show first 10 errors
                print(f"  - {error}")
            if len(stats['errors']) > 10:
                print(f"  ... and {len(stats['errors']) - 10} more errors")
        
        if stats['warnings']:
            print(f"\n⚠️ Warnings ({len(stats['warnings'])}):")
            for warning in stats['warnings']:
                print(f"  - {warning}")
        
        print(f"\n📊 Category Distribution:")
        for category, count in sorted(stats['category_distribution'].items()):
            print(f"  - {category}: {count} annotations")
        
        if stats['bbox_size_stats']['areas']:
            areas = stats['bbox_size_stats']['areas']
            print(f"\n📏 Bounding Box Statistics:")
            print(f"  - Mean area: {sum(areas)/len(areas):.2f}")
            print(f"  - Min area: {min(areas):.2f}")
            print(f"  - Max area: {max(areas):.2f}")
    
    def split_dataset(self, 
                     json_path: str, 
                     images_dir: str,
                     output_dir: str,
                     train_ratio: float = 0.8,
                     val_ratio: float = 0.2,
                     seed: int = 42) -> Tuple[str, str]:
        """
        Split dataset into train/validation sets.
        
        Args:
            json_path: Path to original COCO annotations
            images_dir: Path to original images directory
            output_dir: Directory to save split datasets
            train_ratio: Ratio for training set
            val_ratio: Ratio for validation set
            seed: Random seed for reproducibility
            
        Returns:
            Tuple of (train_json_path, val_json_path)
        """
        random.seed(seed)
        
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Create output directories
        output_path = Path(output_dir)
        train_dir = output_path / "train"
        val_dir = output_path / "val"
        train_images_dir = train_dir / "images"
        val_images_dir = val_dir / "images"
        
        for dir_path in [train_dir, val_dir, train_images_dir, val_images_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Split images
        images = data['images']
        random.shuffle(images)
        
        split_idx = int(len(images) * train_ratio)
        train_images = images[:split_idx]
        val_images = images[split_idx:]
        
        print(f"Splitting {len(images)} images into {len(train_images)} train and {len(val_images)} val")
        
        # Create image ID sets
        train_image_ids = {img['id'] for img in train_images}
        val_image_ids = {img['id'] for img in val_images}
        
        # Split annotations
        train_annotations = []
        val_annotations = []
        
        for ann in data['annotations']:
            if ann['image_id'] in train_image_ids:
                train_annotations.append(ann)
            elif ann['image_id'] in val_image_ids:
                val_annotations.append(ann)
        
        # Copy image files
        for img in train_images:
            src = os.path.join(images_dir, img['file_name'])
            dst = train_images_dir / img['file_name']
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        for img in val_images:
            src = os.path.join(images_dir, img['file_name'])
            dst = val_images_dir / img['file_name']
            if os.path.exists(src):
                shutil.copy2(src, dst)
        
        # Create train JSON
        train_data = {
            'info': data['info'],
            'licenses': data.get('licenses', []),
            'categories': data['categories'],
            'images': train_images,
            'annotations': train_annotations
        }
        
        train_json_path = train_dir / "annotations.json"
        with open(train_json_path, 'w') as f:
            json.dump(train_data, f, indent=2)
        
        # Create val JSON
        val_data = {
            'info': data['info'],
            'licenses': data.get('licenses', []),
            'categories': data['categories'],
            'images': val_images,
            'annotations': val_annotations
        }
        
        val_json_path = val_dir / "annotations.json"
        with open(val_json_path, 'w') as f:
            json.dump(val_data, f, indent=2)
        
        print(f"Train set: {len(train_images)} images, {len(train_annotations)} annotations")
        print(f"Val set: {len(val_images)} images, {len(val_annotations)} annotations")
        print(f"Train data saved to: {train_json_path}")
        print(f"Val data saved to: {val_json_path}")
        
        return str(train_json_path), str(val_json_path)
    
    def visualize_dataset_stats(self, output_path: Optional[str] = None) -> None:
        """
        Create visualizations of dataset statistics.
        
        Args:
            output_path: Path to save plot (optional)
        """
        if not self.stats:
            print("No statistics available. Run validate_coco_format() first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Category distribution
        categories = list(self.stats['category_distribution'].keys())
        counts = list(self.stats['category_distribution'].values())
        
        axes[0, 0].bar(categories, counts)
        axes[0, 0].set_title('Category Distribution')
        axes[0, 0].set_xlabel('Category')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].tick_params(axis='x', rotation=45)
        
        # Bbox width distribution
        widths = self.stats['bbox_size_stats']['widths']
        if widths:
            axes[0, 1].hist(widths, bins=50, alpha=0.7)
            axes[0, 1].set_title('Bounding Box Width Distribution')
            axes[0, 1].set_xlabel('Width (pixels)')
            axes[0, 1].set_ylabel('Frequency')
        
        # Bbox height distribution
        heights = self.stats['bbox_size_stats']['heights']
        if heights:
            axes[1, 0].hist(heights, bins=50, alpha=0.7, color='orange')
            axes[1, 0].set_title('Bounding Box Height Distribution')
            axes[1, 0].set_xlabel('Height (pixels)')
            axes[1, 0].set_ylabel('Frequency')
        
        # Bbox area distribution
        areas = self.stats['bbox_size_stats']['areas']
        if areas:
            axes[1, 1].hist(areas, bins=50, alpha=0.7, color='green')
            axes[1, 1].set_title('Bounding Box Area Distribution')
            axes[1, 1].set_xlabel('Area (pixels²)')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
        
        plt.show()

def main():
    """Example usage of data processing utilities."""
    processor = SAM2DataProcessor()
    
    # Validate your dataset
    stats = processor.validate_coco_format(
        json_path="path/to/your/annotations.json",
        images_dir="path/to/your/images/"
    )
    
    # Print validation report
    processor.print_validation_report(stats)
    
    # Visualize dataset statistics
    processor.visualize_dataset_stats("dataset_stats.png")
    
    # Split dataset if validation passed
    if stats['valid']:
        train_json, val_json = processor.split_dataset(
            json_path="path/to/your/annotations.json",
            images_dir="path/to/your/images/",
            output_dir="./split_dataset/",
            train_ratio=0.8,
            val_ratio=0.2
        )
        print(f"Dataset split complete!")
        print(f"Train: {train_json}")
        print(f"Val: {val_json}")

if __name__ == "__main__":
    main()