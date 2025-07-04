#!/usr/bin/env python3
"""
SAM2 Fine-tuning Pipeline for Custom Objects
This script sets up fine-tuning SAM2 on your custom annotated objects
"""

import os
import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2

class CustomSAM2Dataset(Dataset):
    """Dataset class for fine-tuning SAM2 on custom objects"""
    
    def __init__(self, annotation_dirs, object_names_to_train, 
                 image_size=(1024, 1024), split='train'):
        """
        Args:
            annotation_dirs: List of directories containing your annotated data
            object_names_to_train: List of object names to include in training
            image_size: Target image size for training
            split: 'train' or 'val'
        """
        self.annotation_dirs = annotation_dirs
        self.object_names = object_names_to_train
        self.image_size = image_size
        self.split = split
        
        # Data augmentation
        if split == 'train':
            self.transforms = A.Compose([
                A.RandomResizedCrop(image_size[0], image_size[1], scale=(0.8, 1.0)),
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        else:
            self.transforms = A.Compose([
                A.Resize(image_size[0], image_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ])
        
        self.samples = self._load_samples()
        print(f"Loaded {len(self.samples)} samples for {split}")
    
    def _load_samples(self):
        """Load all training samples from annotation directories"""
        samples = []
        
        for ann_dir in self.annotation_dirs:
            frames_dir = ann_dir
            coco_file = os.path.join(ann_dir, "segmentation_coco.json")
            csv_file = os.path.join(ann_dir, "time_series_metrics.csv")
            
            if not os.path.exists(coco_file):
                print(f"Warning: No COCO file found in {ann_dir}")
                continue
            
            # Load COCO annotations
            with open(coco_file, 'r') as f:
                coco_data = json.load(f)
            
            # Load CSV for object names
            df = pd.read_csv(csv_file) if os.path.exists(csv_file) else None
            
            # Process annotations
            images_dict = {img['id']: img for img in coco_data['images']}
            categories_dict = {cat['id']: cat['name'] for cat in coco_data['categories']}
            
            for ann in coco_data['annotations']:
                obj_name = categories_dict.get(ann['category_id'], 'unknown')
                
                # Only include specified objects
                if obj_name not in self.object_names:
                    continue
                
                image_info = images_dict[ann['image_id']]
                image_path = os.path.join(frames_dir, image_info['file_name'])
                
                if not os.path.exists(image_path):
                    continue
                
                # Convert segmentation to mask
                mask = self._segmentation_to_mask(
                    ann['segmentation'][0], 
                    image_info['width'], 
                    image_info['height']
                )
                
                samples.append({
                    'image_path': image_path,
                    'mask': mask,
                    'bbox': ann['bbox'],
                    'object_name': obj_name,
                    'object_id': ann['category_id'],
                    'area': ann['area']
                })
        
        return samples
    
    def _segmentation_to_mask(self, segmentation, width, height):
        """Convert COCO segmentation to binary mask"""
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Reshape segmentation points
        poly = np.array(segmentation).reshape(-1, 2)
        
        # Fill polygon
        cv2.fillPoly(mask, [poly.astype(np.int32)], 1)
        
        return mask
    
    def _get_prompt_points(self, mask, bbox, num_pos=5, num_neg=5):
        """Generate prompt points for SAM2 training"""
        points = []
        labels = []
        
        # Positive points from mask
        y_coords, x_coords = np.where(mask > 0)
        if len(y_coords) > 0:
            # Sample random points from mask
            indices = np.random.choice(len(y_coords), 
                                     min(num_pos, len(y_coords)), 
                                     replace=False)
            for idx in indices:
                points.append([x_coords[idx], y_coords[idx]])
                labels.append(1)
        
        # Negative points outside mask but inside image
        x, y, w, h = bbox
        for _ in range(num_neg):
            # Sample points in expanded bbox but outside mask
            expand = 50
            px = np.random.randint(max(0, x - expand), 
                                 min(mask.shape[1], x + w + expand))
            py = np.random.randint(max(0, y - expand), 
                                 min(mask.shape[0], y + h + expand))
            
            if mask[py, px] == 0:  # Outside mask
                points.append([px, py])
                labels.append(0)
        
        return np.array(points, dtype=np.float32), np.array(labels, dtype=np.int32)
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image = cv2.imread(sample['image_path'])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load mask
        mask = sample['mask']
        
        # Apply transforms
        transformed = self.transforms(image=image, mask=mask)
        image = transformed['image']
        mask = transformed['mask']
        
        # Generate prompt points
        bbox = sample['bbox']
        points, labels = self._get_prompt_points(mask.numpy(), bbox)
        
        return {
            'image': image,
            'mask': mask,
            'points': torch.from_numpy(points),
            'labels': torch.from_numpy(labels),
            'object_name': sample['object_name'],
            'object_id': sample['object_id']
        }

def setup_sam2_training():
    """Setup SAM2 for fine-tuning"""
    
    # Install SAM2 training dependencies
    install_commands = [
        "pip install 'git+https://github.com/facebookresearch/sam2.git'",
        "pip install torch torchvision",
        "pip install albumentations",
        "pip install wandb",  # For experiment tracking
        "pip install hydra-core",
        "pip install submitit"
    ]
    
    print("Install these dependencies:")
    for cmd in install_commands:
        print(f"  {cmd}")
    
    # Download SAM2 training code
    setup_code = """
# Clone SAM2 repository with training code
git clone https://github.com/facebookresearch/sam2.git
cd sam2

# Download pre-trained checkpoints
cd checkpoints && ./download_ckpts.sh && cd ..

# Setup training environment
export SAM2_BUILD_CUDA=1
pip install -e .
"""
    
    print("\nSetup commands:")
    print(setup_code)

def create_training_config(
    annotation_dirs,
    object_names,
    output_dir="./sam2_finetuned",
    batch_size=4,
    learning_rate=1e-5,
    num_epochs=50
):
    """Create training configuration for SAM2 fine-tuning"""
    
    config = f"""
# SAM2 Fine-tuning Configuration
# Save this as: training_config.yaml

model:
  type: "sam2_hiera_l"  # or sam2_hiera_b+, sam2_hiera_s
  checkpoint: "checkpoints/sam2.1_hiera_large.pt"
  freeze_image_encoder: false  # Set to true for faster training
  freeze_memory_attention: false
  freeze_memory_encoder: false

dataset:
  name: "custom_objects"
  annotation_dirs: {annotation_dirs}
  object_names: {object_names}
  train_split: 0.8
  val_split: 0.2
  image_size: [1024, 1024]
  
training:
  batch_size: {batch_size}
  learning_rate: {learning_rate}
  num_epochs: {num_epochs}
  weight_decay: 1e-4
  warmup_steps: 1000
  
  # Loss weights
  mask_loss_weight: 20.0
  focal_loss_weight: 2.0
  dice_loss_weight: 1.0
  
  # Optimizer
  optimizer: "AdamW"
  scheduler: "cosine"
  
output:
  save_dir: "{output_dir}"
  save_every: 10  # Save checkpoint every N epochs
  eval_every: 5   # Evaluate every N epochs
  
augmentation:
  horizontal_flip: 0.5
  vertical_flip: 0.2
  rotation: 15
  brightness: 0.2
  contrast: 0.2
  
logging:
  use_wandb: true
  project_name: "sam2_custom_objects"
  experiment_name: "finetune_{{object_names}}"
"""
    
    return config

def create_training_script():
    """Create the main training script"""
    
    training_script = '''
#!/usr/bin/env python3
"""
SAM2 Fine-tuning Training Script
"""

import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
import wandb
from tqdm import tqdm
import numpy as np

# Import SAM2 components
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class SAM2FineTuner:
    def __init__(self, config_path):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.setup_model()
        self.setup_data()
        
    def setup_model(self):
        """Initialize SAM2 model for fine-tuning"""
        model_cfg = f"sam2/configs/sam2.1/{self.config['model']['type']}.yaml"
        checkpoint = self.config['model']['checkpoint']
        
        # Build model
        self.model = build_sam2(model_cfg, checkpoint, device=self.device)
        
        # Freeze components if specified
        if self.config['model']['freeze_image_encoder']:
            for param in self.model.image_encoder.parameters():
                param.requires_grad = False
                
        if self.config['model']['freeze_memory_attention']:
            for param in self.model.memory_attention.parameters():
                param.requires_grad = False
                
        if self.config['model']['freeze_memory_encoder']:
            for param in self.model.memory_encoder.parameters():
                param.requires_grad = False
    
    def setup_data(self):
        """Setup training and validation datasets"""
        dataset = CustomSAM2Dataset(
            annotation_dirs=self.config['dataset']['annotation_dirs'],
            object_names_to_train=self.config['dataset']['object_names'],
            image_size=self.config['dataset']['image_size']
        )
        
        # Split dataset
        train_size = int(self.config['dataset']['train_split'] * len(dataset))
        val_size = len(dataset) - train_size
        
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
        # Create data loaders
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=True,
            num_workers=4
        )
        
        self.val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config['training']['batch_size'],
            shuffle=False,
            num_workers=4
        )
    
    def compute_loss(self, predictions, targets):
        """Compute training loss"""
        pred_masks = predictions['masks']
        target_masks = targets['mask']
        
        # Focal loss for segmentation
        focal_loss = self.focal_loss(pred_masks, target_masks)
        
        # Dice loss
        dice_loss = self.dice_loss(pred_masks, target_masks)
        
        # Combine losses
        total_loss = (
            self.config['training']['focal_loss_weight'] * focal_loss +
            self.config['training']['dice_loss_weight'] * dice_loss
        )
        
        return total_loss
    
    def focal_loss(self, pred, target, alpha=0.25, gamma=2.0):
        """Focal loss for handling class imbalance"""
        ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = alpha * (1 - pt) ** gamma * ce_loss
        return focal_loss.mean()
    
    def dice_loss(self, pred, target, smooth=1e-5):
        """Dice loss for segmentation"""
        pred = torch.sigmoid(pred)
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        return 1 - dice
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        for batch in tqdm(self.train_loader, desc="Training"):
            # Move data to device
            images = batch['image'].to(self.device)
            masks = batch['mask'].to(self.device)
            points = batch['points'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            
            # Use SAM2 predictor interface
            predictor = SAM2ImagePredictor(self.model)
            
            batch_loss = 0
            for i in range(images.shape[0]):
                predictor.set_image(images[i].permute(1, 2, 0).cpu().numpy())
                
                pred_masks, _, _ = predictor.predict(
                    point_coords=points[i].cpu().numpy(),
                    point_labels=labels[i].cpu().numpy(),
                    multimask_output=False
                )
                
                # Convert to tensor
                pred_masks = torch.from_numpy(pred_masks).to(self.device)
                target_mask = masks[i].unsqueeze(0)
                
                # Compute loss
                loss = self.compute_loss(
                    {'masks': pred_masks}, 
                    {'mask': target_mask}
                )
                batch_loss += loss
            
            batch_loss = batch_loss / images.shape[0]
            batch_loss.backward()
            self.optimizer.step()
            
            total_loss += batch_loss.item()
        
        return total_loss / len(self.train_loader)
    
    def validate(self):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                # Similar to training but without gradients
                images = batch['image'].to(self.device)
                masks = batch['mask'].to(self.device)
                points = batch['points'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                predictor = SAM2ImagePredictor(self.model)
                
                batch_loss = 0
                for i in range(images.shape[0]):
                    predictor.set_image(images[i].permute(1, 2, 0).cpu().numpy())
                    
                    pred_masks, _, _ = predictor.predict(
                        point_coords=points[i].cpu().numpy(),
                        point_labels=labels[i].cpu().numpy(),
                        multimask_output=False
                    )
                    
                    pred_masks = torch.from_numpy(pred_masks).to(self.device)
                    target_mask = masks[i].unsqueeze(0)
                    
                    loss = self.compute_loss(
                        {'masks': pred_masks}, 
                        {'mask': target_mask}
                    )
                    batch_loss += loss
                
                batch_loss = batch_loss / images.shape[0]
                total_loss += batch_loss.item()
        
        return total_loss / len(self.val_loader)
    
    def train(self):
        """Main training loop"""
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['training']['learning_rate'],
            weight_decay=self.config['training']['weight_decay']
        )
        
        # Setup scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=self.config['training']['num_epochs']
        )
        
        # Setup logging
        if self.config['logging']['use_wandb']:
            wandb.init(
                project=self.config['logging']['project_name'],
                name=self.config['logging']['experiment_name'],
                config=self.config
            )
        
        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(self.config['training']['num_epochs']):
            print(f"\\nEpoch {epoch+1}/{self.config['training']['num_epochs']}")
            
            # Train
            train_loss = self.train_epoch()
            
            # Validate
            if (epoch + 1) % self.config['output']['eval_every'] == 0:
                val_loss = self.validate()
                
                print(f"Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
                
                # Log to wandb
                if self.config['logging']['use_wandb']:
                    wandb.log({
                        'epoch': epoch,
                        'train_loss': train_loss,
                        'val_loss': val_loss,
                        'learning_rate': self.optimizer.param_groups[0]['lr']
                    })
                
                # Save best model
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    self.save_checkpoint(epoch, 'best')
            
            # Save regular checkpoint
            if (epoch + 1) % self.config['output']['save_every'] == 0:
                self.save_checkpoint(epoch, f'epoch_{epoch+1}')
            
            self.scheduler.step()
        
        # Save final model
        self.save_checkpoint(epoch, 'final')
        
        if self.config['logging']['use_wandb']:
            wandb.finish()
    
    def save_checkpoint(self, epoch, name):
        """Save model checkpoint"""
        os.makedirs(self.config['output']['save_dir'], exist_ok=True)
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config
        }
        
        path = os.path.join(
            self.config['output']['save_dir'], 
            f'sam2_finetuned_{name}.pth'
        )
        torch.save(checkpoint, path)
        print(f"Saved checkpoint: {path}")

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python train_sam2.py <config_path>")
        sys.exit(1)
    
    config_path = sys.argv[1]
    trainer = SAM2FineTuner(config_path)
    trainer.train()
'''
    
    return training_script

def create_inference_script():
    """Create script to use fine-tuned model"""
    
    inference_script = '''
#!/usr/bin/env python3
"""
Use Fine-tuned SAM2 Model for Inference
"""

import torch
import cv2
import numpy as np
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

class FineTunedSAM2Predictor:
    def __init__(self, checkpoint_path, model_config):
        """
        Args:
            checkpoint_path: Path to fine-tuned model checkpoint
            model_config: SAM2 model config file path
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load base model
        self.model = build_sam2(model_config, checkpoint_path, device=self.device)
        
        # Load fine-tuned weights
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()
        
        # Create predictor
        self.predictor = SAM2ImagePredictor(self.model)
        
        # Store object names from config
        self.object_names = checkpoint['config']['dataset']['object_names']
        print(f"Loaded fine-tuned model for objects: {self.object_names}")
    
    def predict_objects(self, image, confidence_threshold=0.5):
        """
        Automatically predict all trained objects in image
        
        Args:
            image: Input image (BGR format)
            confidence_threshold: Minimum confidence for predictions
            
        Returns:
            Dictionary of {object_name: mask} predictions
        """
        self.predictor.set_image(image)
        
        results = {}
        
        # For automatic prediction, we need to implement object detection
        # This is a simplified version - in practice you might want to:
        # 1. Use a separate object detector
        # 2. Implement sliding window approach
        # 3. Use class activation maps
        
        # For now, we'll use a grid-based approach
        height, width = image.shape[:2]
        grid_size = 50
        
        for obj_name in self.object_names:
            best_mask = None
            best_score = 0
            
            # Try different grid positions
            for y in range(0, height, grid_size):
                for x in range(0, width, grid_size):
                    try:
                        masks, scores, _ = self.predictor.predict(
                            point_coords=np.array([[x, y]]),
                            point_labels=np.array([1]),
                            multimask_output=True
                        )
                        
                        # Select best mask
                        for mask, score in zip(masks, scores):
                            if score > best_score and score > confidence_threshold:
                                best_score = score
                                best_mask = mask
                    except:
                        continue
            
            if best_mask is not None:
                results[obj_name] = best_mask
        
        return results
    
    def predict_with_prompts(self, image, points, labels):
        """
        Predict with explicit point prompts (like original SAM2)
        """
        self.predictor.set_image(image)
        
        masks, scores, logits = self.predictor.predict(
            point_coords=points,
            point_labels=labels,
            multimask_output=False
        )
        
        return masks[0], scores[0]

# Usage example
def main():
    # Initialize fine-tuned predictor
    predictor = FineTunedSAM2Predictor(
        checkpoint_path="./sam2_finetuned/sam2_finetuned_best.pth",
        model_config="sam2/configs/sam2.1/sam2.1_hiera_l.yaml"
    )
    
    # Load test image
    image = cv2.imread("test_image.jpg")
    
    # Automatic prediction
    results = predictor.predict_objects(image)
    
    # Visualize results
    overlay = image.copy()
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    for i, (obj_name, mask) in enumerate(results.items()):
        color = colors[i % len(colors)]
        overlay[mask] = color
        print(f"Detected {obj_name}")
    
    # Show result
    result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    cv2.imshow('Fine-tuned SAM2 Results', result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
'''
    
    return inference_script

def main():
    """Main setup function"""
    print("🎯 SAM2 Fine-tuning Setup for Custom Objects")
    print("=" * 50)
    
    # Example usage
    annotation_dirs = [
        "./video1_frames",  # Your annotated video directories
        "./video2_frames",
        "./video3_frames",
    ]
    
    object_names = ["Left Eye", "Right Eye", "Nose", "Mouth"]  # Your custom object names
    
    print("\\n1. Setting up training environment...")
    setup_sam2_training()
    
    print("\\n2. Creating training configuration...")
    config = create_training_config(annotation_dirs, object_names)
    
    with open("training_config.yaml", "w") as f:
        f.write(config)
    print("✅ Saved: training_config.yaml")
    
    print("\\n3. Creating training script...")
    training_script = create_training_script()
    
    with open("train_sam2.py", "w") as f:
        f.write(training_script)
    print("✅ Saved: train_sam2.py")
    
    print("\\n4. Creating inference script...")
    inference_script = create_inference_script()
    
    with open("use_finetuned_sam2.py", "w") as f:
        f.write(inference_script)
    print("✅ Saved: use_finetuned_sam2.py")
    
    print("\\n🚀 Fine-tuning Setup Complete!")
    print("\\nNext steps:")
    print("1. Annotate multiple videos with your current tool")
    print("2. Update annotation_dirs and object_names in training_config.yaml")
    print("3. Run: python train_sam2.py training_config.yaml")
    print("4. Use fine-tuned model with: use_finetuned_sam2.py")
    
    print("\\n💡 Benefits of fine-tuning:")
    print("- Much better accuracy on your specific objects")
    print("- Potentially fewer prompts needed")
    print("- Can work in automatic mode for your objects")
    print("- Faster inference on your domain")

if __name__ == "__main__":
    main()