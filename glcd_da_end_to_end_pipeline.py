"""
GLCD-DA End-to-End Pipeline for Gloucester-1 Dataset
=======================================================

This script implements the complete GLCD-DA two-stage pipeline:
1. DATA PREPARATION: Convert Gloucester-1 dataset to correct structure
2. CYCLEGAN TRAINING: Image transfer network (Optical ↔ SAR)
3. CHANGE DETECTION: GLCD-DA network training and inference

No modifications to original GLCD-DA code files are made.
All custom logic is isolated in this standalone script.

Paper Reference:
- Algorithm 1: Image Transfer from Optical to SAR Images
- Figure 3: Change Detection Network Architecture
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
from PIL import Image
from pathlib import Path
import shutil
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import utils
from models.networks import define_G
from models.trainer import CDTrainer
from models.cyclegan_bridge import (
    CycleGANImageTransferModule,
    ImageTransferPreprocessor,
    CycleGANTrainer
)
from datasets.CD_dataset import CDDataset
from misc.logger_tool import Logger


class DatasetPreparer:
    """
    Prepare Gloucester-1 dataset according to GLCD-DA requirements.
    
    Paper Dataset Structure:
    ├─ A: Images from time t1 (optical or SAR)
    ├─ B: Images from time t2 (SAR or optical)
    ├─ label: Ground truth change maps
    └─ list: Train/val/test splits
    
    Gloucester-1 Raw Structure:
    ├─ optical.png: Optical image
    ├─ SAR.png: SAR image
    └─ label.png: Change label
    """
    
    def __init__(self, raw_data_dir, output_dir, dataset_name='Gloucester1'):
        """
        Args:
            raw_data_dir (str): Path to raw Gloucester-1 data (with optical.png, SAR.png, label.png)
            output_dir (str): Output directory for prepared dataset
            dataset_name (str): Name of dataset subdirectory
        """
        self.raw_dir = raw_data_dir
        self.output_dir = output_dir
        self.dataset_name = dataset_name
        self.dataset_path = os.path.join(output_dir, dataset_name)
        
        print("[DatasetPreparer] Initializing...")
        print(f"  Raw data dir: {self.raw_dir}")
        print(f"  Output dir: {self.output_dir}")
    
    def prepare(self, split_ratio=0.7):
        """
        Prepare dataset with proper structure.
        
        Paper Strategy:
        - Split optical image into patches (to create training data)
        - Split SAR image into corresponding patches
        - Create change labels for each patch
        - Create train/val/test splits
        
        Args:
            split_ratio (float): Train/val split ratio
        """
        print("\n[DatasetPreparer] Preparing dataset structure...")
        
        # Create directory structure
        self._create_directory_structure()
        
        # Load images
        optical_img = self._load_image(os.path.join(self.raw_dir, 'optical.png'))
        sar_img = self._load_image(os.path.join(self.raw_dir, 'SAR.png'))
        label_img = self._load_image(os.path.join(self.raw_dir, 'label.png'))
        
        print(f"  Optical image shape: {optical_img.shape}")
        print(f"  SAR image shape: {sar_img.shape}")
        print(f"  Label image shape: {label_img.shape}")
        
        # Create patches (Paper Strategy: Multi-scale patch extraction)
        patch_size = 256  # Standard size from paper
        stride = 128  # 50% overlap
        
        patches_info = self._create_patches(
            optical_img, sar_img, label_img,
            patch_size, stride
        )
        
        print(f"  Created {len(patches_info)} patches")
        
        # Create train/val split
        num_train = int(len(patches_info) * split_ratio)
        train_patches = patches_info[:num_train]
        val_patches = patches_info[num_train:]
        
        # Save patches to disk
        self._save_patches_and_labels(train_patches, 'train')
        self._save_patches_and_labels(val_patches, 'val')
        
        # Create list files (training indices)
        self._create_list_files(len(train_patches), len(val_patches))
        
        print("[DatasetPreparer] Dataset preparation complete!")
        print(f"  Train patches: {len(train_patches)}")
        print(f"  Val patches: {len(val_patches)}")
    
    def _create_directory_structure(self):
        """Create required directory structure."""
        dirs = [
            self.dataset_path,
            os.path.join(self.dataset_path, 'A_SAR'),      # SAR from t1
            os.path.join(self.dataset_path, 'B_SAR'),      # SAR from t2
            os.path.join(self.dataset_path, 'label'),      # Change labels
            os.path.join(self.dataset_path, 'list')        # Train/val/test splits
        ]
        
        for d in dirs:
            os.makedirs(d, exist_ok=True)
    
    def _load_image(self, path):
        """Load image and convert to numpy array."""
        img = Image.open(path).convert('RGB')
        return np.array(img, dtype=np.uint8)
    
    def _create_patches(self, optical, sar, label, patch_size, stride):
        """
        Extract overlapping patches from images.
        
        Paper Strategy: Extract patches to create training dataset
        """
        patches = []
        h, w = optical.shape[:2]
        
        idx = 0
        for i in range(0, h - patch_size, stride):
            for j in range(0, w - patch_size, stride):
                patch_info = {
                    'idx': idx,
                    'x': j,
                    'y': i,
                    'optical': optical[i:i+patch_size, j:j+patch_size],
                    'sar': sar[i:i+patch_size, j:j+patch_size],
                    'label': label[i:i+patch_size, j:j+patch_size]
                }
                patches.append(patch_info)
                idx += 1
        
        return patches
    
    def _save_patches_and_labels(self, patches, split):
        """Save patch images and labels to disk."""
        a_sar_dir = os.path.join(self.dataset_path, 'A_SAR')
        b_sar_dir = os.path.join(self.dataset_path, 'B_SAR')
        label_dir = os.path.join(self.dataset_path, 'label')
        
        for patch in patches:
            idx = patch['idx']
            filename = f"{split}_{idx:04d}.png"
            
            # Save SAR images (both t1 and t2 are SAR for optical-SAR detection)
            Image.fromarray(patch['sar']).save(os.path.join(a_sar_dir, filename))
            Image.fromarray(patch['sar']).save(os.path.join(b_sar_dir, filename))
            
            # Save label (normalized to 0-255)
            label_norm = (patch['label'] > 128).astype(np.uint8) * 255
            Image.fromarray(label_norm).save(os.path.join(label_dir, filename))
    
    def _create_list_files(self, num_train, num_val):
        """Create train/val/test split files."""
        list_dir = os.path.join(self.dataset_path, 'list')
        
        # Train list
        with open(os.path.join(list_dir, 'train.txt'), 'w') as f:
            for i in range(num_train):
                f.write(f"train_{i:04d}.png\n")
        
        # Val list
        with open(os.path.join(list_dir, 'val.txt'), 'w') as f:
            for i in range(num_val):
                f.write(f"val_{i:04d}.png\n")
        
        # Test list (same as val for now)
        with open(os.path.join(list_dir, 'test.txt'), 'w') as f:
            for i in range(num_val):
                f.write(f"val_{i:04d}.png\n")
        
        print(f"  Created list files in {list_dir}")


class GLCDDAPipeline:
    """
    Complete GLCD-DA two-stage pipeline.
    
    Stage 1: CycleGAN Image Transfer (Optional)
    - Convert optical to SAR-like or vice versa
    - Reduces modality gap
    
    Stage 2: GLCD-DA Change Detection
    - Extract multi-scale features (EIFM + GLIM)
    - Fuse edge and semantic information
    - Produce change detection map
    """
    
    def __init__(self, args, data_dir):
        """
        Args:
            args: Command-line arguments
            data_dir: Path to prepared dataset
        """
        self.args = args
        self.data_dir = data_dir
        self.device = torch.device(
            f"cuda:{args.gpu_ids[0]}" 
            if torch.cuda.is_available() and len(args.gpu_ids) > 0 
            else "cpu"
        )
        
        print(f"\n[GLCDDAPipeline] Device: {self.device}")
        
        # Initialize CycleGAN (optional)
        self.cyclegan_module = None
        self.cyclegan_preprocessor = None
        
        if hasattr(args, 'enable_cyclegan') and args.enable_cyclegan:
            self._init_cyclegan()
    
    def _init_cyclegan(self):
        """Initialize CycleGAN module."""
        print("\n[GLCDDAPipeline] Initializing CycleGAN module...")
        
        self.cyclegan_module = CycleGANImageTransferModule(
            input_nc=3,
            output_nc=3,
            ngf=64,
            ndf=64,
            lambda_cycle=10.0,
            lambda_identity=0.5,
            device=str(self.device),
            checkpoint_dir=self.args.checkpoint_root,
            enabled=True
        )
        
        self.cyclegan_module.setup_optimizers(lr=0.0005)
        
        self.cyclegan_preprocessor = ImageTransferPreprocessor(
            self.cyclegan_module,
            transfer_mode='optical_to_sar',
            enabled=True
        )
        
        print("[GLCDDAPipeline] CycleGAN initialized")
    
    def prepare_dataloaders(self):
        """Create dataloaders for GLCD-DA training."""
        print("\n[GLCDDAPipeline] Creating dataloaders...")
        
        # Get dataloaders using existing utility
        self.args.data_name = 'ChangeDetection'
        self.args.split = 'train'
        self.args.split_val = 'val'
        self.args.dataset = 'CDDataset'
        
        dataloaders = utils.get_loaders(self.args)
        
        print(f"  Train batches: {len(dataloaders['train'])}")
        print(f"  Val batches: {len(dataloaders['val'])}")
        
        return dataloaders
    
    def train_cyclegan(self, dataloaders, num_epochs=50):
        """
        Train CycleGAN for image transfer.
        
        Implements Algorithm 1 from paper
        """
        if self.cyclegan_module is None or not self.cyclegan_module.enabled:
            print("\n[GLCDDAPipeline] CycleGAN disabled - skipping image transfer training")
            return
        
        print("\n" + "="*80)
        print("STAGE 1: CYCLEGAN IMAGE TRANSFER TRAINING")
        print("="*80)
        
        trainer = CycleGANTrainer(
            self.cyclegan_module,
            dataloaders['train'],
            dataloaders['val'],
            max_epochs=num_epochs,
            save_interval=10
        )
        
        trainer.train()
        self.cyclegan_module.save_checkpoint('cyclegan_best.pt')
        
        print("\n[GLCDDAPipeline] CycleGAN training completed")
    
    def train_change_detection(self, dataloaders):
        """
        Train GLCD-DA change detection network.
        
        Paper Figure 3: Change Detection Network Architecture
        - Siamese ResNet18 backbone
        - Edge-Injected Fusion Module (EIFM)
        - Global-Local Interaction Module (GLIM)
        - Feature Fusion Module (FFM)
        """
        print("\n" + "="*80)
        print("STAGE 2: GLCD-DA CHANGE DETECTION TRAINING")
        print("="*80)
        
        # Use existing CDTrainer from models/trainer.py
        # This maintains backward compatibility
        trainer = CDTrainer(args=self.args, dataloaders=dataloaders)
        
        # Optionally inject CycleGAN preprocessor into trainer
        if self.cyclegan_preprocessor is not None:
            trainer.cyclegan_preprocessor = self.cyclegan_preprocessor
            print("[GLCDDAPipeline] CycleGAN preprocessor injected into CDTrainer")
        
        # Train
        trainer.train_models()
        
        print("\n[GLCDDAPipeline] Change detection training completed")
    
    def run_full_pipeline(self):
        """Execute complete GLCD-DA pipeline."""
        print("\n" + "="*80)
        print("GLCD-DA COMPLETE PIPELINE")
        print("="*80)
        
        # Stage 0: Prepare dataloaders
        dataloaders = self.prepare_dataloaders()
        
        # Stage 1: Train CycleGAN (if enabled)
        if hasattr(self.args, 'enable_cyclegan') and self.args.enable_cyclegan:
            self.train_cyclegan(dataloaders, num_epochs=self.args.cyclegan_epochs)
            print("\n[GLCDDAPipeline] CycleGAN training completed, proceeding to change detection...")
        
        # Stage 2: Train Change Detection
        self.train_change_detection(dataloaders)
        
        print("\n" + "="*80)
        print("PIPELINE COMPLETE!")
        print("="*80)


def main():
    """
    Main execution function.
    
    Two workflows:
    1. Data Preparation: gloucester-1 → GLCD-DA format
    2. Training: CycleGAN + GLCD-DA
    """
    
    # ========================================================================
    # CONFIGURATION
    # ========================================================================
    
    # Paths (you should update these)
    raw_gloucester_1_path = './gloucester-1'  # Raw dataset location
    output_data_dir = './datasets_prepared'   # Output dataset directory
    
    # ========================================================================
    # STEP 1: PREPARE DATASET
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 1: PREPARE GLOUCESTER-1 DATASET")
    print("="*80)
    
    preparer = DatasetPreparer(
        raw_data_dir=raw_gloucester_1_path,
        output_dir=output_data_dir,
        dataset_name='Gloucester1'
    )
    
    preparer.prepare(split_ratio=0.7)
    
    # Update data_config.py to point to prepared dataset
    prepared_dataset_path = os.path.join(output_data_dir, 'Gloucester1')
    print(f"\nPrepared dataset path: {prepared_dataset_path}")
    print("Update data_config.py to use this path in get_data_config()")
    
    # ========================================================================
    # STEP 2: TRAIN GLCD-DA PIPELINE
    # ========================================================================
    
    print("\n" + "="*80)
    print("STEP 2: TRAIN GLCD-DA PIPELINE")
    print("="*80)
    
    # Create arguments (simulating command line)
    class Args:
        # Device
        gpu_ids = [0]  # CUDA GPU ID
        
        # Dataset
        data_name = 'ChangeDetection'
        dataset = 'CDDataset'
        split = 'train'
        split_val = 'val'
        
        # Model
        net_G = 'base_transformer_pos_s4'  # From paper
        n_class = 2
        
        # Training
        batch_size = 4  # Adjust based on GPU memory
        img_size = 256
        max_epochs = 100
        optimizer = 'sgd'
        lr = 0.01
        lr_policy = 'step'
        lr_decay_iters = 10
        gamma = 0.95
        loss = 'multi'  # From paper: edge + binary loss
        num_workers = 4
        
        # Checkpoints
        checkpoint_root = './CHECKPOINTS_GLOUCESTER1'
        
        # CycleGAN (optional)
        enable_cyclegan = True  # Set to True to enable image transfer
        cyclegan_epochs = 10    # Pre-training epochs
    
    args = Args()
    
    # Create pipeline
    pipeline = GLCDDAPipeline(args, prepared_dataset_path)
    
    # Run full pipeline
    pipeline.run_full_pipeline()
    
    print("\n✅ GLCD-DA Pipeline complete!")
    print(f"Checkpoints saved to: {args.checkpoint_root}")


if __name__ == '__main__':
    main()
