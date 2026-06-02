"""
CycleGAN Integration Bridge for GLCD-DA Change Detection Framework
=====================================================================

This module serves as a non-intrusive bridge for integrating CycleGAN image transfer
into the existing GLCD-DA framework. It maintains complete backward compatibility with
existing code while enabling optional image transfer preprocessing.

Architecture Overview:
======================

GLCD-DA Framework has TWO stages:
1. IMAGE TRANSFER NETWORK (CycleGAN) - Optional preprocessing
2. CHANGE DETECTION NETWORK (GLCD-DA) - Main change detection

This bridge enables flexible activation/deactivation of image transfer without modifying:
- main_cd.py
- models/trainer.py
- models/networks.py
- datasets/CD_dataset.py

Reference: Paper describes Algorithm 1 (Image Transfer from Optical to SAR)
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import numpy as np
from models.cyclegan import CycleGAN, CycleGANLoss
from misc.logger_tool import Logger


class CycleGANImageTransferModule:
    """
    Standalone CycleGAN module for image-to-image translation between optical and SAR.
    
    This module can be optionally initialized and called independently without affecting
    the main change detection training pipeline.
    
    Key Features:
    - Independent training/inference
    - Optional activation flag
    - Configurable loss weights
    - Automatic device management
    - Checkpoint saving/loading
    """
    
    def __init__(self, 
                 input_nc=3,
                 output_nc=3,
                 ngf=64,
                 ndf=64,
                 lambda_cycle=10.0,
                 lambda_identity=0.5,
                 device='cuda',
                 checkpoint_dir=None,
                 enabled=False):
        """
        Initialize CycleGAN Image Transfer Module
        
        Args:
            input_nc (int): Input channels (RGB=3)
            output_nc (int): Output channels (RGB=3)
            ngf (int): Generator filters (base)
            ndf (int): Discriminator filters (base)
            lambda_cycle (float): Cycle consistency loss weight
            lambda_identity (float): Identity loss weight
            device (str): 'cuda' or 'cpu'
            checkpoint_dir (str): Directory for saving checkpoints
            enabled (bool): Enable/disable module
        """
        self.enabled = enabled
        self.device = device
        self.checkpoint_dir = checkpoint_dir
        
        if not self.enabled:
            print("[CycleGAN] Module disabled - image transfer will be skipped")
            return
        
        print("[CycleGAN] Initializing CycleGAN Image Transfer Module...")
        
        # Initialize CycleGAN model
        self.model = CycleGAN(
            input_nc=input_nc,
            output_nc=output_nc,
            ngf=ngf,
            ndf=ndf,
            n_blocks=6,
            use_dropout=False
        ).to(device)
        
        # Initialize loss function
        self.loss_fn = CycleGANLoss(
            lambda_cycle=lambda_cycle,
            lambda_identity=lambda_identity
        ).to(device)
        
        # Training states
        self.optimizer_G = None
        self.optimizer_D_A = None
        self.optimizer_D_B = None
        self.epoch = 0
        self.global_step = 0
        
        print("[CycleGAN] Module initialized successfully")
    
    def setup_optimizers(self, lr=0.0005, betas=(0.5, 0.999)):
        """
        Setup Adam optimizers for generators and discriminators.
        
        Args:
            lr (float): Learning rate
            betas (tuple): Adam betas
        """
        if not self.enabled:
            return
        
        # Generator optimizer (both netG_A and netG_B share one optimizer)
        self.optimizer_G = optim.Adam(
            list(self.model.netG_A.parameters()) + list(self.model.netG_B.parameters()),
            lr=lr,
            betas=betas
        )
        
        # Discriminator optimizers
        self.optimizer_D_A = optim.Adam(
            self.model.netD_A.parameters(),
            lr=lr,
            betas=betas
        )
        
        self.optimizer_D_B = optim.Adam(
            self.model.netD_B.parameters(),
            lr=lr,
            betas=betas
        )
        
        print(f"[CycleGAN] Optimizers initialized (lr={lr})")
    
    def transfer_optical_to_sar(self, optical_image):
        """
        Transfer optical image to SAR-like image (netG_A: optical → SAR).
        
        Args:
            optical_image (torch.Tensor): Optical image [B, 3, H, W] in [-1, 1]
        
        Returns:
            torch.Tensor: SAR-like image [B, 3, H, W]
        """
        if not self.enabled:
            return optical_image
        
        with torch.no_grad():
            sar_like = self.model.netG_A(optical_image)
        
        return sar_like
    
    def transfer_sar_to_optical(self, sar_image):
        """
        Transfer SAR image to optical-like image (netG_B: SAR → optical).
        
        Args:
            sar_image (torch.Tensor): SAR image [B, 3, H, W] in [-1, 1]
        
        Returns:
            torch.Tensor: Optical-like image [B, 3, H, W]
        """
        if not self.enabled:
            return sar_image
        
        with torch.no_grad():
            optical_like = self.model.netG_B(sar_image)
        
        return optical_like
    
    def train_step(self, real_A, real_B):
        """
        Single training step: update generators and discriminators.
        
        Paper Algorithm 1:
        - Step 1-2: Encode optical, generate SAR-like, compute reconstruction loss
        - Step 3-4: Discriminate SAR-like vs real SAR, compute adversarial loss
        - Step 4: Cycle consistency: SAR-like → optical-like → SAR-like
        
        Args:
            real_A (torch.Tensor): Real optical images [B, 3, H, W]
            real_B (torch.Tensor): Real SAR images [B, 3, H, W]
        
        Returns:
            dict: Loss values for logging
        """
        if not self.enabled:
            return None
        
        self.model.train()
        
        # Forward pass through CycleGAN
        output = self.model(real_A, real_B)
        
        # ========== Update Generators ==========
        self.optimizer_G.zero_grad()
        
        # Generator loss
        loss_G = self.loss_fn(
            real_A, output['fake_B'], output['rec_A'],
            real_B, output['fake_A'], output['rec_B'],
            output['identity_A'], output['identity_B'],
            self.model.netD_A, self.model.netD_B,
            is_training_generator=True
        )
        
        loss_G.backward()
        self.optimizer_G.step()
        
        # ========== Update Discriminators ==========
        self.optimizer_D_A.zero_grad()
        self.optimizer_D_B.zero_grad()
        
        # Discriminator loss
        loss_D = self.loss_fn(
            real_A, output['fake_B'], output['rec_A'],
            real_B, output['fake_A'], output['rec_B'],
            output['identity_A'], output['identity_B'],
            self.model.netD_A, self.model.netD_B,
            is_training_generator=False
        )
        
        loss_D.backward()
        self.optimizer_D_A.step()
        self.optimizer_D_B.step()
        
        self.global_step += 1
        
        return {
            'loss_G': loss_G.item(),
            'loss_D': loss_D.item()
        }
    
    def save_checkpoint(self, ckpt_name='cyclegan_ckpt.pt'):
        """Save CycleGAN checkpoint."""
        if not self.enabled or not self.checkpoint_dir:
            return
        
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
        torch.save({
            'epoch': self.epoch,
            'global_step': self.global_step,
            'model_netG_A': self.model.netG_A.state_dict(),
            'model_netG_B': self.model.netG_B.state_dict(),
            'model_netD_A': self.model.netD_A.state_dict(),
            'model_netD_B': self.model.netD_B.state_dict(),
            'optimizer_G': self.optimizer_G.state_dict(),
            'optimizer_D_A': self.optimizer_D_A.state_dict(),
            'optimizer_D_B': self.optimizer_D_B.state_dict(),
        }, ckpt_path)
        print(f"[CycleGAN] Checkpoint saved: {ckpt_path}")
    
    def load_checkpoint(self, ckpt_name='cyclegan_ckpt.pt'):
        """Load CycleGAN checkpoint."""
        if not self.enabled or not self.checkpoint_dir:
            return
        
        ckpt_path = os.path.join(self.checkpoint_dir, ckpt_name)
        if not os.path.exists(ckpt_path):
            print(f"[CycleGAN] Checkpoint not found: {ckpt_path}")
            return
        
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.model.netG_A.load_state_dict(checkpoint['model_netG_A'])
        self.model.netG_B.load_state_dict(checkpoint['model_netG_B'])
        self.model.netD_A.load_state_dict(checkpoint['model_netD_A'])
        self.model.netD_B.load_state_dict(checkpoint['model_netD_B'])
        
        if self.optimizer_G is not None:
            self.optimizer_G.load_state_dict(checkpoint['optimizer_G'])
            self.optimizer_D_A.load_state_dict(checkpoint['optimizer_D_A'])
            self.optimizer_D_B.load_state_dict(checkpoint['optimizer_D_B'])
        
        self.epoch = checkpoint['epoch']
        self.global_step = checkpoint['global_step']
        
        print(f"[CycleGAN] Checkpoint loaded: {ckpt_path}")


class ImageTransferPreprocessor:
    """
    Lightweight preprocessor that applies CycleGAN transfer in inference mode.
    
    Can be inserted into the data loading pipeline OR applied as preprocessing
    before feeding data to change detection network.
    
    Integration Points:
    1. In-memory transfer (before dataloader)
    2. On-the-fly transfer (in dataloader)
    3. Separate preprocessing stage
    """
    
    def __init__(self, cyclegan_module, transfer_mode='optical_to_sar', enabled=False):
        """
        Args:
            cyclegan_module (CycleGANImageTransferModule): Initialized CycleGAN module
            transfer_mode (str): 'optical_to_sar' or 'sar_to_optical'
            enabled (bool): Enable/disable preprocessing
        """
        self.module = cyclegan_module
        self.transfer_mode = transfer_mode
        self.enabled = enabled and cyclegan_module.enabled
        
        if self.enabled:
            self.module.model.eval()
            print(f"[Transfer Preprocessor] Initialized in {transfer_mode} mode")
    
    def preprocess_batch(self, batch):
        """
        Apply CycleGAN transfer to batch.
        
        Modifies batch in-place to replace images with transferred versions.
        
        Paper Strategy:
        - Convert optical images to SAR-like using netG_A
        - Feed both SAR-like and real SAR to change detection network
        
        Args:
            batch (dict): Batch from dataloader with keys:
                - 'A': Optical image [B, 3, H, W] or SAR-like
                - 'B': SAR image [B, 3, H, W]
                - 'L': Change label [B, 1, H, W]
                - 'name': Image name
        
        Returns:
            dict: Modified batch with transferred images
        """
        if not self.enabled:
            return batch
        
        with torch.no_grad():
            if self.transfer_mode == 'optical_to_sar':
                # Transfer A (optical) to SAR-like, keep B (SAR) unchanged
                batch['A'] = self.module.transfer_optical_to_sar(batch['A'])
            elif self.transfer_mode == 'sar_to_optical':
                # Transfer B (SAR) to optical-like, keep A unchanged
                batch['B'] = self.module.transfer_sar_to_optical(batch['B'])
        
        return batch


class CycleGANTrainer:
    """
    Standalone trainer for CycleGAN pre-training stage.
    
    Can be run BEFORE or AFTER change detection training independently.
    Does NOT interfere with CDTrainer in models/trainer.py
    
    This implements Algorithm 1 from the paper:
    Train: optical images (x), SAR images (y)
    Initialize: multi-scale discriminators Dx, Dy, generators G1, G2
    """
    
    def __init__(self, cyclegan_module, train_dataloader, val_dataloader=None, 
                 max_epochs=100, save_interval=10, logger=None):
        """
        Args:
            cyclegan_module (CycleGANImageTransferModule): CycleGAN module
            train_dataloader: DataLoader with batches {'A': optical, 'B': sar, 'L': label}
            val_dataloader: Optional validation dataloader
            max_epochs (int): Maximum training epochs
            save_interval (int): Save checkpoint every N epochs
            logger (Logger): Optional logger for writing
        """
        self.module = cyclegan_module
        self.train_loader = train_dataloader
        self.val_loader = val_dataloader
        self.max_epochs = max_epochs
        self.save_interval = save_interval
        self.logger = logger
        self.epoch = 0
        
        if not self.module.enabled:
            print("[CycleGANTrainer] Module disabled - training skipped")
            return
        
        self.module.setup_optimizers(lr=0.0005)
    
    def train(self):
        """Train CycleGAN on image transfer task."""
        if not self.module.enabled:
            return
        
        print("[CycleGANTrainer] Starting CycleGAN training...")
        
        for epoch in range(self.max_epochs):
            self.epoch = epoch
            self.module.epoch = epoch
            
            # Training loop
            total_loss_G = 0.0
            total_loss_D = 0.0
            num_batches = 0
            
            for batch_idx, batch in enumerate(self.train_loader):
                real_A = batch['A'].to(self.module.device)
                real_B = batch['B'].to(self.module.device)
                
                # Normalize to [-1, 1] if needed
                real_A = real_A * 2 - 1
                real_B = real_B * 2 - 1
                
                # Training step
                losses = self.module.train_step(real_A, real_B)
                
                if losses is not None:
                    total_loss_G += losses['loss_G']
                    total_loss_D += losses['loss_D']
                    num_batches += 1
                
                # Log progress
                if batch_idx % 50 == 0:
                    msg = f"Epoch {epoch}/{self.max_epochs}, Batch {batch_idx}/{len(self.train_loader)}, "
                    msg += f"Loss_G: {losses['loss_G']:.4f}, Loss_D: {losses['loss_D']:.4f}"
                    if self.logger:
                        self.logger.write(msg + '\n')
                    print(msg)
            
            # Epoch summary
            avg_loss_G = total_loss_G / max(num_batches, 1)
            avg_loss_D = total_loss_D / max(num_batches, 1)
            msg = f"Epoch {epoch} Summary - Loss_G: {avg_loss_G:.4f}, Loss_D: {avg_loss_D:.4f}"
            
            if self.logger:
                self.logger.write(msg + '\n')
            print(msg)
            
            # Save checkpoint
            if (epoch + 1) % self.save_interval == 0:
                self.module.save_checkpoint(f'cyclegan_ckpt_epoch_{epoch}.pt')
        
        print("[CycleGANTrainer] Training completed")


# ============================================================================
# INTEGRATION HELPER: How to use this module
# ============================================================================

"""
USAGE GUIDE:
============

Option 1: DISABLE Image Transfer (Default - No changes to existing code)
----------
Enable = False in args/config
→ CycleGAN module remains disabled
→ Change detection network uses original images
→ Existing pipeline continues unchanged

Option 2: PRE-TRAIN CycleGAN (Separate Stage)
----------
Step 1: Create and train CycleGAN
    from models.cyclegan_bridge import CycleGANImageTransferModule, CycleGANTrainer
    
    # Initialize module
    cyclegan_module = CycleGANImageTransferModule(
        device='cuda',
        checkpoint_dir=args.checkpoint_dir,
        enabled=True
    )
    cyclegan_module.setup_optimizers(lr=0.0005)
    
    # Train CycleGAN
    trainer = CycleGANTrainer(
        cyclegan_module,
        train_dataloader,
        max_epochs=100
    )
    trainer.train()

Step 2: Use pretrained CycleGAN in change detection
    # Load checkpoint
    cyclegan_module.load_checkpoint('cyclegan_ckpt.pt')
    
    # Create preprocessor
    preprocessor = ImageTransferPreprocessor(
        cyclegan_module,
        transfer_mode='optical_to_sar',
        enabled=True
    )
    
    # Option A: Preprocess before dataloader
    for batch in dataloader:
        batch = preprocessor.preprocess_batch(batch)
        # Feed to change detection network
    
    # Option B: Use in training loop
    for batch in dataloader:
        preprocessed_batch = preprocessor.preprocess_batch(batch)
        cd_output = cd_model(preprocessed_batch['A'], preprocessed_batch['B'])

Option 3: END-TO-END Integration (If needed)
----------
Can be integrated into CDTrainer._forward_pass if desired:

    def _forward_pass(self, batch):
        self.batch = batch
        img_in1 = batch['A'].to(self.device)
        img_in2 = batch['B'].to(self.device)
        
        # Optional: Apply image transfer
        if self.cyclegan_preprocessor is not None:
            img_in1 = self.cyclegan_preprocessor.transfer_optical_to_sar(img_in1)
        
        self.G_pred_64, self.G_pred_128, self.G_pred = self.net_G(img_in1, img_in2)
"""

