"""
CycleGAN Implementation for Optical-SAR Image Translation
=========================================================

This module implements CycleGAN for unsupervised image-to-image translation 
between optical and SAR imagery. The architecture consists of:

1. Generators: ResNet-based networks that translate images between domains
2. Discriminators: PatchGAN discriminators that distinguish real from generated images
3. Loss Functions: Cycle consistency, adversarial, and identity losses

Reference: Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks
           (Zhu et al., ICCV 2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import functools


class ResnetGenerator(nn.Module):
    """
    Resnet-based generator that consists of:
    - Downsampling layers
    - Residual blocks
    - Upsampling layers
    
    Architecture designed for image translation tasks with preserved spatial dimensions.
    """
    
    def __init__(self, input_nc=3, output_nc=3, ngf=64, norm_layer=nn.InstanceNorm2d,
                 use_dropout=False, n_blocks=6, padding_type='reflect'):
        """
        Args:
            input_nc (int): Number of input channels
            output_nc (int): Number of output channels
            ngf (int): Number of generator filters
            norm_layer: Normalization layer (InstanceNorm2d for CycleGAN)
            use_dropout (bool): Use dropout in residual blocks
            n_blocks (int): Number of residual blocks
            padding_type (str): Padding type ('reflect', 'replicate', 'zero')
        """
        assert n_blocks >= 0
        super(ResnetGenerator, self).__init__()
        
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.use_dropout = use_dropout
        self.n_blocks = n_blocks
        self.padding_type = padding_type
        
        # Initial convolution layer
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]
        
        # Downsampling layers
        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                               stride=2, padding=1),
                     norm_layer(ngf * mult * 2),
                     nn.ReLU(True)]
        
        # Residual blocks
        mult = 2 ** n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                 norm_layer=norm_layer, use_dropout=use_dropout,
                                 use_bias=False)]
        
        # Upsampling layers
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                        kernel_size=3, stride=2, padding=1,
                                        output_padding=1),
                     norm_layer(int(ngf * mult / 2)),
                     nn.ReLU(True)]
        
        # Final convolution layer
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]
        
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        """Forward pass through the generator"""
        return self.model(input)


class ResnetBlock(nn.Module):
    """
    Residual block with two convolution layers and instance normalization.
    Implements the basic building block for ResNet-based generators.
    """
    
    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """
        Args:
            dim (int): Number of channels
            padding_type (str): Padding type for convolution
            norm_layer: Normalization layer
            use_dropout (bool): Use dropout
            use_bias (bool): Use bias in convolution
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer,
                                               use_dropout, use_bias)
    
    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Build a convolution block with normalization and activation"""
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                      norm_layer(dim),
                      nn.ReLU(True)]
        
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]
        
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                      norm_layer(dim)]
        
        return nn.Sequential(*conv_block)
    
    def forward(self, x):
        """Forward pass with residual connection"""
        out = x + self.conv_block(x)
        return out


class NLayerDiscriminator(nn.Module):
    """
    PatchGAN discriminator that classifies overlapping image patches.
    
    The receptive field of each output pixel corresponds to a patch on the input.
    Outputs are not a single classification but a map of classifications.
    """
    
    def __init__(self, input_nc=3, ndf=64, n_layers=3, norm_layer=nn.InstanceNorm2d):
        """
        Args:
            input_nc (int): Number of input channels
            ndf (int): Number of discriminator filters
            n_layers (int): Number of layers in discriminator
            norm_layer: Normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        
        use_bias = norm_layer == nn.InstanceNorm2d
        
        kw = 4
        padw = 1
        
        # First layer (no normalization)
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                   nn.LeakyReLU(0.2, True)]
        
        # Intermediate layers
        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                         stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]
        
        # Final layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw,
                     stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]
        
        # Classification layer
        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        
        self.model = nn.Sequential(*sequence)
    
    def forward(self, input):
        """Forward pass through discriminator"""
        return self.model(input)


class CycleGAN(nn.Module):
    """
    Complete CycleGAN model for unpaired image-to-image translation.
    
    Consists of two generators (A→B and B→A) and two discriminators (D_A and D_B).
    Uses cycle consistency loss to ensure that translating an image to another domain
    and back reconstructs the original image.
    """
    
    def __init__(self, input_nc=3, output_nc=3, ngf=64, ndf=64, n_blocks=6,
                 norm_layer=nn.InstanceNorm2d, use_dropout=False):
        """
        Args:
            input_nc (int): Number of input channels
            output_nc (int): Number of output channels
            ngf (int): Number of generator filters
            ndf (int): Number of discriminator filters
            n_blocks (int): Number of residual blocks in generator
            norm_layer: Normalization layer
            use_dropout (bool): Use dropout in generators
        """
        super(CycleGAN, self).__init__()
        
        # Generators
        self.netG_A = ResnetGenerator(input_nc, output_nc, ngf, norm_layer,
                                     use_dropout, n_blocks)
        self.netG_B = ResnetGenerator(output_nc, input_nc, ngf, norm_layer,
                                     use_dropout, n_blocks)
        
        # Discriminators
        self.netD_A = NLayerDiscriminator(output_nc, ndf, n_layers=3, norm_layer=norm_layer)
        self.netD_B = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize network weights"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.ConvTranspose2d):
                init.normal_(m.weight.data, 0.0, 0.02)
                if m.bias is not None:
                    init.constant_(m.bias.data, 0.0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.InstanceNorm2d):
                init.normal_(m.weight.data, 1.0, 0.02)
                init.constant_(m.bias.data, 0.0)
    
    def forward(self, input_A, input_B):
        """
        Forward pass through CycleGAN
        
        Args:
            input_A: Batch of images from domain A (e.g., Optical)
            input_B: Batch of images from domain B (e.g., SAR)
        
        Returns:
            Dictionary containing:
            - fake_B: Translated images from A→B
            - rec_A: Reconstructed images A→B→A
            - fake_A: Translated images from B→A
            - rec_B: Reconstructed images B→A→B
            - identity_A: Identity-mapped images from domain A
            - identity_B: Identity-mapped images from domain B
        """
        # Translate A to B and back
        fake_B = self.netG_A(input_A)
        rec_A = self.netG_B(fake_B)
        
        # Translate B to A and back
        fake_A = self.netG_B(input_B)
        rec_B = self.netG_A(fake_A)
        
        # Identity mapping (for identity loss)
        identity_B = self.netG_A(input_B)
        identity_A = self.netG_B(input_A)
        
        return {
            'fake_B': fake_B,
            'rec_A': rec_A,
            'fake_A': fake_A,
            'rec_B': rec_B,
            'identity_A': identity_A,
            'identity_B': identity_B
        }


class CycleGANLoss(nn.Module):
    """
    Loss functions for CycleGAN training.
    
    Components:
    - Adversarial loss: GAN loss between real and fake images
    - Cycle consistency loss: L1 loss between original and reconstructed images
    - Identity loss: L1 loss for identity mapping
    """
    
    def __init__(self, lambda_cycle=10.0, lambda_identity=0.5):
        """
        Args:
            lambda_cycle (float): Weight for cycle consistency loss
            lambda_identity (float): Weight for identity loss
        """
        super(CycleGANLoss, self).__init__()
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity
        
        self.criterion_gan = nn.MSELoss()
        self.criterion_cycle = nn.L1Loss()
        self.criterion_identity = nn.L1Loss()
    
    def adversarial_loss(self, discriminator_output, is_real):
        """
        Compute adversarial loss
        
        Args:
            discriminator_output: Output from discriminator
            is_real (bool): Whether target is real or fake
        
        Returns:
            Adversarial loss value
        """
        if is_real:
            target = torch.ones_like(discriminator_output)
        else:
            target = torch.zeros_like(discriminator_output)
        
        return self.criterion_gan(discriminator_output, target)
    
    def cycle_consistency_loss(self, original, reconstructed):
        """
        Compute cycle consistency loss: L1(x, G_B(G_A(x)))
        
        Args:
            original: Original images
            reconstructed: Reconstructed images
        
        Returns:
            Cycle consistency loss value
        """
        return self.criterion_cycle(original, reconstructed) * self.lambda_cycle
    
    def identity_loss(self, input_img, identity_img):
        """
        Compute identity loss: L1(x, G_A(x)) when x is from domain B
        
        Args:
            input_img: Input images
            identity_img: Identity-mapped images
        
        Returns:
            Identity loss value
        """
        return self.criterion_identity(input_img, identity_img) * self.lambda_identity
    
    def forward(self, real_A, fake_B, rec_A, real_B, fake_A, rec_B,
                identity_A, identity_B, D_A, D_B, is_training_generator=True):
        """
        Compute total loss for CycleGAN
        
        Args:
            real_A: Real images from domain A
            fake_B: Translated images A→B
            rec_A: Reconstructed images A→B→A
            real_B: Real images from domain B
            fake_A: Translated images B→A
            rec_B: Reconstructed images B→A→B
            identity_A: Identity-mapped images for domain A
            identity_B: Identity-mapped images for domain B
            D_A: Discriminator for domain A
            D_B: Discriminator for domain B
            is_training_generator (bool): Whether computing generator or discriminator loss
        
        Returns:
            Loss value
        """
        if is_training_generator:
            # Generator losses
            loss_G_A = self.adversarial_loss(D_B(fake_B), is_real=True)
            loss_G_B = self.adversarial_loss(D_A(fake_A), is_real=True)
            
            # Cycle consistency losses
            loss_cycle_A = self.cycle_consistency_loss(real_A, rec_A)
            loss_cycle_B = self.cycle_consistency_loss(real_B, rec_B)
            
            # Identity losses
            loss_idt_A = self.identity_loss(real_B, identity_B)
            loss_idt_B = self.identity_loss(real_A, identity_A)
            
            # Total generator loss
            total_loss = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B
            
            return total_loss
        else:
            # Discriminator losses
            loss_D_A_real = self.adversarial_loss(D_A(real_A), is_real=True)
            loss_D_A_fake = self.adversarial_loss(D_A(fake_A), is_real=False)
            loss_D_A = (loss_D_A_real + loss_D_A_fake) * 0.5
            
            loss_D_B_real = self.adversarial_loss(D_B(real_B), is_real=True)
            loss_D_B_fake = self.adversarial_loss(D_B(fake_B), is_real=False)
            loss_D_B = (loss_D_B_real + loss_D_B_fake) * 0.5
            
            # Total discriminator loss
            total_loss = loss_D_A + loss_D_B
            
            return total_loss


# Example usage and integration helper
def create_cyclegan_model(input_nc=3, output_nc=3, device='cuda'):
    """
    Create a CycleGAN model ready for training.
    
    Args:
        input_nc (int): Number of input channels
        output_nc (int): Number of output channels
        device (str): Device to move model to ('cuda' or 'cpu')
    
    Returns:
        Tuple of (model, loss_fn) ready for training
    """
    model = CycleGAN(input_nc=input_nc, output_nc=output_nc, ngf=64, ndf=64,
                    n_blocks=6, use_dropout=False)
    loss_fn = CycleGANLoss(lambda_cycle=10.0, lambda_identity=0.5)
    
    model = model.to(device)
    loss_fn = loss_fn.to(device)
    
    return model, loss_fn
