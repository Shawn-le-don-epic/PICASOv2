# In model.py

import torch
from torch import nn

class JND_FTM(nn.Module):
    """
    Implements the JND-based Feature Transform Module from the JND-LIC paper.
    This module refines image features based on JND features.
    """
    def __init__(self, in_channels):
        super().__init__()
        # Convolutional layer for processing combined features
        self.conv_gate = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)
        self.conv_agg = nn.Conv2d(in_channels * 2, in_channels, kernel_size=3, padding=1)

    def forward(self, image_features, jnd_features):
        # Concatenate the features from both branches along the channel dimension
        combined = torch.cat([image_features, jnd_features], dim=1)

        # --- Forget Gate Logic ---
        # Generate a weight vector 'w' that reflects perceptual sensitivity
        w = torch.sigmoid(self.conv_gate(combined))
        # The forget mask represents imperceptible regions where info can be dropped
        forget_mask = 1 - w
        
        # --- Memory Gate Logic ---
        # Aggregate perceptual info from JND and image content
        f1 = torch.sigmoid(self.conv_agg(combined))
        f2 = f1 * image_features
        f3 = torch.cat([f2, jnd_features], dim=1)
        aggregated_features = torch.tanh(self.conv_agg(f3))
        #aggregated_features = torch.tanh(self.conv_agg(combined))
        # The memory mask enhances features in perceptible regions
        memory_mask = w * aggregated_features #Omg
        
        # --- Final Combination ---
        # Suppress features in imperceptible regions and add enhanced features from perceptible regions
        refined_features = (image_features * forget_mask) + memory_mask
        #print("Refined features shape:", refined_features.shape)
        return refined_features
    
# In model.py, below the JND_FTM class

# In model.py, replace the JND_LIC_Lite_Autoencoder class

class JND_LIC_Lite_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        # Layer definitions are the same
        self.img_enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, stride=2), nn.ReLU())
        self.img_enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU())
        self.jnd_enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, stride=2), nn.ReLU())
        self.jnd_enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU())
        self.ftm1 = JND_FTM(in_channels=32)
        self.ftm2 = JND_FTM(in_channels=64)
        self.bottleneck_enc = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU())
        self.bottleneck_dec = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.img_dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU())
        self.img_dec2 = nn.Sequential(nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid())
        self.jnd_dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU())

    # --- NEW: Explicit encode method ---
    def encode(self, image, jnd_map):
        img_f1 = self.img_enc1(image)
        jnd_f1 = self.jnd_enc1(jnd_map)
        refined_f1 = self.ftm1(img_f1, jnd_f1)
        
        img_f2 = self.img_enc2(refined_f1)
        jnd_f2 = self.jnd_enc2(jnd_f1)
        refined_f2 = self.ftm2(img_f2, jnd_f2)
        
        bottleneck = self.bottleneck_enc(refined_f2)
        # Return the bottleneck and the jnd features needed for decoding
        return bottleneck, jnd_f1, jnd_f2

    # --- NEW: Explicit decode method ---
    def decode(self, bottleneck, jnd_f1, jnd_f2):
        d_b = self.bottleneck_dec(bottleneck)
        
        d_f2 = self.ftm2(d_b, jnd_f2)
        img_d1 = self.img_dec1(d_f2)
        jnd_d1 = self.jnd_dec1(jnd_f2)

        d_f1 = self.ftm1(img_d1, jnd_d1)
        reconstructed_image = self.img_dec2(d_f1)
        
        return reconstructed_image

    def forward(self, image, jnd_map):
        bottleneck, jnd_f1, jnd_f2 = self.encode(image, jnd_map)
        return self.decode(bottleneck, jnd_f1, jnd_f2), bottleneck
        

'''
class JND_LIC_Lite_Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- ENCODER ---
        # Branch 1: Image Processing
        self.img_enc1 = nn.Sequential(nn.Conv2d(3, 32, 3, padding=1, stride=2), nn.ReLU()) # 128x128 -> 64x64
        self.img_enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU()) # 64x64 -> 32x32

        # Branch 2: JND Map Processing (starts with 1 channel - grayscale)
        self.jnd_enc1 = nn.Sequential(nn.Conv2d(1, 32, 3, padding=1, stride=2), nn.ReLU()) # 128x128 -> 64x64
        self.jnd_enc2 = nn.Sequential(nn.Conv2d(32, 64, 3, padding=1, stride=2), nn.ReLU()) # 64x64 -> 32x32
        
        # JND Feature Transform Modules
        self.ftm1 = JND_FTM(in_channels=32)
        self.ftm2 = JND_FTM(in_channels=64)
        
        # Bottleneck layer
        self.bottleneck_enc = nn.Sequential(nn.Conv2d(64, 128, 3, padding=1, stride=2), nn.ReLU()) # 32x32 -> 16x16
        
        # --- DECODER ---
        self.bottleneck_dec = nn.Sequential(nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1), nn.ReLU()) # 16x16 -> 32x32

        # Symmetrical decoder paths
        self.img_dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU()) # 32x32 -> 64x64
        self.img_dec2 = nn.Sequential(nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1), nn.Sigmoid()) # 64x64 -> 128x128

        self.jnd_dec1 = nn.Sequential(nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1), nn.ReLU()) # 32x32 -> 64x64

    def forward(self, image, jnd_map):
        # --- Encoding Path ---
        # Stage 1
        img_f1 = self.img_enc1(image)
        jnd_f1 = self.jnd_enc1(jnd_map)
        refined_f1 = self.ftm1(img_f1, jnd_f1)
        
        # Stage 2
        img_f2 = self.img_enc2(refined_f1)
        jnd_f2 = self.jnd_enc2(jnd_f1) # JND features can be passed down
        refined_f2 = self.ftm2(img_f2, jnd_f2)
        
        # Bottleneck
        bottleneck = self.bottleneck_enc(refined_f2)
        
        # --- Decoding Path ---
        d_b = self.bottleneck_dec(bottleneck)
        
        # Here we re-use the JND features from the encoder to guide the decoder
        d_f2 = self.ftm2(d_b, jnd_f2)
        img_d1 = self.img_dec1(d_f2)
        jnd_d1 = self.jnd_dec1(jnd_f2)

        d_f1 = self.ftm1(img_d1, jnd_d1)
        reconstructed_image = self.img_dec2(d_f1)
        
        return reconstructed_image, bottleneck
'''