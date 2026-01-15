import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
import os
import cv2
import lpips

from model import JND_LIC_Lite_Autoencoder

class ResidualDataset(Dataset):
    """
    Dataset for residual-based compression training.
    Returns: (normalized_residual, jnd_map, original_image, base_image)
    """
    def __init__(self, directory_path):
        self.directory_path = directory_path
        # Find all residual files
        all_files = os.listdir(directory_path)
        self.residual_files = sorted([f for f in all_files if f.endswith('_residual.npy')])
        
        print(f"Found {len(self.residual_files)} training samples")

    def __len__(self):
        return len(self.residual_files)

    def __getitem__(self, idx):
        residual_name = self.residual_files[idx]
        base_name = residual_name.replace('_residual.npy', '')
        
        # 1. Load Normalized Residual (input to model)
        residual_path = os.path.join(self.directory_path, residual_name)
        residual_norm = np.load(residual_path).astype(np.float32)  # [H, W, 3], range [0, 1]
        residual_tensor = torch.from_numpy(residual_norm).permute(2, 0, 1)  # [3, H, W]
        
        # 2. Load JND Map (guidance for model)
        jnd_path = os.path.join(self.directory_path, f"{base_name}_jnd.npy")
        jnd_map = np.load(jnd_path).astype(np.float32)  # [H, W]
        jnd_tensor = torch.from_numpy(jnd_map).unsqueeze(0)  # [1, H, W]
        
        # 3. Load Original Image (target for loss)
        orig_path = os.path.join(self.directory_path, f"{base_name}_original.png")
        orig_bgr = cv2.imread(orig_path).astype(np.float32)
        orig_rgb = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2RGB)
        orig_tensor = torch.from_numpy(orig_rgb).permute(2, 0, 1) / 255.0  # [3, H, W], [0, 1]
        
        # 4. Load Base Image (for reconstruction)
        base_path = os.path.join(self.directory_path, f"{base_name}_base.png")
        base_bgr = cv2.imread(base_path).astype(np.float32)
        base_rgb = cv2.cvtColor(base_bgr, cv2.COLOR_BGR2RGB)
        base_tensor = torch.from_numpy(base_rgb).permute(2, 0, 1) / 255.0  # [3, H, W], [0, 1]
        
        return residual_tensor, jnd_tensor, orig_tensor, base_tensor

def train_residual_model(data_dir, num_epochs=50, batch_size=8, learning_rate=1e-4, save_path='picaso_v3_residual.pth'):
    """
    Train model on residuals with proper reconstruction loss.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")
    
    # 1. Prepare Dataset
    dataset = ResidualDataset(data_dir)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # 2. Initialize Model
    model = JND_LIC_Lite_Autoencoder().to(device)
    
    # 3. Optimizers and Loss Functions
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    mse_loss = nn.MSELoss()
    perceptual_loss = lpips.LPIPS(net='alex').to(device)  # 'alex' is faster than 'vgg'
    
    print(f"Starting Residual-Based Training for {num_epochs} epochs...")
    print(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    
    best_loss = float('inf')
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_mse = 0.0
        running_lpips = 0.0
        
        for batch_idx, (residual_batch, jnd_batch, orig_batch, base_batch) in enumerate(dataloader):
            residual_batch = residual_batch.to(device)
            jnd_batch = jnd_batch.to(device)
            orig_batch = orig_batch.to(device)
            base_batch = base_batch.to(device)
            
            # Forward pass: model reconstructs the RESIDUAL
            reconstructed_residual, bottleneck = model(residual_batch, jnd_batch)
            
            # Denormalize reconstructed residual: [0, 1] → [-255, 255]
            recon_res_denorm = (reconstructed_residual * 510.0 - 255.0) / 255.0  # Convert to [0, 1] range for loss
            
            # Reconstruct full image: base + residual
            reconstructed_full = base_batch + recon_res_denorm
            reconstructed_full = torch.clamp(reconstructed_full, 0.0, 1.0)
            
            # Calculate Loss on FULL IMAGE (this is critical!)
            l_mse = mse_loss(reconstructed_full, orig_batch)
            l_lpips = perceptual_loss(reconstructed_full, orig_batch).mean()
            
            # Combined loss
            loss = l_mse + 0.5 * l_lpips
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (helps with training stability)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            running_loss += loss.item()
            running_mse += l_mse.item()
            running_lpips += l_lpips.item()
            
            # Print progress every 50 batches
            if (batch_idx + 1) % 50 == 0:
                print(f"  Batch [{batch_idx+1}/{len(dataloader)}] Loss: {loss.item():.4f}")
        
        # Epoch summary
        avg_loss = running_loss / len(dataloader)
        avg_mse = running_mse / len(dataloader)
        avg_lpips = running_lpips / len(dataloader)
        
        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Loss: {avg_loss:.4f} | MSE: {avg_mse:.4f} | LPIPS: {avg_lpips:.4f}")
        
        # After epoch 1 and epoch 10, check residual statistics
        if epoch in [0, 9]:
            print(f"\n  Diagnostic Check:")
            print(f"    Input residual mean: {residual_batch.mean():.3f} (should be ~0.5)")
            print(f"    Output residual mean: {reconstructed_residual.mean():.3f} (should be ~0.5)")
            print(f"    Reconstructed full range: [{reconstructed_full.min():.3f}, {reconstructed_full.max():.3f}]")

        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"  ✓ Saved best model with loss: {best_loss:.4f}")
    
    print(f"\nTraining complete! Best model saved to: {save_path}")
    print(f"Best loss achieved: {best_loss:.4f}")

if __name__ == "__main__":
    DATA_PATH = "data/residual_training_ready"  # Folder with residual data
    train_residual_model(
        data_dir=DATA_PATH,
        num_epochs=50,
        batch_size=8,  # Adjust based on your GPU memory
        learning_rate=1e-4,
        save_path='picaso_v3_residual_50e.pth'
    )