import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import lpips
from model import Autoencoder, AutoencoderWithSkip # Import the model we just created

# --- 1. Create a custom Dataset to load our images ---
class ImageDataset(Dataset):
    def __init__(self, directory_path, transform=None):
        self.directory_path = directory_path
        self.image_files = [f for f in os.listdir(directory_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.directory_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

# --- 2. Define the main training function ---
def train_model(data_dir, num_epochs=100, batch_size=16, learning_rate=1e-3):
    # Define transformations: resize images to 128x128 and convert to PyTorch tensors
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
    ])
    
    # Create the dataset and dataloader
    dataset = ImageDataset(directory_path=data_dir, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Initialize the model, loss function, and optimizer
    model = AutoencoderWithSkip()
    #criterion = nn.MSELoss() # Mean Squared Error is a good choice for image reconstruction
    #optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    mse_loss = nn.MSELoss()
    perceptual_loss = lpips.LPIPS(net='vgg') # Use the VGG network for feature extraction

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    print("Starting training...")
    # The main training loop
    '''
    for epoch in range(num_epochs):
        for data in dataloader:
            # The 'data' is our original image
            original_images = data
            
            # Get the reconstructed images from the model
            reconstructed_images = model(original_images)
            
            # Calculate the loss (the difference between original and reconstructed)
            loss = criterion(reconstructed_images, original_images)
            
            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
    '''
    # Using LPIPS loss for better perceptual quality
    for epoch in range(num_epochs):
        for data in dataloader:
            original_images = data
            reconstructed_images = model(original_images)

            # --- NEW: Calculate a combined loss ---
            loss_mse = mse_loss(reconstructed_images, original_images)
            loss_lpips = perceptual_loss(reconstructed_images, original_images).mean()

            # We combine them. The 0.5 is a weight you can tune.
            # This tells the model to care about both pixel accuracy and perceptual similarity.
            loss = loss_mse + 0.5 * loss_lpips

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

        
    print("Training finished!")
    # Save the trained model for later use
    torch.save(model.state_dict(), 'autoencoder.pth')
    print("Model saved as autoencoder.pth")

# --- 3. Run the training ---
if __name__ == '__main__':
    # IMPORTANT: Create a folder named 'data' in your project directory
    # and fill it with at least 20-30 high-quality images.
    image_directory = 'data'
    train_model(image_directory)