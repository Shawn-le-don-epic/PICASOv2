import os
import cv2
import numpy as np
from tqdm import tqdm
from jnd import calculate_jnd_map
import io
from PIL import Image

def prepare_residual_dataset(source_dir, output_dir, base_jpeg_quality=25):
    """
    Generates training data for residual-based compression:
    - Creates base JPEG at specified quality
    - Calculates residual (original - base)
    - Normalizes residual to [0, 1]
    - Generates JND map from original image
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    image_files = [f for f in os.listdir(source_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    print(f"Found {len(image_files)} images. Generating residual training data...")
    
    for filename in tqdm(image_files):
        # 1. Load Original Image
        img_path = os.path.join(source_dir, filename)
        img_bgr = cv2.imread(img_path)
        if img_bgr is None:
            continue
            
        # 2. Resize to 128x128 (training patch size)
        img_bgr = cv2.resize(img_bgr, (128, 128), interpolation=cv2.INTER_AREA)
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        # 3. Create Base Layer (JPEG compressed)
        pil_img = Image.fromarray(img_rgb)
        jpeg_buffer = io.BytesIO()
        pil_img.save(jpeg_buffer, format="JPEG", quality=base_jpeg_quality)
        jpeg_buffer.seek(0)
        base_pil = Image.open(jpeg_buffer).convert("RGB")
        base_np = np.array(base_pil).astype(np.float32)
        
        # 4. Calculate Residual
        orig_np = img_rgb.astype(np.float32)
        residual = orig_np - base_np  # Range: [-255, 255]
        
        # 5. Normalize Residual (CRITICAL: same as inference!)
        residual_norm = (residual + 255.0) / 510.0  # Range: [0, 1]
        
        # 6. Calculate JND Map (from original image)
        jnd_map = calculate_jnd_map(img_bgr)
        
        # 7. Save Training Pairs
        base_name = os.path.splitext(filename)[0]
        
        # Save normalized residual as float32 numpy array (preserve precision)
        residual_path = os.path.join(output_dir, f"{base_name}_residual.npy")
        np.save(residual_path, residual_norm.astype(np.float32))
        
        # Save original image (for loss calculation)
        orig_path = os.path.join(output_dir, f"{base_name}_original.png")
        cv2.imwrite(orig_path, img_bgr)
        
        # Save base layer (for reconstruction during training)
        base_path = os.path.join(output_dir, f"{base_name}_base.png")
        base_bgr = cv2.cvtColor(base_np.astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite(base_path, base_bgr)
        
        # Save JND Map
        jnd_path = os.path.join(output_dir, f"{base_name}_jnd.npy")
        np.save(jnd_path, jnd_map.astype(np.float32))

    print(f"Residual dataset preparation complete! Files saved to {output_dir}")
    print(f"Each sample has: _residual.npy, _original.png, _base.png, _jnd.npy")

if __name__ == "__main__":
    # Update these paths
    RAW_DATA_PATH = "data/DIV2K_train_HR"  # Your source images
    PROCESSED_DATA_PATH = "data/residual_training_ready"  # New folder for residual data
    
    prepare_residual_dataset(RAW_DATA_PATH, PROCESSED_DATA_PATH, base_jpeg_quality=25)