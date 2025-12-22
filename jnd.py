# In jnd.py
import numpy as np
import cv2

def calculate_jnd_map(image_bgr):
    # 1. Grayscale Conversion
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    
    # 2. Luminance Adaptation (LA) - Kept mostly same, standard model
    background_lum = cv2.GaussianBlur(image_gray, (7, 7), 0)
    lum_adaptation = np.zeros_like(image_gray)
    # Pitch-black or blinding-white areas hide noise better
    lum_adaptation[background_lum < 127] = 17 * (1 - (background_lum[background_lum < 127] / 127.0) ** 0.5)
    lum_adaptation[background_lum >= 127] = (3 / 128) * (background_lum[background_lum >= 127] - 127) + 3
    
    # 3. Visual Masking (VM) - UPGRADED
    # Base Paper Reference [38] uses "Visual Regularity"
    # We approximate this by penalizing "clean edges" (Canny) so they aren't compressed.
    
    # A. Calculate raw texture energy (Sobel)
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)
    gradient_mag = np.sqrt(sobelx**2 + sobely**2)
    
    # B. Calculate Structural Edge Map (Canny)
    # Edges = 1 (Protect), Texture/Flat = 0 (Hide noise)
    edges = cv2.Canny(image_bgr, 100, 200).astype(np.float32) / 255.0
    # Dilate edges slightly to protect the transition zone
    edges = cv2.dilate(edges, np.ones((3,3), np.uint8))
    
    # C. Calculate Visual Masking
    # High Gradient + NO Edge = High Masking (Grass/Texture)
    # High Gradient + Edge = Low Masking (Don't compress the outline!)
    visual_masking = (gradient_mag * 0.1) * (1.0 - (edges * 0.8))
    
    # 4. Fusion using Base Paper Equation (1) 
    C_gain = 0.3 # 
    
    # J(x) = LA(x) + VM(x) - C * min(LA(x), VM(x))
    jnd_map = lum_adaptation + visual_masking - C_gain * np.minimum(lum_adaptation, visual_masking)
    
    # Normalize for the model (0 to 1)
    jnd_map_normalized = (jnd_map - np.min(jnd_map)) / (np.max(jnd_map) - np.min(jnd_map) + 1e-6)
    
    return jnd_map_normalized

'''
def calculate_jnd_map(image_bgr):
    image_gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    background_lum = cv2.GaussianBlur(image_gray, (7, 7), 0)
    lum_adaptation = np.zeros_like(image_gray)
    lum_adaptation[background_lum < 127] = 1 - (background_lum[background_lum < 127] / 127.0) ** 0.5
    lum_adaptation[background_lum >= 127] = ((background_lum[background_lum >= 127] - 127) / 128.0) ** 0.5
    lum_adaptation = lum_adaptation * 2.5
    sobelx = cv2.Sobel(image_gray, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image_gray, cv2.CV_64F, 0, 1, ksize=5)
    texture_mask = np.sqrt(sobelx**2 + sobely**2)
    texture_mask = texture_mask / (np.max(texture_mask) + 1e-6) * 10.0
    jnd_map = lum_adaptation + texture_mask
    jnd_map_normalized = (jnd_map - np.min(jnd_map)) / (np.max(jnd_map) - np.min(jnd_map))
    return jnd_map_normalized
'''