# In jnd.py
import numpy as np
import cv2

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