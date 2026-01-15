import tkinter as tk
from tkinter import filedialog, messagebox, Toplevel
from PIL import Image, ImageTk
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
import io
import os
import math

# Import model and JND logic
from model import JND_LIC_Lite_Autoencoder
from jnd import calculate_jnd_map
from skimage.metrics import structural_similarity as ssim

class PICASO_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("PICASO v2.1 - Full Resolution Human-in-the-Loop Compression")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # --- State Variables ---
        self.image_path = None
        self.original_image = None  # Full Resolution PIL Image
        self.compressed_image = None
        self.tk_original = None
        self.tk_compressed = None
        
        # Canvas State
        self.selection_rect = None
        self.start_x, self.start_y = None, None
        self.end_x, self.end_y = None, None
        
        # Scale factor for display (to fit large images in UI)
        self.display_scale = 1.0 
        
        self.model = self.load_model()
        self.validate_model()
        self._setup_ui()

    def _setup_ui(self):
        # Top Control Panel
        control_frame = tk.Frame(self.root, bg="#e0e0e0", pady=10)
        control_frame.pack(fill="x", side="top")

        btn_style = {"bg": "#4a7a8c", "fg": "white", "font": ("Arial", 12, "bold"), "padx": 15, "pady": 5}
        
        self.btn_load = tk.Button(control_frame, text="1. Load Image", command=self.load_image, **btn_style)
        self.btn_load.pack(side="left", padx=20)

        self.lbl_instruction = tk.Label(control_frame, text="Draw ROI ->", bg="#e0e0e0", font=("Arial", 11))
        self.lbl_instruction.pack(side="left", padx=10)

        self.btn_compress = tk.Button(control_frame, text="2. Compress (Full Res)", command=self.run_tiled_compression, **btn_style)
        self.btn_compress.pack(side="left", padx=20)

        self.btn_analyze = tk.Button(control_frame, text="3. View Maps", command=self.show_analysis_window, state="disabled", bg="#555", fg="white", font=("Arial", 10))
        self.btn_analyze.pack(side="right", padx=20)

        # Main Display Area
        display_frame = tk.Frame(self.root, bg="#f0f0f0")
        display_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left Canvas
        self.canvas_frame = tk.Frame(display_frame, bg="white", bd=2, relief="sunken")
        self.canvas_frame.pack(side="left", fill="both", expand=True, padx=5)
        self.lbl_orig = tk.Label(self.canvas_frame, text="Original (Scaled to Fit)", bg="white", font=("Arial", 10, "bold"))
        self.lbl_orig.pack(side="top", pady=5)
        self.canvas = tk.Canvas(self.canvas_frame, cursor="cross", bg="#d9d9d9")
        self.canvas.pack(fill="both", expand=True)
        
        # Right Canvas
        self.result_frame = tk.Frame(display_frame, bg="white", bd=2, relief="sunken")
        self.result_frame.pack(side="right", fill="both", expand=True, padx=5)
        self.lbl_res = tk.Label(self.result_frame, text="Output (Full Res Detail)", bg="white", font=("Arial", 10, "bold"))
        self.lbl_res.pack(side="top", pady=5)
        self.canvas_result = tk.Canvas(self.result_frame, bg="#d9d9d9")
        self.canvas_result.pack(fill="both", expand=True)

        # Status Bar
        self.status_var = tk.StringVar()
        self.status_var.set("Ready.")
        status_bar = tk.Label(self.root, textvariable=self.status_var, bd=1, relief="sunken", anchor="w", bg="#e0e0e0")
        status_bar.pack(side="bottom", fill="x")

        self.canvas.bind("<ButtonPress-1>", self.on_button_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_button_release)

    def validate_model(self):
        """Check if loaded model is compatible with residual processing pipeline"""
        if not self.model:
            return False
            
        print("\n=== Model Validation ===")
        try:
            # Create dummy inputs matching your pipeline
            dummy_residual = torch.randn(1, 3, 128, 128) * 0.1 + 0.5  # Simulate normalized residual (centered at 0.5)
            dummy_jnd = torch.rand(1, 1, 128, 128) * 0.5 + 0.25  # JND typically in [0.25, 0.75]
            
            # Test forward pass
            with torch.no_grad():
                bottleneck, jnd_f1, jnd_f2 = self.model.encode(dummy_residual, dummy_jnd)
                print(f"✓ Encode successful: bottleneck shape = {bottleneck.shape}")
                
                recon = self.model.decode(bottleneck, jnd_f1, jnd_f2)
                print(f"✓ Decode successful: output shape = {recon.shape}")
                
                # Check output range (should be [0, 1] due to Sigmoid)
                print(f"  Output range: [{recon.min():.3f}, {recon.max():.3f}]")
                
                # Check if model preserves residual characteristics
                output_mean = recon.mean().item()
                expected_mean = 0.5
                mean_diff = abs(output_mean - expected_mean)
                
                print(f"  Output mean: {output_mean:.3f} (expected ~{expected_mean:.1f})")
                
                if mean_diff > 0.15:
                    print(f"  ⚠ WARNING: Model may not be trained on residuals!")
                    print(f"             Mean deviation: {mean_diff:.3f} (should be < 0.15)")
                    print(f"             This model was likely trained on full images, not residuals.")
                    print(f"             Results will be poor. Please retrain with residual data.")
                    return False
                else:
                    print(f"  ✓ Output statistics look correct for residual processing")
                
            print("✓ Model validation passed\n")
            return True
            
        except Exception as e:
            print(f"✗ Model validation failed: {e}")
            import traceback
            traceback.print_exc()
            return False

    '''
    def load_model(self):
        try:
            model = JND_LIC_Lite_Autoencoder()
            model_path = 'picaso_v2_model.pth'
            if not os.path.exists(model_path):
                 model_path = 'best_autoencoder.pth'
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            model.eval()
            print(f"Loaded: {model_path}")
            return model
        except Exception as e:
            print(f"Error: {e}")
            return None
    '''
    def load_model(self):
        try:
            model = JND_LIC_Lite_Autoencoder()
            
            # Try v3 model first (trained on residuals)
            model_path = 'picaso_v3_residual_50e.pth'
            if not os.path.exists(model_path):
                print("Warning: v3 model not found, falling back to v2 (trained on raw images, may not work properly with residuals)")
                model_path = 'picaso_v2_model.pth'
            if not os.path.exists(model_path):
                model_path = 'best_autoencoder.pth'
                
            # Load state dict
            state_dict = torch.load(model_path, map_location=torch.device('cpu'))
            
            # Check if state dict keys match (sometimes training saves extra wrapper keys)
            if 'model_state_dict' in state_dict:
                state_dict = state_dict['model_state_dict']
            
            model.load_state_dict(state_dict, strict=False)  # strict=False to ignore minor mismatches
            model.eval()
            print(f"✓ Loaded: {model_path}")
            return model
            
        except Exception as e:
            print(f"✗ Model loading error: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png")])
        if not path: return
        self.image_path = path
        
        # Load FULL resolution
        self.original_image = Image.open(path).convert("RGB")
        w, h = self.original_image.size
        
        # Calculate display scale to fit in canvas (e.g., 500x500)
        max_disp = 500
        self.display_scale = min(max_disp/w, max_disp/h)
        disp_w, disp_h = int(w * self.display_scale), int(h * self.display_scale)
        
        disp_img = self.original_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.tk_original = ImageTk.PhotoImage(disp_img)
        
        self.canvas.config(width=disp_w, height=disp_h)
        self.canvas.create_image(0, 0, image=self.tk_original, anchor="nw")
        self.canvas.image = self.tk_original 
        
        self.status_var.set(f"Loaded {w}x{h} image. Displaying at {int(self.display_scale*100)}%. Draw ROI.")
        self.selection_rect = None

    def on_button_press(self, event):
        self.start_x, self.start_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        if self.selection_rect: self.canvas.delete(self.selection_rect)
        self.selection_rect = self.canvas.create_rectangle(self.start_x, self.start_y, self.start_x, self.start_y, outline='#00ff00', width=2)

    def on_mouse_drag(self, event):
        cur_x, cur_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)
        self.canvas.coords(self.selection_rect, self.start_x, self.start_y, cur_x, cur_y)

    def on_button_release(self, event):
        self.end_x, self.end_y = self.canvas.canvasx(event.x), self.canvas.canvasy(event.y)

    def run_tiled_compression(self):
        if not self.original_image or not self.model: return
        if self.start_x is None:
            messagebox.showerror("Error", "Select ROI first.")
            return

        self.status_var.set("Processing Full Resolution (Tiling)... this may take a moment.")
        self.root.update()

        # 1. Map ROI to Full Resolution Coordinates
        real_x1 = int(min(self.start_x, self.end_x) / self.display_scale)
        real_y1 = int(min(self.start_y, self.end_y) / self.display_scale)
        real_x2 = int(max(self.start_x, self.end_x) / self.display_scale)
        real_y2 = int(max(self.start_y, self.end_y) / self.display_scale)
        
        full_w, full_h = self.original_image.size

        # 2. Create Base Layer & Residual (Full Res)
        BASE_JPEG_QUALITY = 25
        jpeg_buffer = io.BytesIO()
        self.original_image.save(jpeg_buffer, format="JPEG", quality=BASE_JPEG_QUALITY)
        jpeg_buffer.seek(0)
        base_layer_pil = Image.open(jpeg_buffer).convert("RGB")
        
        orig_np = np.array(self.original_image).astype(np.float32)
        base_np = np.array(base_layer_pil).astype(np.float32)
        residual_np = orig_np - base_np
        
        # Normalize Residual
        residual_norm = (residual_np + 255.0) / 510.0 # 0-1 range

        # 3. Full JND and Mask Calculation
        # Note: Calculating JND on 4K image might be slow, so we resize for JND *estimation* then upscale, 
        # or compute on full res if CV2 is fast enough. Let's try full res first.
        orig_cv = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
        jnd_map = calculate_jnd_map(orig_cv) 
        
        mask = np.zeros((full_h, full_w), dtype=np.float32)
        mask[real_y1:real_y2, real_x1:real_x2] = 1.0 #ROI injector
        #print("Mask: ", mask.min(), mask.max(), mask.mean())
        
        #guided_map = np.full_like(jnd_map, 15.0) # Higher quantization step for background
        #guided_map[mask == 1] = 0.1 # Low quantization step for ROI
        #print("Guided map: ", guided_map.min(), guided_map.max(), guided_map.mean())

        # Smooth mask to give soft transitions at ROI boundaries (reduces visible seams)
        k = 21 if min(full_w, full_h) >= 21 else 11
        if k % 2 == 0:
            k += 1
        smoothed_mask = cv2.GaussianBlur(mask.astype(np.float32), (k, k), 0)
        jnd_scaled = jnd_map * 5.0
        guided_map = smoothed_mask * 0.1 + (1.0 - smoothed_mask) * jnd_scaled
        guided_map = guided_map.astype(np.float32)


        # Save maps for analysis view
        self.vis_mask = Image.fromarray((mask * 255).astype(np.uint8))
        self.vis_jnd = Image.fromarray((jnd_map * 255).astype(np.uint8))
        self.vis_guided = Image.fromarray((guided_map * 255).astype(np.uint8))
        self.vis_residual = Image.fromarray(((residual_np - residual_np.min())/(residual_np.max()-residual_np.min())*255).astype(np.uint8))
    

        # 4. Tiled Processing
        patch_size = 128
        # Pad image to multiple of 128
        pad_h = (patch_size - full_h % patch_size) % patch_size
        pad_w = (patch_size - full_w % patch_size) % patch_size
        
        # Pad arrays
        res_padded = np.pad(residual_norm, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
        guided_padded = np.pad(guided_map, ((0, pad_h), (0, pad_w)), mode='edge')
        jnd_padded = np.pad(jnd_map, ((0, pad_h), (0, pad_w)), mode='edge')
        
        recon_padded = np.zeros_like(res_padded)
        
        new_h, new_w, _ = res_padded.shape
        
        img_tx = transforms.Compose([transforms.ToTensor()])
        # Important: JND transform should NOT resize here, just Tensor
        jnd_tx = transforms.Compose([transforms.ToTensor()]) 

        rows = new_h // patch_size
        cols = new_w // patch_size
        
        total_patches = rows * cols
        count = 0

        self.quantized_bottlenecks = [] #quantized tensors

        with torch.no_grad():
            for i in range(0, new_h, patch_size):
                for j in range(0, new_w, patch_size):
                    # Extract patches
                    res_patch = res_padded[i:i+patch_size, j:j+patch_size, :]
                    guided_patch = guided_padded[i:i+patch_size, j:j+patch_size]
                    jnd_patch = jnd_padded[i:i+patch_size, j:j+patch_size]
                    
                    # Prepare Tensors (Model expects batches)
                    # Convert numpy (H,W,C) -> PIL -> Tensor (C,H,W)
                    res_patch_pil = Image.fromarray(np.uint8(res_patch * 255.0))
                    input_tensor = img_tx(res_patch_pil).unsqueeze(0)
                    
                    # Prepare JND
                    jnd_tensor = torch.from_numpy(jnd_patch).float().unsqueeze(0).unsqueeze(0)
                    guided_tensor = torch.from_numpy(guided_patch).float().unsqueeze(0).unsqueeze(0)
                    
                    # Ensure input/jnd tensors are same device/dtype as model params if needed
                    model_param_dtype = next(self.model.parameters()).dtype
                    # Cast inputs to model dtype to avoid mismatches (safe even if already same)
                    input_tensor = input_tensor.to(dtype=model_param_dtype)
                    jnd_tensor = jnd_tensor.to(dtype=model_param_dtype)
                    guided_tensor = guided_tensor.to(dtype=model_param_dtype)
                    
                    # Encode
                    bottleneck, jnd_f1, jnd_f2 = self.model.encode(input_tensor, guided_tensor)
                    #bottleneck, jnd_f1, jnd_f2 = self.model.encode(input_tensor, jnd_tensor)
                    
                    # Quantize using the guided patch
                    # Resize guided patch to latent size (16x16)
                    q_map_patch = cv2.resize(guided_patch, (bottleneck.shape[3], bottleneck.shape[2]), interpolation=cv2.INTER_NEAREST)
                    q_map_tensor = torch.from_numpy(q_map_patch).unsqueeze(0).unsqueeze(0)
                    
                    # Align q_map_tensor dtype to bottleneck (so division is valid)
                    q_map_tensor = q_map_tensor.to(dtype=bottleneck.dtype)
                    
                    quantized = torch.round(bottleneck / q_map_tensor)
                    self.quantized_bottlenecks.append(quantized.clone()) #Store for compression measurement
                    dequantized = quantized * q_map_tensor
                    
                    # IMPORTANT: ensure dequantized matches model parameter dtype (avoid Float vs Double)
                    if dequantized.dtype != model_param_dtype:
                        dequantized = dequantized.to(dtype=model_param_dtype)
                    
                    # Decode
                    recon_patch_tensor = self.model.decode(dequantized, jnd_f1, jnd_f2)
                    
                    # Tensor back to Numpy
                    recon_patch_np = recon_patch_tensor.squeeze(0).permute(1, 2, 0).numpy()
                    
                    # Place back in big array
                    recon_padded[i:i+patch_size, j:j+patch_size, :] = recon_patch_np
                    
                    count += 1
                    if count % 5 == 0:
                        self.status_var.set(f"Processing patch {count}/{total_patches}...")
                        self.root.update()

        # 5. Unpad and Reconstruct
        recon_res = recon_padded[:full_h, :full_w, :]

        # ✓ ADD DIAGNOSTIC: Check residual statistics
        print(f"\n=== Reconstruction Diagnostics ===")
        print(f"Reconstructed residual range: [{recon_res.min():.3f}, {recon_res.max():.3f}]")
        print(f"Reconstructed residual mean: {recon_res.mean():.3f} (should be ~0.5)")
        
        # Denormalize residual
        recon_res_final = (recon_res) * 510.0 - 255.0

        print(f"Denormalized residual range: [{recon_res_final.min():.1f}, {recon_res_final.max():.1f}]")
        print(f"Denormalized residual mean: {recon_res_final.mean():.1f} (should be ~0)")
        
        # Add to base
        final_np = np.clip(base_np + recon_res_final, 0, 255).astype(np.uint8)
        self.compressed_image = Image.fromarray(final_np)

        # ✓ ADD: Check how much clipping occurred
        pre_clip = base_np + recon_res_final
        clipped_pixels = np.sum((pre_clip < 0) | (pre_clip > 255))
        total_pixels = pre_clip.size
        clip_percentage = (clipped_pixels / total_pixels) * 100
        print(f"Clipped pixels: {clip_percentage:.2f}% (lower is better)")
        print("="*40 + "\n")

        total_bits_estimated = 0
        for stored_quant in self.quantized_bottlenecks:  # We'll store these in the loop
            # Estimate bits needed using entropy: -sum(p * log2(p))
            unique, counts = torch.unique(stored_quant, return_counts=True)
            probabilities = counts.float() / counts.sum()
            entropy = -torch.sum(probabilities * torch.log2(probabilities + 1e-10))
            total_bits_estimated += entropy.item() * stored_quant.numel()
        
        compressed_bitstream_size = int(total_bits_estimated / 8)  # Convert bits to bytes

        # Display Result (Scaled)
        disp_w, disp_h = int(full_w * self.display_scale), int(full_h * self.display_scale)
        disp_res = self.compressed_image.resize((disp_w, disp_h), Image.Resampling.LANCZOS)
        self.tk_compressed = ImageTk.PhotoImage(disp_res)
        self.canvas_result.config(width=disp_w, height=disp_h)
        self.canvas_result.create_image(0, 0, image=self.tk_compressed, anchor="nw")
        
        # Save & Metric
        s = ssim(np.array(self.original_image), np.array(self.compressed_image), channel_axis=2, data_range=255)
        # Compute Compression Ratio (original file size on disk vs compressed JPEG bytes at quality=25)

        total_bits_estimated = 0
        for stored_quant in self.quantized_bottlenecks:
            # Simple entropy estimation (real codec would use arithmetic/Huffman coding)
            unique_vals = stored_quant.unique()
            num_unique = len(unique_vals)
            # Bits per symbol ≈ log2(num_unique_values)
            bits_per_value = np.log2(num_unique) if num_unique > 1 else 1
            total_bits_estimated += bits_per_value * stored_quant.numel()
        
        compressed_bitstream_size = int(total_bits_estimated / 8)  # Convert to bytes
        try:
            orig_size = os.path.getsize(self.image_path) if self.image_path else None
            buf = io.BytesIO()
            self.compressed_image.save(buf, format="JPEG", quality=BASE_JPEG_QUALITY)
            comp_size = buf.getbuffer().nbytes
            # Calculate base layer size
            #base_buf = io.BytesIO()
            #base_layer_pil.save(base_buf, format="JPEG", quality=self.BASE_JPEG_QUALITY)
            #base_size = base_buf.getbuffer().nbytes
            
            # Total compressed size = base JPEG + compressed residual bitstream
            #total_compressed_size = base_size + compressed_bitstream_size
            
            #buf = io.BytesIO()
            #self.compressed_image.save(buf, format="JPEG", quality=self.BASE_JPEG_QUALITY)
            #comp_size = buf.getbuffer().nbytes

            def _hr(num):
                if not num or num <= 0:
                    return "N/A"
                for unit in ['bytes', 'KB', 'MB', 'GB', 'TB']:
                    if num < 1024.0 or unit == 'TB':
                        if unit == 'bytes':
                            return f"{num} {unit}"
                        else:
                            return f"{num / (1024.0 ** (['bytes','KB','MB','GB','TB'].index(unit))):.2f} {unit}"
                return f"{num} bytes"

            orig_hr = _hr(orig_size)
            comp_hr = _hr(comp_size)

            if orig_size and comp_size > 0:
                cr = orig_size / comp_size
                cr_str = f"{cr:.2f}x"
            
            #if orig_size and total_compressed_size > 0:
            #    cr_actual = orig_size / total_compressed_size
            #    cr_jpeg_only = orig_size / comp_size
            #    cr_str = f"{orig_size}B (Original Size) | {total_compressed_size}B (Compressed Size) | {cr_actual:.2f}x (PICASO) | {cr_jpeg_only:.2f}x (JPEG only)"
            else:
                cr_str = "N/A"
        except Exception:
            cr_str = "N/A"
            orig_hr = "N/A"
            comp_hr = "N/A"

        self.status_var.set(f"Full Res Compression Complete! SSIM: {s:.4f} | CR: {cr_str} | Original Size: {orig_size} B -> Compressed Size: {comp_size} B")
        self.btn_analyze.config(state="normal")
        
        save_path = filedialog.asksaveasfilename(defaultextension=".jpg", filetypes=[("JPEG files", "*.jpg;*.jpeg")])
        if save_path:
            #self.compressed_image.save(save_path)
            # We save at a standard "good" quality (e.g., 75).
            # PICASO's achievement is that at this quality level, 
            # your ROI will look much better than if you just saved the raw image at quality 75.
            self.compressed_image.save(save_path, format="JPEG", quality=BASE_JPEG_QUALITY)
            print(f"Saved compressed image to: {save_path}")

    def show_analysis_window(self):
        top = Toplevel(self.root)
        top.title("PICASO v2.1 Analysis Dashboard")
        top.geometry("900x400")
        
        def place_img(parent, pil_img, title):
            fr = tk.Frame(parent, bg="white", bd=1, relief="solid")
            fr.pack(side="left", padx=10, pady=10, expand=True)
            tk.Label(fr, text=title, font=("Arial", 9, "bold")).pack(side="top")
            # Resize for thumbnail
            tk_img = ImageTk.PhotoImage(pil_img.resize((200, 200), Image.Resampling.NEAREST))
            lbl = tk.Label(fr, image=tk_img)
            lbl.image = tk_img
            lbl.pack(side="bottom")

        place_img(top, self.vis_mask, "1. User Priority Mask")
        place_img(top, self.vis_jnd, "2. Base JND Map")
        place_img(top, self.vis_guided, "3. Guided JND Map")
        place_img(top, self.vis_residual, "4. Residual Map")

if __name__ == '__main__':
    root = tk.Tk()
    app = PICASO_GUI(root)
    root.mainloop()