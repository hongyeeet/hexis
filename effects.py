import cv2
import numpy as np

class SpaceWarper:
    def __init__(self, size=300, intensity=0.45, sigma=70.0):
        """
        Precomputes relative coordinate maps for localized gravity warping.
        size: Width/height of the warp region (must be even).
        intensity: Strengths of distortion (0.0 to 1.0).
        sigma: Spread of the warp (Gaussian falloff).
        """
        self.size = size
        self.half_size = size // 2
        W = self.half_size
        
        # Grid coordinates from -W to W-1
        grid_y, grid_x = np.mgrid[-W:W, -W:W].astype(np.float32)
        r = np.sqrt(grid_x**2 + grid_y**2)
        
        # Gravity pull mapping: pull coordinates inward
        # f(r) = 1.0 - intensity * exp(-r^2 / 2*sigma^2)
        factor = 1.0 - intensity * np.exp(-r**2 / (2.0 * sigma**2))
        factor[W, W] = 1.0 # Prevent division by zero at center
        
        self.map_x_rel = grid_x * factor
        self.map_y_rel = grid_y * factor

    def apply(self, frame, center_x, center_y):
        """
        Applies localized lens distortion around (center_x, center_y).
        """
        h, w = frame.shape[:2]
        cx, cy = int(center_x), int(center_y)
        W = self.half_size
        
        # Bounding box
        x0, x1 = cx - W, cx + W
        y0, y1 = cy - W, cy + W
        
        # Ensure the bounding box is fully within the screen limits
        if x0 >= 0 and x1 < w and y0 >= 0 and y1 < h:
            # Shift relative maps to absolute coordinates
            map_x = cx + self.map_x_rel
            map_y = cy + self.map_y_rel
            
            # Apply warp using cv2.remap (very fast)
            warped = cv2.remap(
                frame, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Paste back
            frame[y0:y1, x0:x1] = warped
            
        return frame


def process_background(bgr_frame, desaturation=0.0, darkening=1.0):
    """
    Desaturates and darkens the live camera feed.
    bgr_frame: input BGR frame from webcam
    """
    # Convert BGR to Grayscale
    gray = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    
    # Desaturate
    desat = cv2.addWeighted(bgr_frame, 1.0 - desaturation, gray_bgr, desaturation, 0)
    
    # Darken
    darkened = cv2.multiply(desat, np.array([darkening, darkening, darkening, 1.0]))
    
    return darkened


def apply_bloom_composite(background, foreground_rgb):
    """
    Applies an optimized dual-pass downsampled Gaussian Bloom to the foreground elements
    and additively composites them onto the colored camera background.
    Downsampling the blur pass achieves a 16x performance increase and a smoother glow.
    """
    # Pygame outputs RGB, but OpenCV background is BGR. Convert foreground to BGR first.
    foreground = cv2.cvtColor(foreground_rgb, cv2.COLOR_RGB2BGR)
    
    # Get original dimensions
    h, w = foreground.shape[:2]
    
    # Downsample by 4x to speed up Gaussian Blur (16x fewer pixels to process)
    fg_small = cv2.resize(foreground, (w // 4, h // 4), interpolation=cv2.INTER_AREA)
    
    # Apply narrow and wide blurs on the small image
    blur_tight = cv2.GaussianBlur(fg_small, (5, 5), 0)
    blur_wide = cv2.GaussianBlur(fg_small, (15, 15), 0)
    
    # Upsample blurred images back to original size (adds smooth interpolation/glow)
    glow_tight = cv2.resize(blur_tight, (w, h), interpolation=cv2.INTER_LINEAR)
    glow_wide = cv2.resize(blur_wide, (w, h), interpolation=cv2.INTER_LINEAR)
    
    # Blend the bloom components together (bloom = original + tight * 0.9 + wide * 0.55)
    bloom = cv2.add(foreground, cv2.multiply(glow_tight, 0.9))
    bloom = cv2.add(bloom, cv2.multiply(glow_wide, 0.55))
    
    # Additively blend onto the camera background
    composite = cv2.add(background, bloom)
    
    # Convert back to RGB for Pygame display
    composite_rgb = cv2.cvtColor(composite, cv2.COLOR_BGR2RGB)
    
    return composite_rgb

def apply_chromatic_aberration(frame, offset_x, offset_y):
    """
    Shifts the Red (index 0) and Blue (index 2) channels of an RGB image to create a chromatic glitch.
    offset_x, offset_y: pixel shifts for channel translation
    """
    ox = int(offset_x)
    oy = int(offset_y)
    if ox == 0 and oy == 0:
        return frame
        
    h, w = frame.shape[:2]
    result = frame.copy()
    
    # Shift Red channel left/up
    M_r = np.float32([[1, 0, -ox], [0, 1, -oy]])
    result[:, :, 0] = cv2.warpAffine(frame[:, :, 0], M_r, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    # Shift Blue channel right/down
    M_b = np.float32([[1, 0, ox], [0, 1, oy]])
    result[:, :, 2] = cv2.warpAffine(frame[:, :, 2], M_b, (w, h), borderMode=cv2.BORDER_REPLICATE)
    
    return result
