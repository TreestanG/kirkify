import os
from PIL import Image
import numpy as np
import cv2

def same_res(path1: str, path2: str):
    """Loads two images and resizes the second to match the first."""
    img1 = Image.open(path1).convert('RGB')
    img2 = Image.open(path2).convert('RGB')
    
    img2_resized = img2.resize(img1.size, Image.Resampling.LANCZOS)
    
    return img1, img2_resized

def to_lab_pixels(img: Image.Image):
    """Converts a PIL image to Lab color space and flattens it to a list of pixels."""
    arr_rgb = np.array(img)    
    arr_lab = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2Lab)

    # Flatten (H, W, 3) -> (H*W, 3)
    pixels = arr_lab.reshape(-1, 3)
    
    return pixels

def from_lab_pixels(pixels: np.ndarray, shape: tuple):
    """Converts flattened Lab pixels back to a PIL Image."""
    arr_lab = pixels.reshape(shape)    
    arr_rgb = cv2.cvtColor(arr_lab, cv2.COLOR_Lab2RGB)
    return Image.fromarray(arr_rgb)

def _bin_indices(pixels: np.ndarray, bins: tuple[int, int, int]):
    l_bins, a_bins, b_bins = bins
    l = pixels[:, 0].astype(np.int32)
    a = pixels[:, 1].astype(np.int32)
    b = pixels[:, 2].astype(np.int32)

    l_bin = (l * l_bins // 256)
    a_bin = (a * a_bins // 256)
    b_bin = (b * b_bins // 256)

    l_bin = np.clip(l_bin, 0, l_bins - 1)
    a_bin = np.clip(a_bin, 0, a_bins - 1)
    b_bin = np.clip(b_bin, 0, b_bins - 1)

    return l_bin, a_bin, b_bin

def match_by_lightness(source_pixels: np.ndarray, target_pixels: np.ndarray):
    """Matches pixels by L channel (lightness) only."""
    src_order = np.argsort(source_pixels[:, 0], kind="stable")
    tgt_order = np.argsort(target_pixels[:, 0], kind="stable")

    matched = np.empty_like(target_pixels)
    matched[tgt_order] = source_pixels[src_order]
    return matched

def match_by_bins(
    source_pixels: np.ndarray,
    target_pixels: np.ndarray,
    shape: tuple,
    bins: tuple[int, int, int] = (8, 8, 8),
):
    """Matches pixels using color bins + spatial coordinates to minimize travel distance."""
    h, w, _ = shape
    src_l, src_a, src_b = _bin_indices(source_pixels, bins)
    tgt_l, tgt_a, tgt_b = _bin_indices(target_pixels, bins)

    y, x = np.indices((h, w))
    y_flat = y.ravel()
    x_flat = x.ravel()

    src_order = np.lexsort((x_flat, y_flat, src_b, src_a, src_l))
    tgt_order = np.lexsort((x_flat, y_flat, tgt_b, tgt_a, tgt_l))

    matched = np.empty_like(target_pixels)
    matched[tgt_order] = source_pixels[src_order]
    
    return matched, src_order, tgt_order

def create_frame(source_pixels, src_order, tgt_order, shape, t):
    """Generates a single frame of the animation at time t (0 to 1)."""
    h, w, _ = shape
    
    y, x = np.indices((h, w))
    coords = np.stack([y.ravel(), x.ravel()], axis=1)
    
    start_coords = coords[src_order]
    end_coords = coords[tgt_order]
    current_coords = (1 - t) * start_coords + t * end_coords
    current_coords = np.round(current_coords).astype(np.int32)
    
    current_coords[:, 0] = np.clip(current_coords[:, 0], 0, h - 1)
    current_coords[:, 1] = np.clip(current_coords[:, 1], 0, w - 1)
    
    frame_arr = np.zeros((h, w, 3), dtype=np.uint8)
    source_rgb = cv2.cvtColor(source_pixels[src_order].reshape(1, -1, 3).astype(np.uint8), cv2.COLOR_Lab2RGB).reshape(-1, 3)
    
    frame_arr[current_coords[:, 0], current_coords[:, 1]] = source_rgb
    
    kernel = np.ones((2,2), np.uint8)
    frame_arr = cv2.dilate(frame_arr, kernel, iterations=1)
    
    return frame_arr

def save_animation_video(source_pixels, src_order, tgt_order, shape, filename="kirkify.mp4", fps=30, duration=5, start_pause=1.0, end_pause=1.5):
    """Direct linear interpolation of pixels from source to target."""
    h, w, _ = shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(filename, fourcc, fps, (w, h))
    
    start_frames = int(start_pause * fps)
    moving_frames = int(duration * fps)
    end_frames = int(end_pause * fps)
    
    y, x = np.indices((h, w))
    coords = np.stack([y.ravel(), x.ravel()], axis=1).astype(np.float32)
    
    start_pos = coords[src_order]
    target_pos = coords[tgt_order]
    
    source_bgr = cv2.cvtColor(
        source_pixels[src_order].reshape(1, -1, 3).astype(np.uint8), 
        cv2.COLOR_Lab2BGR
    ).reshape(-1, 3)
    
    print(f"Generating direct interpolation animation...")
    
    for i in range(start_frames + moving_frames + end_frames):
        if i < start_frames:
            frame_pos = start_pos
        elif i < start_frames + moving_frames:
            t = (i - start_frames) / (moving_frames - 1)
            t_eased = t * t * (3 - 2 * t)
            frame_pos = (1 - t_eased) * start_pos + t_eased * target_pos
        else:
            frame_pos = target_pos

        frame_arr = np.zeros((h, w, 3), dtype=np.uint8)
        render_coords = np.round(frame_pos).astype(np.int32)
        render_coords[:, 0] = np.clip(render_coords[:, 0], 0, h - 1)
        render_coords[:, 1] = np.clip(render_coords[:, 1], 0, w - 1)
        
        frame_arr[render_coords[:, 0], render_coords[:, 1]] = source_bgr
        
        kernel = np.ones((2,2), np.uint8)
        frame_arr = cv2.dilate(frame_arr, kernel, iterations=1)
        out.write(frame_arr)
            
        if i % 30 == 0:
            print(f"Rendered {i}/{start_frames + moving_frames + end_frames} frames")
            
    out.release()
    print(f"Animation saved to {filename}")
