"""Helper utilities for the enhanced dashboard.

This file provides a small, self-contained implementation of the helpers
previously in `patchcore_dashboard.py`. Keep these functions minimal and
easy to adapt to your real dataset later.
"""
from typing import Tuple
import io
import base64
import numpy as np
from PIL import Image, ImageDraw


def array_to_base64(arr: np.ndarray) -> str:
    """Convert a uint8 RGB numpy array to a base64 PNG data URI."""
    if arr.dtype != np.uint8:
        arr = (np.clip(arr, 0, 1) * 255).astype(np.uint8)
    im = Image.fromarray(arr)
    buf = io.BytesIO()
    im.save(buf, format='PNG')
    data = base64.b64encode(buf.getvalue()).decode('ascii')
    return f"data:image/png;base64,{data}"


def calculate_anomaly_map(patch_size: int, stride: int, image: np.ndarray) -> np.ndarray:
    """A simple placeholder anomaly map: computes local variance over patches.

    Returns a 2D float array with shape (H_patches, W_patches).
    """
    H, W = image.shape[0], image.shape[1]
    ph = max(1, patch_size)
    sh = max(1, stride)
    Hp = max(1, (H - ph) // sh + 1)
    Wp = max(1, (W - ph) // sh + 1)
    amap = np.zeros((Hp, Wp), dtype=float)
    gray = np.mean(image, axis=2) if image.ndim == 3 else image
    for i in range(Hp):
        for j in range(Wp):
            y = i * sh
            x = j * sh
            patch = gray[y:y + ph, x:x + ph]
            amap[i, j] = float(np.var(patch))
    # normalize to 0..1
    if amap.max() > 0:
        amap = (amap - amap.min()) / (amap.max() - amap.min())
    return amap


def draw_smiley(image: Image.Image, center: Tuple[int, int], size: int = 20, color: Tuple[int, int, int] = (255, 0, 0)) -> Image.Image:
    """Draws a simple smiley on a PIL Image at center (x,y)."""
    draw = ImageDraw.Draw(image)
    cx, cy = center
    r = size // 2
    # face
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=color, width=max(1, size // 10))
    # eyes (symmetrically)
    eye_dx = int(r * 0.4)
    eye_dy = int(r * 0.2)
    eye_r = max(1, size // 10)
    draw.ellipse((cx - eye_dx - eye_r, cy - eye_dy - eye_r, cx - eye_dx + eye_r, cy - eye_dy + eye_r), fill=color)
    draw.ellipse((cx + eye_dx - eye_r, cy - eye_dy - eye_r, cx + eye_dx + eye_r, cy - eye_dy + eye_r), fill=color)
    # smile
    draw.arc((cx - r // 1, cy - r // 1, cx + r // 1, cy + r // 1), start=20, end=160, fill=color, width=max(1, size // 12))
    return image
