"""Demo dataset generator for the enhanced dashboard.

Produces synthetic 'good' and 'anomalous' RGB images and small thumbnails.
"""
from typing import Dict, List, Tuple
import numpy as np
from PIL import Image, ImageDraw


def _make_base_image(size: Tuple[int, int], seed: int = 0) -> np.ndarray:
    rng = np.random.RandomState(seed)
    H, W = size
    # smooth gradient background with slight noise
    xv = np.linspace(0, 1, W)
    yv = np.linspace(0, 1, H)
    base = (np.outer(yv, xv) * 255).astype(np.uint8)
    img = np.stack([base, (base * (0.8 + rng.rand() * 0.3)).astype(np.uint8), (base * (0.6 + rng.rand() * 0.4)).astype(np.uint8)], axis=2)
    noise = (rng.randn(H, W, 3) * (4 + rng.rand() * 12)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    # add some random shapes to increase diversity
    pil = Image.fromarray(img)
    draw = ImageDraw.Draw(pil)
    for _ in range(rng.randint(1, 6)):
        x1 = rng.randint(0, W)
        y1 = rng.randint(0, H)
        x2 = rng.randint(0, W)
        y2 = rng.randint(0, H)
        shape_type = rng.choice(['ellipse', 'line', 'rectangle'])
        color = tuple(int(c) for c in rng.randint(0, 255, size=3))
        if shape_type == 'ellipse':
            x0, x1p = min(x1, x2), max(x1, x2)
            y0, y1p = min(y1, y2), max(y1, y2)
            draw.ellipse([x0, y0, x1p, y1p], outline=color)
        elif shape_type == 'line':
            draw.line([x1, y1, x2, y2], fill=color, width=max(1, rng.randint(1, 4)))
        else:
            x0, x1p = min(x1, x2), max(x1, x2)
            y0, y1p = min(y1, y2), max(y1, y2)
            draw.rectangle([x0, y0, x1p, y1p], outline=color)
    return np.array(pil)


def _add_anomaly(img: np.ndarray, seed: int = 1) -> np.ndarray:
    H, W = img.shape[0], img.shape[1]
    pil = Image.fromarray(img.copy())
    draw = ImageDraw.Draw(pil)
    rng = np.random.RandomState(seed)
    # draw a random rectangle with contrasting color
    w = rng.randint(W // 8, W // 3)
    h = rng.randint(H // 8, H // 3)
    x = rng.randint(0, W - w)
    y = rng.randint(0, H - h)
    color = tuple(int(c) for c in rng.randint(0, 255, size=3))
    draw.rectangle([x, y, x + w, y + h], fill=color)
    # draw some texture lines
    for _ in range(5):
        x1 = rng.randint(x, x + w)
        y1 = rng.randint(y, y + h)
        x2 = rng.randint(x, x + w)
        y2 = rng.randint(y, y + h)
        draw.line([x1, y1, x2, y2], fill=tuple(int(c) for c in rng.randint(0, 255, size=3)), width=2)
    return np.array(pil)


def generate_demo_dataset(num_good: int = 20, num_anom: int = 5, select_total: int = 10, image_size: Tuple[int, int] = (256, 256)) -> Dict:
    """Generate a simple demo dataset.

    Returns a dict with keys:
      - images: list of full-size numpy arrays
      - thumbs: list of thumbnail numpy arrays
      - labels: list of 'good' or 'anom'
      - selectable_indices: list of indices available for selection (mixed)
    """
    images: List[np.ndarray] = []
    labels: List[str] = []
    for i in range(num_good):
        images.append(_make_base_image(image_size, seed=100 + i))
        labels.append('good')

    for i in range(num_anom):
        img = _make_base_image(image_size, seed=200 + i)
        img = _add_anomaly(img, seed=300 + i)
        images.append(img)
        labels.append('anom')

    # create thumbnails
    thumbs: List[np.ndarray] = []
    for img in images:
        pil = Image.fromarray(img)
        t = pil.resize((64, 64), Image.BILINEAR)
        thumbs.append(np.array(t))

    # pick selectable indices (mix of good and anomalous)
    total = len(images)
    rng = np.random.RandomState(42)
    selectable = list(rng.choice(total, size=min(select_total, total), replace=False))

    return {
        'images': images,
        'thumbs': thumbs,
        'labels': labels,
        'selectable_indices': selectable,
    }


def get_image(dataset: Dict, index: int) -> np.ndarray:
    return dataset['images'][index]
