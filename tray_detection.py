import cv2
import numpy as np
import os
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# -----------------------------
# Configuration
# -----------------------------
INPUT_DIR = "./nvcl_matched_images"  # folder with original HyLogger tray images
OUTPUT_DIR = "./cores_extracted"     # folder where split cores will go
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Add near the top (config)
HEADER_EXCLUDE_FRAC = 0.10   # 0.08–0.15
FOOTER_EXCLUDE_FRAC = 0.10   # distance scale text at bottom
SKIP_SPLIT_GAP      = False  # set True to skip trays with a big empty row
LEFT_EXCLUDE_FRAC   = 0.06   # trim left-side rulers/scales
RIGHT_EXCLUDE_FRAC  = 0.02   # occasional right labels

# -----------------------------
# Helper functions
# -----------------------------
def isolate_tray(image):
    """Crop the tray by masking 'white-ish' areas and taking the bbox of all foreground pixels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # White-ish margins/labels
    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 40, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    # Non-white foreground (tray + rocks)
    fg = cv2.bitwise_not(white_mask)

    # Exclude title/top, bottom scale, and side rulers (fractional + adaptive by white density)
    H, W = fg.shape
    ht = int(H * HEADER_EXCLUDE_FRAC)
    hb = int(H * FOOTER_EXCLUDE_FRAC)
    wl = int(W * LEFT_EXCLUDE_FRAC)
    wr = int(W * RIGHT_EXCLUDE_FRAC)
    if ht > 0: fg[:ht, :] = 0
    if hb > 0: fg[H - hb :, :] = 0
    if wl > 0: fg[:, :wl] = 0
    if wr > 0: fg[:, W - wr :] = 0

    # Adaptive: grow left/bottom exclusion where the area is predominantly white (scales)
    # Left: count leading columns with high white ratio
    col_white_ratio = white_mask.mean(axis=0) / 255.0
    left_idx = 0
    while left_idx < W and col_white_ratio[left_idx] > 0.6:
        left_idx += 1
    if left_idx > 0:
        fg[:, :left_idx] = 0

    # Bottom: count trailing rows with high white ratio
    row_white_ratio = white_mask.mean(axis=1) / 255.0
    bottom_rows = 0
    i = H - 1
    while i >= 0 and row_white_ratio[i] > 0.6:
        bottom_rows += 1
        i -= 1
    if bottom_rows > 0:
        fg[H - bottom_rows :, :] = 0

    # Morphology to de-noise; we don't try to bridge a large gap intentionally
    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    k_open  = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN,  k_open,  iterations=1)

    # Optionally skip files with a strong middle white band
    if SKIP_SPLIT_GAP and has_middle_split_gap(fg):
        return None  # signal to skip

    # Robust bbox over all foreground pixels (handles internal white rows)
    ys, xs = np.where(fg > 0)
    if ys.size == 0:
        return image  # fallback
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    # Small padding
    pad = max(0, min(H, W) // 200)
    x0 = max(0, x0 - pad); y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad); y1 = min(H, y1 + pad)
    return image[y0:y1, x0:x1]

def has_middle_split_gap(fg):
    """Detect a large horizontal white band that splits upper/lower cores."""
    H, W = fg.shape
    row_ratio = (fg > 0).mean(axis=1)  # 0..1 foreground ratio per row

    # Look only in the central band to avoid headers/footers
    top = int(H * 0.15); bot = int(H * 0.85)
    mid = row_ratio[top:bot]

    # A 'gap' is a run of very low foreground density
    low = mid < 0.05
    if not low.any():
        return False

    # longest consecutive low run length
    max_run = 0; run = 0
    for v in low:
        run = run + 1 if v else 0
        max_run = max(max_run, run)

    # Consider it a split if the gap is tall enough AND both sides have content
    gap_frac = max_run / (bot - top)
    has_upper = (row_ratio[:top] > 0.10).sum() > H * 0.02
    has_lower = (row_ratio[bot:] > 0.10).sum() > H * 0.02
    return gap_frac > 0.06 and has_upper and has_lower  # tweak 0.04–0.10 as needed

def log_message(message: str):
    """Print a message with a timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")

def process_folder():
    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    for p in Path(INPUT_DIR).iterdir():
        log_message(f"Processing {p}")
        if p.suffix.lower() not in exts:
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        crop = isolate_tray(img)
        if crop is None:  # skipped due to split gap
            continue
        cv2.imwrite(str(Path(OUTPUT_DIR, p.name)), crop)

if __name__ == "__main__":
    process_folder()
