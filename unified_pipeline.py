"""
Unified ReSolveSA Core Processing Pipeline
Combines tray detection, core segmentation, lithology mapping, and geochemistry mapping

Usage:
    python unified_pipeline.py                    # Run all steps
    python unified_pipeline.py --skip-steps 1     # Skip step 1
    python unified_pipeline.py --skip-steps 1 2   # Skip steps 1 and 2
    python unified_pipeline.py --skip-steps 1,2,3 # Skip steps 1, 2, and 3
"""

import cv2
import numpy as np
import os
import json
import pandas as pd
import argparse
from pathlib import Path
from PIL import Image
from datetime import datetime
from tqdm import tqdm

# ==============================
# CONFIGURATION
# ==============================

# Step 1: Tray Detection
INPUT_DIR = "./nvcl_matched_images"
CORES_EXTRACTED_DIR = "./cores_extracted"
HEADER_EXCLUDE_FRAC = 0.10
FOOTER_EXCLUDE_FRAC = 0.10
SKIP_SPLIT_GAP = False
LEFT_EXCLUDE_FRAC = 0.06
RIGHT_EXCLUDE_FRAC = 0.02

# Step 2: Core Segmentation
SEGMENTS_DIR = "./split_segments"
DEPTH_FILE = "tray_depth_labels.jsonl"
SEGMENTS_PER_CORE = 10

# Step 3: Lithology Mapping
LITHO_CSV_PATH = "sarig_dh_litho_exp.csv"
SEGMENTS_WITH_LITHOLOGY_CSV = "segments_with_lithology.csv"

# Step 4: Geochemistry Mapping
GEOCHEM_CSV_PATH = "sarig_rs_chem_exp.csv"
FINAL_OUTPUT_CSV = "segments_with_geochem.csv"

# Processing parameters
SAVE_EVERY = 500  # Save intermediate results every N records


# ==============================
# STEP 1: TRAY DETECTION
# ==============================

def log_message(message: str):
    """Print a message with a timestamp."""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"[{timestamp}] {message}")


def has_middle_split_gap(fg):
    """Detect a large horizontal white band that splits upper/lower cores."""
    H, W = fg.shape
    row_ratio = (fg > 0).mean(axis=1)

    top = int(H * 0.15)
    bot = int(H * 0.85)
    mid = row_ratio[top:bot]

    low = mid < 0.05
    if not low.any():
        return False

    max_run = 0
    run = 0
    for v in low:
        run = run + 1 if v else 0
        max_run = max(max_run, run)

    gap_frac = max_run / (bot - top)
    has_upper = (row_ratio[:top] > 0.10).sum() > H * 0.02
    has_lower = (row_ratio[bot:] > 0.10).sum() > H * 0.02
    return gap_frac > 0.06 and has_upper and has_lower


def isolate_tray(image):
    """Crop the tray by masking 'white-ish' areas and taking the bbox of all foreground pixels."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_white = np.array([0, 0, 180], dtype=np.uint8)
    upper_white = np.array([180, 40, 255], dtype=np.uint8)
    white_mask = cv2.inRange(hsv, lower_white, upper_white)

    fg = cv2.bitwise_not(white_mask)

    H, W = fg.shape
    ht = int(H * HEADER_EXCLUDE_FRAC)
    hb = int(H * FOOTER_EXCLUDE_FRAC)
    wl = int(W * LEFT_EXCLUDE_FRAC)
    wr = int(W * RIGHT_EXCLUDE_FRAC)
    if ht > 0:
        fg[:ht, :] = 0
    if hb > 0:
        fg[H - hb:, :] = 0
    if wl > 0:
        fg[:, :wl] = 0
    if wr > 0:
        fg[:, W - wr:] = 0

    col_white_ratio = white_mask.mean(axis=0) / 255.0
    left_idx = 0
    while left_idx < W and col_white_ratio[left_idx] > 0.6:
        left_idx += 1
    if left_idx > 0:
        fg[:, :left_idx] = 0

    row_white_ratio = white_mask.mean(axis=1) / 255.0
    bottom_rows = 0
    i = H - 1
    while i >= 0 and row_white_ratio[i] > 0.6:
        bottom_rows += 1
        i -= 1
    if bottom_rows > 0:
        fg[H - bottom_rows:, :] = 0

    k_close = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 25))
    k_open = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k_close, iterations=2)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k_open, iterations=1)

    if SKIP_SPLIT_GAP and has_middle_split_gap(fg):
        return None

    ys, xs = np.where(fg > 0)
    if ys.size == 0:
        return image
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    x0, x1 = int(xs.min()), int(xs.max()) + 1

    pad = max(0, min(H, W) // 200)
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(W, x1 + pad)
    y1 = min(H, y1 + pad)
    return image[y0:y1, x0:x1]


def step1_tray_detection():
    """Step 1: Detect and extract trays from raw images."""
    log_message("=" * 60)
    log_message("STEP 1: TRAY DETECTION")
    log_message("=" * 60)

    os.makedirs(CORES_EXTRACTED_DIR, exist_ok=True)

    exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
    image_files = [p for p in Path(INPUT_DIR).iterdir() if p.suffix.lower() in exts]

    processed = 0
    skipped = 0

    for p in tqdm(image_files, desc="Processing tray images"):
        img = cv2.imread(str(p))
        if img is None:
            log_message(f"⚠️ Could not read {p.name}")
            skipped += 1
            continue

        crop = isolate_tray(img)
        if crop is None:
            log_message(f"⚠️ Skipped {p.name} (split gap detected)")
            skipped += 1
            continue

        output_path = Path(CORES_EXTRACTED_DIR, p.name)
        cv2.imwrite(str(output_path), crop)
        processed += 1

    log_message(f"✅ Tray detection complete: {processed} processed, {skipped} skipped")
    return processed


# ==============================
# STEP 2: CORE SEGMENTATION
# ==============================

def step2_core_segmentation():
    """Step 2: Split cores into segments with depth labels."""
    log_message("=" * 60)
    log_message("STEP 2: CORE SEGMENTATION")
    log_message("=" * 60)

    os.makedirs(SEGMENTS_DIR, exist_ok=True)

    # Load depth information
    if not os.path.exists(DEPTH_FILE):
        log_message(f"⚠️ Depth file not found: {DEPTH_FILE}")
        log_message("⚠️ Skipping core segmentation step")
        return 0

    depth_map = {}
    with open(DEPTH_FILE, "r") as f:
        for line in f:
            entry = json.loads(line)
            depth_map[os.path.basename(entry["image"])] = entry["depth_labels_m"]

    log_message(f"Loaded depth labels for {len(depth_map)} images")

    total_segments = 0
    image_files = [f for f in os.listdir(CORES_EXTRACTED_DIR)
                   if f.lower().endswith((".jpg", ".jpeg", ".png"))]

    for image_file in tqdm(image_files, desc="Segmenting cores"):
        if image_file not in depth_map:
            log_message(f"⚠️ Skipping {image_file}: no depth labels found")
            continue

        depths = depth_map[image_file]
        num_cores = len(depths)

        if num_cores == 0:
            log_message(f"⚠️ {image_file}: empty depth list")
            continue

        tray_img = Image.open(os.path.join(CORES_EXTRACTED_DIR, image_file))
        width, height = tray_img.size
        core_height = height // num_cores
        tray_id = Path(image_file).stem

        for i in range(num_cores):
            y1 = i * core_height
            y2 = (i + 1) * core_height if i < num_cores - 1 else height
            core_img = tray_img.crop((0, y1, width, y2))

            core_top = depths[i]
            if i < num_cores - 1:
                core_bottom = depths[i + 1]
            elif num_cores >= 2:
                core_bottom = core_top + (depths[-1] - depths[-2])
            else:
                core_bottom = core_top + 1.0

            depth_range = core_bottom - core_top
            segment_width = width // SEGMENTS_PER_CORE

            for j in range(SEGMENTS_PER_CORE):
                x1 = j * segment_width
                x2 = (j + 1) * segment_width if j < SEGMENTS_PER_CORE - 1 else width
                segment = core_img.crop((x1, 0, x2, core_height))

                depth_label = round(core_top + (j + 1) * (depth_range / SEGMENTS_PER_CORE), 2)
                out_name = f"{tray_id}_{i+1}_{j+1}_{depth_label}.jpg"
                out_path = os.path.join(SEGMENTS_DIR, out_name)
                segment.save(out_path, quality=95)
                total_segments += 1

    log_message(f"✅ Core segmentation complete: {total_segments} segments created")
    return total_segments


# ==============================
# STEP 3: LITHOLOGY MAPPING
# ==============================

def safe_read_csv(path):
    """Try multiple encodings for SARIG CSV compatibility."""
    for enc in ["utf-8", "cp1252", "iso-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read CSV with standard encodings")


def extract_metadata_from_filename(fname):
    """Extract metadata using simple string split assuming consistent filename format."""
    name = fname.replace(".jpg", "").strip()
    parts = name.split("_")

    if len(parts) != 4:
        return None

    try:
        drillhole_number = parts[0]
        core = int(parts[1])
        segment = int(parts[2])
        depth_m = float(parts[3])
        return {
            "filename": fname,
            "drillhole_number": drillhole_number,
            "core": core,
            "segment": segment,
            "depth_m": depth_m
        }
    except ValueError:
        return None


def match_lithology(row, litho_df):
    """Match one segment to lithology data based on depth overlap."""
    dh = str(row["drillhole_number"])
    depth = row["depth_m"]
    subset = litho_df[litho_df["drillhole_number"] == dh]
    match = subset[(subset["depth_from_m"] <= depth) & (subset["depth_to_m"] >= depth)]

    if not match.empty:
        return {**row, **match.iloc[0].to_dict()}
    else:
        return {**row, **{col: None for col in litho_df.columns}}


def step3_lithology_mapping():
    """Step 3: Map segments to lithology data."""
    log_message("=" * 60)
    log_message("STEP 3: LITHOLOGY MAPPING")
    log_message("=" * 60)

    if not os.path.exists(LITHO_CSV_PATH):
        log_message(f"⚠️ Lithology CSV not found: {LITHO_CSV_PATH}")
        log_message("⚠️ Skipping lithology mapping step")
        return 0

    log_message("Loading lithology CSV...")
    litho_df = safe_read_csv(LITHO_CSV_PATH)
    litho_df = litho_df.rename(columns={
        "DRILLHOLE_NO": "drillhole_number",
        "DEPTH_FROM_M": "depth_from_m",
        "DEPTH_TO_M": "depth_to_m",
        "MAJOR_LITHOLOGY": "major_lithology"
    })
    litho_df["drillhole_number"] = litho_df["drillhole_number"].astype(str).str.strip()

    log_message("Scanning segment image filenames...")
    filenames = [f for f in os.listdir(SEGMENTS_DIR) if f.lower().endswith(".jpg")]

    processed_files = set()
    if os.path.exists(SEGMENTS_WITH_LITHOLOGY_CSV):
        existing = pd.read_csv(SEGMENTS_WITH_LITHOLOGY_CSV)
        processed_files = set(existing["filename"].tolist())
        log_message(f"Resuming from previous run — {len(processed_files)} files already processed")

    records = []
    total = len(filenames)

    with tqdm(total=total, desc="Mapping lithology") as pbar:
        for i, fname in enumerate(filenames):
            if fname in processed_files:
                pbar.update(1)
                continue

            meta = extract_metadata_from_filename(fname)
            if meta is None:
                pbar.update(1)
                continue

            matched = match_lithology(meta, litho_df)
            records.append(matched)

            if len(records) >= SAVE_EVERY:
                df_partial = pd.DataFrame(records)
                df_partial.to_csv(
                    SEGMENTS_WITH_LITHOLOGY_CSV,
                    mode='a',
                    header=not os.path.exists(SEGMENTS_WITH_LITHOLOGY_CSV),
                    index=False
                )
                records = []

            pbar.update(1)

    if records:
        df_partial = pd.DataFrame(records)
        df_partial.to_csv(
            SEGMENTS_WITH_LITHOLOGY_CSV,
            mode='a',
            header=not os.path.exists(SEGMENTS_WITH_LITHOLOGY_CSV),
            index=False
        )

    log_message(f"✅ Lithology mapping complete: {SEGMENTS_WITH_LITHOLOGY_CSV}")
    return total


# ==============================
# STEP 4: GEOCHEMISTRY MAPPING
# ==============================

def step4_geochem_mapping():
    """Step 4: Map geochemistry data to segments."""
    log_message("=" * 60)
    log_message("STEP 4: GEOCHEMISTRY MAPPING")
    log_message("=" * 60)

    if not os.path.exists(SEGMENTS_WITH_LITHOLOGY_CSV):
        log_message(f"⚠️ Segments with lithology CSV not found: {SEGMENTS_WITH_LITHOLOGY_CSV}")
        log_message("⚠️ Skipping geochemistry mapping step")
        return 0

    if not os.path.exists(GEOCHEM_CSV_PATH):
        log_message(f"⚠️ Geochemistry CSV not found: {GEOCHEM_CSV_PATH}")
        log_message("⚠️ Skipping geochemistry mapping step")
        return 0

    log_message("Loading data...")
    segments_df = pd.read_csv(SEGMENTS_WITH_LITHOLOGY_CSV)
    geochem_df = safe_read_csv(GEOCHEM_CSV_PATH)

    geochem_df = geochem_df.rename(columns={
        "DRILLHOLE_NUMBER": "drillhole_number",
        "DH_DEPTH_FROM": "dh_depth_from",
        "DH_DEPTH_TO": "dh_depth_to",
        "CHEM_CODE": "chem_code",
        "VALUE": "value",
        "UNIT": "unit",
        "CHEM_METHOD_CODE": "chem_method_code",
        "CHEM_METHOD_DESC": "chem_method_desc"
    })

    geochem_df["dh_depth_from"] = pd.to_numeric(geochem_df["dh_depth_from"], errors="coerce")
    geochem_df["dh_depth_to"] = pd.to_numeric(geochem_df["dh_depth_to"], errors="coerce")
    segments_df["depth_m"] = pd.to_numeric(segments_df["depth_m"], errors="coerce")

    segments_df["drillhole_number"] = (
        segments_df["drillhole_number"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.upper()
    )
    geochem_df["drillhole_number"] = (
        geochem_df["drillhole_number"]
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
        .str.upper()
    )

    processed_rows = 0
    if os.path.exists(FINAL_OUTPUT_CSV):
        log_message(f"Resuming from previous run: {FINAL_OUTPUT_CSV}")
        existing = pd.read_csv(FINAL_OUTPUT_CSV)
        processed_rows = len(existing)
        log_message(f"   → {processed_rows} rows already processed")

    log_message("Mapping geochemistry to segments based on depth_m...")
    results = []
    total = len(segments_df)
    tolerance = 0.05

    with tqdm(total=total, desc="Mapping geochemistry") as pbar:
        for i, (_, seg) in enumerate(segments_df.iterrows()):
            if i < processed_rows:
                pbar.update(1)
                continue

            dh = seg["drillhole_number"]
            seg_depth = seg.get("depth_m", None)

            if pd.isna(dh) or pd.isna(seg_depth):
                pbar.update(1)
                continue

            subset = geochem_df[geochem_df["drillhole_number"] == dh]
            matches = subset[
                (subset["dh_depth_from"] - tolerance <= seg_depth) &
                (subset["dh_depth_to"] + tolerance >= seg_depth)
            ]

            if not matches.empty:
                for _, row in matches.iterrows():
                    result_row = {
                        **seg.to_dict(),
                        **row[["chem_code", "value", "unit", "chem_method_code", "chem_method_desc"]].to_dict()
                    }
                    results.append(result_row)
            else:
                results.append({
                    **seg.to_dict(),
                    "chem_code": None,
                    "value": None,
                    "unit": None,
                    "chem_method_code": None,
                    "chem_method_desc": None
                })

            if len(results) >= SAVE_EVERY:
                pd.DataFrame(results).to_csv(
                    FINAL_OUTPUT_CSV,
                    mode='a',
                    header=not os.path.exists(FINAL_OUTPUT_CSV) or processed_rows == 0,
                    index=False
                )
                processed_rows += len(results)
                results = []

            pbar.update(1)

    if results:
        pd.DataFrame(results).to_csv(
            FINAL_OUTPUT_CSV,
            mode='a',
            header=not os.path.exists(FINAL_OUTPUT_CSV) or processed_rows == 0,
            index=False
        )

    log_message(f"✅ Geochemistry mapping complete: {FINAL_OUTPUT_CSV}")
    return total


# ==============================
# MAIN PIPELINE
# ==============================

def run_pipeline(skip_steps=None):
    """
    Run the complete ReSolveSA processing pipeline.

    Args:
        skip_steps: List of step numbers to skip (e.g., [1, 2] to skip steps 1 and 2)
    """
    if skip_steps is None:
        skip_steps = []

    start_time = datetime.now()
    log_message("=" * 60)
    log_message("RESOLVES UNIFIED PROCESSING PIPELINE")
    log_message("=" * 60)
    log_message(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")

    results = {}

    try:
        if 1 not in skip_steps:
            results['step1'] = step1_tray_detection()
        else:
            log_message("Skipping Step 1: Tray Detection")

        if 2 not in skip_steps:
            results['step2'] = step2_core_segmentation()
        else:
            log_message("Skipping Step 2: Core Segmentation")

        if 3 not in skip_steps:
            results['step3'] = step3_lithology_mapping()
        else:
            log_message("Skipping Step 3: Lithology Mapping")

        if 4 not in skip_steps:
            results['step4'] = step4_geochem_mapping()
        else:
            log_message("Skipping Step 4: Geochemistry Mapping")

    except Exception as e:
        log_message(f"❌ Pipeline failed with error: {str(e)}")
        raise

    end_time = datetime.now()
    duration = end_time - start_time

    log_message("=" * 60)
    log_message("PIPELINE COMPLETE")
    log_message("=" * 60)
    log_message(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    log_message(f"Total duration: {duration}")
    log_message("\nResults summary:")
    for step, count in results.items():
        log_message(f"  {step}: {count} items processed")
    log_message("\nFinal output file: " + FINAL_OUTPUT_CSV)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="ReSolveSA Core Processing Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unified_pipeline.py                    # Run all steps
  python unified_pipeline.py --skip-steps 1     # Skip step 1 (tray detection)
  python unified_pipeline.py --skip-steps 1 2   # Skip steps 1 and 2
  python unified_pipeline.py --skip-steps 1,2,3 # Skip steps 1, 2, and 3

Steps:
  1 - Tray Detection
  2 - Core Segmentation
  3 - Lithology Mapping
  4 - Geochemistry Mapping
        """
    )
    parser.add_argument(
        '--skip-steps',
        nargs='*',
        type=str,
        default=[],
        help='Steps to skip (space or comma-separated). Example: 1 2 or 1,2,3'
    )

    args = parser.parse_args()

    # Parse skip_steps - handle both space-separated and comma-separated
    skip_steps = []
    for item in args.skip_steps:
        if ',' in item:
            skip_steps.extend([int(x.strip()) for x in item.split(',') if x.strip()])
        else:
            skip_steps.append(int(item))

    # Run the pipeline
    run_pipeline(skip_steps=skip_steps if skip_steps else None)
