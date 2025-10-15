import os
import json
from pathlib import Path
from PIL import Image

# === CONFIG ===
INPUT_DIR = "cores_extracted"          # Directory containing tray images
OUTPUT_DIR = "split_segments"      # Directory to save output segments
DEPTH_FILE = "tray_depth_labels.jsonl"  # JSONL with: {"image": "75573.jpg", "depth_labels_m": [160.3, 161.7, ...]}
SEGMENTS_PER_CORE = 10             # Number of equal parts per core

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Load JSONL depth information ---
depth_map = {}
with open(DEPTH_FILE, "r") as f:
    for line in f:
        entry = json.loads(line)
        depth_map[os.path.basename(entry["image"])] = entry["depth_labels_m"]

# --- Process tray images ---
for image_file in os.listdir(INPUT_DIR):
    if not image_file.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    if image_file not in depth_map:
        print(f"⚠️ Skipping {image_file}: no depth labels found.")
        continue

    depths = depth_map[image_file]
    num_cores = len(depths)

    # Skip invalid
    if num_cores == 0:
        print(f"⚠️ {image_file}: empty depth list.")
        continue

    # Load tray image
    tray_img = Image.open(os.path.join(INPUT_DIR, image_file))
    width, height = tray_img.size
    core_height = height // num_cores

    tray_id = Path(image_file).stem

    for i in range(num_cores):
        # Crop one horizontal row (core)
        y1 = i * core_height
        y2 = (i + 1) * core_height if i < num_cores - 1 else height
        core_img = tray_img.crop((0, y1, width, y2))

        # Define top/bottom depths safely
        core_top = depths[i]
        if i < num_cores - 1:
            core_bottom = depths[i + 1]
        elif num_cores >= 2:
            # Estimate last interval same as previous interval
            core_bottom = core_top + (depths[-1] - depths[-2])
        else:
            # Only one depth — make arbitrary 1 m range
            core_bottom = core_top + 1.0

        depth_range = core_bottom - core_top
        segment_width = width // SEGMENTS_PER_CORE

        # Split into 10 equal segments
        for j in range(SEGMENTS_PER_CORE):
            x1 = j * segment_width
            x2 = (j + 1) * segment_width if j < SEGMENTS_PER_CORE - 1 else width
            segment = core_img.crop((x1, 0, x2, core_height))

            # Compute midpoint depth
            depth_label = round(core_top + (j + 1) * (depth_range / SEGMENTS_PER_CORE), 2)

            # Filename: id_row_segment_depth.jpg
            out_name = f"{tray_id}_{i+1}_{j+1}_{depth_label}.jpg"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            segment.save(out_path, quality=95)

        print(f"✅ Tray {tray_id}: Core {i+1}/{num_cores} split into {SEGMENTS_PER_CORE} segments")

print("\n🎉 Done! All trays processed safely.")