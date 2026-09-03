# ReSolveSA Core Processing Pipeline

A comprehensive pipeline for processing drill core tray images from NVCL data, extracting segments, and mapping them to lithology and geochemistry data from SARIG.

The dataset this pipeline produces is published at [**huggingface.co/datasets/EigenformAI/ReSolveSA**](https://huggingface.co/datasets/EigenformAI/ReSolveSA) (17,992 core-segment rows linked to SARIG lithology and geochemistry).

## Overview

This pipeline processes drill core tray images through four sequential steps:
1. **Tray Detection** - Isolates core trays from raw images
2. **Core Segmentation** - Splits cores into 10 equal segments with depth labels
3. **Lithology Mapping** - Maps segments to lithology data based on depth
4. **Geochemistry Mapping** - Maps geochemistry data to segments

## Prerequisites

Before running the main pipeline, you need to:
1. Download NVCL tray images
2. Extract depth labels using OCR
3. Have SARIG lithology and geochemistry CSV files ready

### Required Files

- `drillhole_mineral_summary.csv` - List of matched drillholes with NVCL IDs
- `sarig_dh_litho_exp.csv` - SARIG lithology data (~300 MB, download from [SARIG](https://pid.sarig.sa.gov.au/dataset/mesac1017))
- `sarig_rs_chem_exp.csv` - SARIG geochemistry data (~21 GB, download from [SARIG](https://pid.sarig.sa.gov.au/dataset/mesac1017))

> **⚠️ Important - Large File Handling:**
> The SARIG CSV files are very large, especially `sarig_rs_chem_exp.csv` (21 GB). Loading these files will require significant RAM (32+ GB recommended).
>
> **To reduce memory usage**, consider pre-filtering these CSV files to include only the drillholes you need:
> ```bash
> # Example: Filter geochemistry data for specific drillholes
> # Create a filtered version with only your drillholes from drillhole_mineral_summary.csv
> python -c "
> import pandas as pd
> summary = pd.read_csv('drillhole_mineral_summary.csv')
> dh_list = summary['DRILLHOLE_NUMBER'].astype(str).str.strip().str.upper().unique()
>
> # Process in chunks to avoid loading entire file
> chunks = []
> for chunk in pd.read_csv('sarig_rs_chem_exp.csv', chunksize=100000, encoding='cp1252'):
>     chunk['DRILLHOLE_NUMBER'] = chunk['DRILLHOLE_NUMBER'].astype(str).str.strip().str.upper()
>     filtered = chunk[chunk['DRILLHOLE_NUMBER'].isin(dh_list)]
>     chunks.append(filtered)
>
> result = pd.concat(chunks, ignore_index=True)
> result.to_csv('sarig_rs_chem_exp_filtered.csv', index=False)
> print(f'Filtered from {len(chunk)} to {len(result)} rows')
> "
> ```
> Then update the pipeline configuration to use the filtered files.

### Python Environment Setup

It's recommended to use a virtual environment to manage dependencies:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Linux/Mac:
source venv/bin/activate
# On Windows:
# venv\Scripts\activate

# Install required packages
pip install -r requirements.txt
```

**Required packages** (from `requirements.txt`):
- `opencv-python` - Image processing for tray detection
- `numpy` - Numerical operations
- `pandas` - Data manipulation and CSV handling
- `matplotlib` - Visualization (optional)
- `Pillow` - Image handling for segmentation
- `tqdm` - Progress bars
- `openai` - OpenAI API for OCR (depth label extraction)
- `nvcl-kit` - NVCL data access for downloading tray images

## Step-by-Step Instructions

### Step 0: Preparation (One-time setup)

#### 0.1 Download NVCL Tray Images

This script downloads drill core tray images from NVCL for all drillholes listed in `drillhole_mineral_summary.csv`:

**Before running:**
Check your CPU cores to optimize concurrent workers:
```bash
# Check available CPU cores
# Linux/Mac:
nproc
# or
lscpu | grep "^CPU(s):"

# Windows (PowerShell):
# $env:NUMBER_OF_PROCESSORS
```

Then adjust `max_workers` in `helper/download_nvcl_tray_images.py` line 131:
```python
# Adjust based on your CPU cores (recommended: 0.5x to 1x your CPU count)
with ThreadPoolExecutor(max_workers=10) as executor:  # Change 10 to your preferred value
```

**Run the script:**
```bash
python helper/download_nvcl_tray_images.py
```

**What it does:**
- Reads drillhole IDs from `drillhole_mineral_summary.csv`
- Downloads tray images using the `nvcl_kit` library
- Saves images to `nvcl_matched_images/` directory
- Uses concurrent workers for faster downloads (default: 10 workers)
- Progress tracked with detailed logging

**Performance tips:**
- **Local machine:** Use 4-8 workers to avoid overwhelming your system
- **Remote server/cloud:** Can use 10-20 workers depending on CPU cores and network bandwidth
- **Network-limited:** If you have slow internet, reduce workers to 2-4 to prevent timeouts

**Output:**
- Images saved to `nvcl_matched_images/` with filenames: `{drillhole_number}.jpg` or `{drillhole_number}_{index}.jpg`

**Expected time:** Varies based on number of drillholes, network speed, and workers count

---

#### 0.2 Extract Depth Labels (OCR)

This script uses OpenAI's GPT-4o vision model to extract depth labels from tray images:

```bash
python helper/openai_ocr.py
```

> **Note:** OpenAI OCR is one approach to extract depth labels. You can use any preferred OCR service (Google Cloud Vision, Azure Computer Vision, Claude API, Tesseract, etc.) as long as the output is in the required JSONL format (see below).

**Before running:**
1. Set your OpenAI API key in `helper/openai_ocr.py`:
   ```python
   client = OpenAI(api_key="your-api-key-here")
   ```

**What it does:**
- Processes all images in `nvcl_matched_images/`
- Uses GPT-4o to read depth labels from each tray image
- Extracts numeric depth values in ascending order
- Saves results to `tray_depth_labels.jsonl`
- Uses 5 concurrent workers (adjustable based on rate limits)

**Configuration options:**
- `MAX_WORKERS` - Number of parallel API calls (default: 5)
- `MODEL` - OpenAI model to use (default: "gpt-4o")
- `RETRY_LIMIT` - Number of retries per image (default: 3)

**Output:**
- `tray_depth_labels.jsonl` - JSONL file with format:
  ```json
  {"image": "75573.jpg", "depth_labels_m": [160.3, 161.7, 163.1, ...]}
  ```

**Expected time:** Depends on number of images and API rate limits

---

### Step 1: Run the Main Pipeline

Once prerequisites are complete, run the unified pipeline:

```bash
# Run all steps
python unified_pipeline.py

# Skip specific steps (if already completed)
python unified_pipeline.py --skip-steps 1     # Skip step 1 (tray detection)
python unified_pipeline.py --skip-steps 1 2   # Skip steps 1 and 2
python unified_pipeline.py --skip-steps 1,2,3 # Skip steps 1, 2, and 3

# Show help
python unified_pipeline.py --help
```

**What it does:**

1. **Tray Detection**
   - Reads images from `nvcl_matched_images/`
   - Removes headers, footers, rulers, and scale markers
   - Isolates the actual core tray region
   - Outputs cleaned images to `cores_extracted/`

2. **Core Segmentation**
   - Reads cleaned trays from `cores_extracted/`
   - Uses depth labels from `tray_depth_labels.jsonl`
   - Splits each horizontal core row into 10 equal segments
   - Saves segments with depth-labeled filenames to `split_segments/`
   - Filename format: `{tray_id}_{core_row}_{segment}_{depth}.jpg`

3. **Lithology Mapping**
   - Reads segment filenames from `split_segments/`
   - Matches each segment to lithology data from `sarig_dh_litho_exp.csv` based on depth
   - Outputs `segments_with_lithology.csv`

4. **Geochemistry Mapping**
   - Reads `segments_with_lithology.csv`
   - Matches segments to geochemistry data from `sarig_rs_chem_exp.csv` based on depth
   - Outputs final dataset: `segments_with_geochem.csv`

**Output directories:**
- `cores_extracted/` - Cleaned tray images
- `split_segments/` - Individual core segments
- `segments_with_lithology.csv` - Segments with lithology data
- `segments_with_geochem.csv` - **Final output** with all data

---

## Pipeline Configuration

All configuration parameters are in `unified_pipeline.py` (lines 17-36):

```python
# Step 1: Tray Detection
INPUT_DIR = "./nvcl_matched_images"
CORES_EXTRACTED_DIR = "./cores_extracted"
HEADER_EXCLUDE_FRAC = 0.10  # Adjust header exclusion
FOOTER_EXCLUDE_FRAC = 0.10  # Adjust footer exclusion
SKIP_SPLIT_GAP = False      # Skip trays with large gaps

# Step 2: Core Segmentation
SEGMENTS_DIR = "./split_segments"
DEPTH_FILE = "tray_depth_labels.jsonl"
SEGMENTS_PER_CORE = 10      # Number of segments per core row

# Step 3 & 4: Data Mapping
LITHO_CSV_PATH = "sarig_dh_litho_exp.csv"
GEOCHEM_CSV_PATH = "sarig_rs_chem_exp.csv"
SAVE_EVERY = 500            # Save intermediate results every N records
```

---

## Advanced Usage

### Skip Specific Pipeline Steps

If you've already run certain steps and want to resume, use the `--skip-steps` command-line argument:

```bash
# Skip tray detection (if cores_extracted/ already exists)
python unified_pipeline.py --skip-steps 1

# Skip tray detection and segmentation (if split_segments/ already exists)
python unified_pipeline.py --skip-steps 1 2

# Skip first three steps (if segments_with_lithology.csv already exists)
python unified_pipeline.py --skip-steps 1,2,3

# Alternative comma-separated format
python unified_pipeline.py --skip-steps 1,2,3
```

**Step numbers:**
- `1` = Tray Detection
- `2` = Core Segmentation
- `3` = Lithology Mapping
- `4` = Geochemistry Mapping

**Or use directly in Python:**
```python
from unified_pipeline import run_pipeline
run_pipeline(skip_steps=[1, 2])  # Skip steps 1 and 2
```

### Resume After Interruption

Steps 3 and 4 (lithology and geochemistry mapping) support automatic resume:
- If output CSV files exist, the pipeline will skip already processed records
- Safe to interrupt and restart without losing progress

---

## Output Files

### Final Output: `segments_with_geochem.csv`

Contains all processed segments with:
- **Segment metadata:** filename, drillhole_number, core, segment, depth_m
- **Lithology data:** major_lithology, depth_from_m, depth_to_m, etc.
- **Geochemistry data:** chem_code, value, unit, chem_method_code, chem_method_desc

This file is published as a dataset at [huggingface.co/datasets/EigenformAI/ReSolveSA](https://huggingface.co/datasets/EigenformAI/ReSolveSA).

### Intermediate Files

- `cores_extracted/` - Cleaned tray images (Step 1 output)
- `split_segments/` - Individual segment images (Step 2 output)
- `segments_with_lithology.csv` - Segments with lithology only (Step 3 output)

---

## Troubleshooting

### Missing depth labels
- Ensure `tray_depth_labels.jsonl` exists and contains depth data for your images
- Check OCR results for accuracy

### No lithology/geochemistry matches
- Verify drillhole numbers match between datasets
- Check depth ranges overlap with your segments
- Review CSV file encodings (pipeline tries utf-8, cp1252, iso-8859-1)

### Memory issues
- **Large CSV files:** The SARIG geochemistry file (`sarig_rs_chem_exp.csv`) is ~21 GB and will require 32+ GB RAM to load
- **Solution:** Pre-filter the CSV files to include only your drillholes (see "Large File Handling" section above)
- Reduce `SAVE_EVERY` parameter for more frequent saves
- Process fewer images at once
- Consider using a machine with more RAM or a server environment

### API rate limits (OCR step)
- Reduce `MAX_WORKERS` in `helper/openai_ocr.py`
- Add delay between requests if needed

---

## Complete Workflow Summary

```bash
# 1. Download tray images (run once)
python helper/download_nvcl_tray_images.py

# 2. Extract depth labels with OCR (run once)
python helper/openai_ocr.py

# 3. Run the complete processing pipeline
python unified_pipeline.py

# Or skip already completed steps:
# python unified_pipeline.py --skip-steps 1 2

# Final output: segments_with_geochem.csv
```

**Quick reference for skipping steps:**
```bash
# Already have cleaned trays? Skip step 1
python unified_pipeline.py --skip-steps 1

# Already have segments? Skip steps 1-2
python unified_pipeline.py --skip-steps 1 2

# Only need geochemistry mapping? Skip steps 1-3
python unified_pipeline.py --skip-steps 1 2 3
```

---

## Project Structure

```
ReSolveSA/
├── helper/
│   ├── download_nvcl_tray_images.py  # Downloads NVCL images
│   └── openai_ocr.py                 # Extracts depth labels
├── unified_pipeline.py               # Main processing pipeline
├── tray_detection.py                 # (Legacy - now in unified_pipeline)
├── cores_into_segments.py            # (Legacy - now in unified_pipeline)
├── map_segments_to_lithology.py      # (Legacy - now in unified_pipeline)
├── map_segments_to_geochem.py        # (Legacy - now in unified_pipeline)
├── requirements.txt
└── README.md
```

---

## License

[MIT](LICENSE).

## Contact

Open an issue on [GitHub](https://github.com/EigenformAI/ReSolveSA/issues).