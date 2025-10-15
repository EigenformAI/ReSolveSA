import pandas as pd
import os
from tqdm import tqdm


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
        return None  # skip if filename doesn't match expected 4 parts

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
        # Skip malformed numeric values
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
        # Fill missing lithology info with None
        return {**row, **{col: None for col in litho_df.columns}}


def map_segments_to_lithology(segments_dir: str, litho_csv_path: str, output_csv_path: str, save_every=500):
    """Main mapping function with periodic saving and resume support."""
    print("🔹 Loading lithology CSV...")
    litho_df = safe_read_csv(litho_csv_path)
    litho_df = litho_df.rename(columns={
        "DRILLHOLE_NO": "drillhole_number",
        "DEPTH_FROM_M": "depth_from_m",
        "DEPTH_TO_M": "depth_to_m",
        "MAJOR_LITHOLOGY": "major_lithology"
    })
    litho_df["drillhole_number"] = litho_df["drillhole_number"].astype(str).str.strip()

    print("🔹 Scanning segment image filenames...")
    filenames = [f for f in os.listdir(segments_dir) if f.lower().endswith(".jpg")]

    # Resume logic: skip already processed files
    processed_files = set()
    if os.path.exists(output_csv_path):
        existing = pd.read_csv(output_csv_path)
        processed_files = set(existing["filename"].tolist())
        print(f"Resuming from previous run — {len(processed_files)} files already processed.")

    records = []
    total = len(filenames)

    with tqdm(total=total, desc="Processing segments") as pbar:
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

            # Save periodically to prevent data loss
            if len(records) >= save_every:
                df_partial = pd.DataFrame(records)
                df_partial.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)
                records = []

            pbar.update(1)

    # Save remaining records
    if records:
        df_partial = pd.DataFrame(records)
        df_partial.to_csv(output_csv_path, mode='a', header=not os.path.exists(output_csv_path), index=False)

    print(f"\n✅ Mapping complete. Saved all results to: {output_csv_path}")


if __name__ == "__main__":
    segments_dir = "split_segments"
    litho_csv_path = "sarig_dh_litho_exp.csv"
    output_csv_path = "segments_with_lithology.csv"

    map_segments_to_lithology(segments_dir, litho_csv_path, output_csv_path)