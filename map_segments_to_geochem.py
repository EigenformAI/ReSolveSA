import pandas as pd
from tqdm import tqdm
import os


def safe_read_csv(path):
    """Try multiple encodings for SARIG CSV compatibility."""
    for enc in ["utf-8", "cp1252", "iso-8859-1"]:
        try:
            return pd.read_csv(path, encoding=enc)
        except UnicodeDecodeError:
            continue
    raise ValueError("Could not read CSV with standard encodings")


def map_geochem_to_segments(segments_csv: str, geochem_csv: str, output_csv: str, save_every: int = 500):
    """Map geochemistry to segment center depths (depth_m) with periodic saving."""
    print("🔹 Loading data...")
    segments_df = pd.read_csv(segments_csv)
    geochem_df = safe_read_csv(geochem_csv)

    # Normalize geochem column names
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

    # Ensure numeric depth fields
    geochem_df["dh_depth_from"] = pd.to_numeric(geochem_df["dh_depth_from"], errors="coerce")
    geochem_df["dh_depth_to"] = pd.to_numeric(geochem_df["dh_depth_to"], errors="coerce")
    segments_df["depth_m"] = pd.to_numeric(segments_df["depth_m"], errors="coerce")

    # Normalize drillhole formats
    segments_df["drillhole_number"] = segments_df["drillhole_number"].astype(str).str.strip().str.upper()
    geochem_df["drillhole_number"] = geochem_df["drillhole_number"].astype(str).str.strip().str.upper()

    # Resume if output exists
    processed_rows = 0
    if os.path.exists(output_csv):
        print(f"🔸 Resuming from previous run: {output_csv}")
        existing = pd.read_csv(output_csv)
        processed_rows = len(existing)
        print(f"   → {processed_rows} rows already processed")

    print("🔹 Mapping geochemistry to segments based on depth_m...")
    results = []
    total = len(segments_df)
    tolerance = 0.05  # 5 cm depth tolerance

    # Normalize drillhole formats — remove trailing .0 from floats
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


    with tqdm(total=total, desc="Processing segments") as pbar:
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

            # Save intermediate results periodically
            if len(results) >= save_every:
                pd.DataFrame(results).to_csv(
                    output_csv, mode='a',
                    header=not os.path.exists(output_csv) or processed_rows == 0,
                    index=False
                )
                processed_rows += len(results)
                results = []

            pbar.update(1)

    # Save remaining results
    if results:
        pd.DataFrame(results).to_csv(
            output_csv, mode='a',
            header=not os.path.exists(output_csv) or processed_rows == 0,
            index=False
        )

    print(f"\n✅ Mapping complete. Saved to: {output_csv}")


if __name__ == "__main__":
    segments_csv = "segments_with_lithology.csv"
    geochem_csv = "sarig_rs_chem_exp.csv"
    output_csv = "segments_with_geochem.csv"

    # Adjust this number to control how often it saves
    map_geochem_to_segments(segments_csv, geochem_csv, output_csv, save_every=500)