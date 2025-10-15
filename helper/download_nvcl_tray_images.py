from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from pathlib import Path
import time

from nvcl_kit.reader import NVCLReader
from nvcl_kit.param_builder import param_builder
import pandas as pd

# Create directory for images
image_output_dir = Path("nvcl_matched_images")
image_output_dir.mkdir(exist_ok=True)

# Read the drillhole summary to get list of matched drillholes (now with nvcl_id)
summary_df = pd.read_csv('drillhole_mineral_summary.csv')
print(f"Total matched drillholes to fetch images for: {len(summary_df)}")
print(f"Summary columns: {summary_df.columns.tolist()}")

# Initialize counters
total_images_saved = 0
boreholes_with_images = 0
boreholes_without_images = 0

print(f"\nStarting image download...")
print(f"Images will be saved to: {image_output_dir}")
print(f"{'='*60}")

# Thread-safe counters
counter_lock = Lock()
total_images_saved = 0
boreholes_with_images = 0
boreholes_without_images = 0

param = param_builder('sa', max_boreholes=0)
reader = NVCLReader(param)

def download_borehole_images(row, idx, total):
    """Download images for a single borehole"""
    global total_images_saved, boreholes_with_images, boreholes_without_images
    
    drillhole_number = str(row['DRILLHOLE_NUMBER'])
    borehole_name = row['borehole_name']
    nvcl_id = str(row['nvcl_id'])
    
    result = {
        'drillhole_number': drillhole_number,
        'borehole_name': borehole_name,
        'nvcl_id': nvcl_id,
        'images_saved': 0,
        'status': 'pending'
    }
    
    print(f"[{idx+1}/{total}] Processing: {borehole_name} (DH: {drillhole_number}, NVCL ID: {nvcl_id})")
    
    try:
        # Get dataset list for this borehole using nvcl_id
        dataset_list = reader.get_dataset_list(nvcl_id)
        
        if not dataset_list or len(dataset_list) == 0:
            print(f"  ⚠️  No datasets found for {borehole_name}")
            result['status'] = 'no_datasets'
            with counter_lock:
                boreholes_without_images += 1
            return result
        
        ds = dataset_list[0]
        
        # Get tray image logs
        ilog_list = reader.get_tray_imglogs(ds.dataset_id)
        
        if not ilog_list or len(ilog_list) == 0:
            print(f"  ⚠️  No tray images found for {borehole_name}")
            result['status'] = 'no_images'
            with counter_lock:
                boreholes_without_images += 1
            return result
        
        print(f"  Found {len(ilog_list)} tray images for {borehole_name}")
        
        # Download each tray image
        borehole_images_saved = 0
        for img_idx, ilog in enumerate(ilog_list, 1):
            try:
                # Get image data
                img_data = reader.get_tray_thumb_jpg(ilog.log_id)
                
                if img_data:
                    # Create filename with drillhole_number and index
                    if len(ilog_list) == 1:
                        filename = f"{drillhole_number}.jpg"
                    else:
                        filename = f"{drillhole_number}_{img_idx}.jpg"
                    
                    filepath = image_output_dir / filename
                    
                    # Save image
                    with open(filepath, 'wb') as f:
                        f.write(img_data)
                    
                    borehole_images_saved += 1
                    
            except Exception as e:
                print(f"    ⚠️  Error downloading tray {img_idx} for {borehole_name}: {e}")
                continue
        
        # Update counters
        with counter_lock:
            total_images_saved += borehole_images_saved
            if borehole_images_saved > 0:
                boreholes_with_images += 1
            else:
                boreholes_without_images += 1
        
        result['images_saved'] = borehole_images_saved
        result['status'] = 'success' if borehole_images_saved > 0 else 'no_images_saved'
        print(f"  ✅ Saved {borehole_images_saved} images for {borehole_name}")
        
    except Exception as e:
        print(f"  ❌ Error processing {borehole_name}: {e}")
        result['status'] = f'error: {str(e)}'
        with counter_lock:
            boreholes_without_images += 1
    
    return result

# Process boreholes concurrently
print(f"\nDownloading images with {max(1, min(10, len(summary_df)))} concurrent workers...")
print(f"{'='*60}\n")

results = []
with ThreadPoolExecutor(max_workers=10) as executor:
    # Submit all tasks
    future_to_row = {
        executor.submit(download_borehole_images, row, idx, len(summary_df)): (idx, row)
        for idx, row in summary_df.iterrows()
    }
    
    # Process completed tasks
    for future in as_completed(future_to_row):
        try:
            result = future.result()
            results.append(result)
        except Exception as e:
            print(f"Task failed with exception: {e}")

print(f"\n{'='*60}")
print(f"Image download complete!")
print(f"✅ Total images saved: {total_images_saved}")
print(f"✅ Boreholes with images: {boreholes_with_images}")
print(f"⚠️  Boreholes without images: {boreholes_without_images}")
print(f"📁 Images saved to: {image_output_dir}")
print(f"{'='*60}")

# Show summary of results
results_df = pd.DataFrame(results)
print(f"\nDownload Summary by Status:")
print(results_df['status'].value_counts())