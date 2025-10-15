import os
import json
import base64
import time
import concurrent.futures
from openai import OpenAI
from tqdm import tqdm

# -----------------------------
# Configuration
# -----------------------------
IMAGE_DIR = "nvcl_matched_images"        # folder containing tray images
OUTPUT_JSONL = "tray_depth_labels.jsonl"
MODEL = "gpt-4o"           # or "gpt-4o" for better OCR accuracy
MAX_WORKERS = 5                # number of parallel threads (adjust to your rate limit)
RETRY_LIMIT = 3

client = OpenAI(api_key="")

# -----------------------------
# Function: extract depths from one image
# -----------------------------
def extract_depth_labels(image_path: str):
    """Send one image to OpenAI API and extract depth labels."""
    with open(image_path, "rb") as img_file:
        image_bytes = img_file.read()
    b64_image = base64.b64encode(image_bytes).decode("utf-8")

    for attempt in range(RETRY_LIMIT):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a geology assistant that reads depth labels from "
                            "NVCL drill core tray images. Return only numeric depth values "
                            "in meters, in ascending order, as a JSON array of floats."
                        ),
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "Extract all visible depth labels in meters."},
                            {
                                "type": "image_url",
                                "image_url": {"url": f"data:image/jpeg;base64,{b64_image}"}
                            },
                        ],
                    },
                ],
                temperature=0,
            )

            text = response.choices[0].message.content.strip()

            # Try parsing as JSON first
            try:
                depths = json.loads(text)
                if isinstance(depths, list):
                    return depths
            except json.JSONDecodeError:
                pass

            # Fallback text cleanup
            cleaned = (
                text.replace("m", "")
                .replace("[", "")
                .replace("]", "")
                .replace(",", " ")
            )
            depths = []
            for token in cleaned.split():
                try:
                    depths.append(float(token))
                except ValueError:
                    continue
            return depths

        except Exception as e:
            print(f"⚠️ Error ({attempt+1}/{RETRY_LIMIT}) for {os.path.basename(image_path)}: {e}")
            time.sleep(2)

    return []


# -----------------------------
# Worker: process one image file
# -----------------------------
def process_image(img_name: str):
    path = os.path.join(IMAGE_DIR, img_name)
    depths = extract_depth_labels(path)
    return {"image": img_name, "depth_labels_m": depths}


# -----------------------------
# Main function
# -----------------------------
def main():
    image_files = [
        f for f in os.listdir(IMAGE_DIR)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]

    print(f"🪨 Found {len(image_files)} images in '{IMAGE_DIR}'")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for result in tqdm(executor.map(process_image, image_files), total=len(image_files)):
            results.append(result)

    # Save to JSONL
    with open(OUTPUT_JSONL, "w") as f:
        for row in results:
            f.write(json.dumps(row) + "\n")

    print(f"\n✅ Done! Processed {len(results)} images. Results saved to {OUTPUT_JSONL}")

main()