#!/usr/bin/env python3
"""
Generate multi-size test images for Hough Transform benchmarking.
"""

import os
from PIL import Image

SOURCE_IMAGE = "./star.png"
OUTPUT_DIR = "./"

if not os.path.exists(SOURCE_IMAGE):
    print(f"Error: Source image not found: {SOURCE_IMAGE}, now is downloading...")
    import requests

    url = "https://assets.science.nasa.gov/content/dam/science/missions/webb/science/2022/07/STScI-01GA6KKWG229B16K4Q38CH3BXS.png"              # 你要存的檔名

    r = requests.get(url)

    with open(SOURCE_IMAGE, "wb") as f:
        f.write(r.content)


TEST_SIZES = [
    (256, 144, "star_256.png"),
    (512, 288, "star_512.png"),
    (2048, 1152, "star_2k.png"),
    (4096, 2304, "star_4k.png"),
    (8192, 4608, "star_8k.png"),
]

def generate_test_images():
    """Generate all test images from the source image."""
    
    if not os.path.exists(SOURCE_IMAGE):
        print(f"Error: Source image not found: {SOURCE_IMAGE}")
        return False
    
    # Open the source image
    print("Loading source image...")
    source_img = Image.open(SOURCE_IMAGE)
    original_size = source_img.size
    print(f"Original size: {original_size[0]} x {original_size[1]}")
    print()
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Generate resized images
    print("Generating test images...")
    for width, height, filename in TEST_SIZES:
        output_path = os.path.join(OUTPUT_DIR, filename)
        
        print(f"  Creating {filename} ({width}x{height})...", end=" ", flush=True)
        resized_img = source_img.resize((width, height), Image.Resampling.LANCZOS)
        resized_img.save(output_path, "PNG")
        
        file_size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"Done ({file_size_mb:.2f} MB)")
    
    print()
    print("All test images saved successfully!")
    return True

if __name__ == "__main__":
    generate_test_images()