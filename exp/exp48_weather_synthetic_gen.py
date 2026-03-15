# =============================================================================
# EXP48: Synthetic Weather Data Generator for SUES-200
# =============================================================================
# Copy this file to Kaggle notebook → Run All → Done
# Generates 10 weather-augmented versions of drone images using imgaug
# Compatible with WeatherPrompt (NeurIPS 2025) training pipeline
#
# Weather conditions: normal, fog, rain, snow, dark, light,
#                     fog_rain, fog_snow, rain_snow, wind
#
# Dataset: SUES-200 (drone-view/satellite-view structure)
# Output:  /kaggle/working/weather_synthetic/
# =============================================================================

import subprocess, sys
for _pkg in ["imgaug"]:
    try:
        __import__(_pkg)
    except ImportError:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", _pkg],
                       capture_output=True)

import os
import json
import time
import random
import platform
import warnings
import numpy as np

# --- NumPy 2.0 compatibility shim for imgaug ---
with warnings.catch_warnings():
    warnings.simplefilter("ignore", FutureWarning)
    if not hasattr(np, "sctypes"):
        np.sctypes = {
            "float": [np.float16, np.float32, np.float64],
            "int":   [np.int8, np.int16, np.int32, np.int64],
            "uint":  [np.uint8, np.uint16, np.uint32, np.uint64],
            "complex": [np.complex64, np.complex128],
        }
    for _attr, _val in [("bool", np.bool_), ("int", np.intp), ("float", np.float64),
                         ("complex", np.complex128), ("object", np.object_), ("str", np.str_)]:
        if not hasattr(np, _attr):
            setattr(np, _attr, _val)

from pathlib import Path
from datetime import datetime
from collections import OrderedDict
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm
import imgaug.augmenters as iaa


# =============================================================================
# CONFIG — Change these if your Kaggle dataset name is different
# =============================================================================
SUES_ROOT   = "/kaggle/input/datasets/chinguyeen/sues-dataset/SUES-200"
OUTPUT_ROOT = "/kaggle/working/weather_synthetic"
SEED        = 42
IMG_SIZE    = 384          # WeatherPrompt default; 0 = keep original
JPEG_QUALITY = 95
MAX_IMAGES_PER_CLASS = 1   # 1 = random pick per location (same as WeatherPrompt)

# Run mode:  "quick_test"  → 3 locations only (verify pipeline)
#            "full"        → all 200 locations, train+test
RUN_MODE = "full"

# SUES-200 structure on Kaggle:
#   {SUES_ROOT}/drone-view/{loc_id}/{altitude}/*.png
#   {SUES_ROOT}/satellite-view/{loc_id}/0.png
DRONE_DIR = "drone-view"
SAT_DIR   = "satellite-view"
ALTITUDES = ["150", "200", "250", "300"]
TRAIN_LOCS = list(range(1, 121))   # 120 training locations
TEST_LOCS  = list(range(121, 201)) #  80 test locations


# =============================================================================
# WEATHER AUGMENTATION DEFINITIONS (exact copy from WeatherPrompt/weather.py)
# =============================================================================
WEATHER_NAMES = [
    "normal",      # 0: No augmentation
    "fog",         # 1: Cloud/fog layer
    "rain",        # 2: Multiple rain layers
    "snow",        # 3: Multiple snowflake layers
    "dark",        # 4: Darkening + reduced brightness
    "light",       # 5: Overexposure / bright sunlight
    "fog_rain",    # 6: Fog + rain composite
    "fog_snow",    # 7: Fog + snow composite
    "rain_snow",   # 8: Rain + snow composite
    "wind",        # 9: Motion blur (simulates wind)
]


def build_weather_augmenters():
    """10 weather pipelines — EXACT params from WeatherPrompt."""
    return [
        # 0: normal
        None,
        # 1: fog
        iaa.Sequential([
            iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
                           intensity_coarse_scale=2, alpha_min=1.0,
                           alpha_multiplier=0.9, alpha_size_px_max=10,
                           alpha_freq_exponent=-2, sparsity=0.9,
                           density_multiplier=0.5, seed=35)
        ]),
        # 2: rain
        iaa.Sequential([
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=38),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2),  speed=(0.04, 0.06), seed=73),
            iaa.Rain(drop_size=(0.1, 0.2),  speed=(0.04, 0.06), seed=93),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=95),
        ]),
        # 3: snow
        iaa.Sequential([
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=38),
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=94),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=96),
        ]),
        # 4: dark
        iaa.Sequential([
            iaa.BlendAlpha(0.5, foreground=iaa.Add(100),
                           background=iaa.Multiply(0.2), seed=31),
            iaa.MultiplyAndAddToBrightness(mul=0.2, add=(-30, -15), seed=1991),
        ]),
        # 5: light
        iaa.Sequential([
            iaa.MultiplyAndAddToBrightness(mul=1.6, add=(0, 30), seed=1992),
        ]),
        # 6: fog_rain
        iaa.Sequential([
            iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
                           intensity_coarse_scale=2, alpha_min=1.0,
                           alpha_multiplier=0.9, alpha_size_px_max=10,
                           alpha_freq_exponent=-2, sparsity=0.9,
                           density_multiplier=0.5, seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=36),
        ]),
        # 7: fog_snow
        iaa.Sequential([
            iaa.CloudLayer(intensity_mean=225, intensity_freq_exponent=-2,
                           intensity_coarse_scale=2, alpha_min=1.0,
                           alpha_multiplier=0.9, alpha_size_px_max=10,
                           alpha_freq_exponent=-2, sparsity=0.9,
                           density_multiplier=0.5, seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=35),
            iaa.Snowflakes(flake_size=(0.5, 0.9), speed=(0.007, 0.03), seed=36),
        ]),
        # 8: rain_snow
        iaa.Sequential([
            iaa.Snowflakes(flake_size=(0.5, 0.8), speed=(0.007, 0.03), seed=35),
            iaa.Rain(drop_size=(0.05, 0.1), speed=(0.04, 0.06), seed=35),
            iaa.Rain(drop_size=(0.1, 0.2),  speed=(0.04, 0.06), seed=92),
            iaa.Rain(drop_size=(0.05, 0.2), speed=(0.04, 0.06), seed=91),
            iaa.Snowflakes(flake_size=(0.6, 0.9), speed=(0.007, 0.03), seed=74),
        ]),
        # 9: wind
        iaa.Sequential([
            iaa.MotionBlur(15, seed=17),
        ]),
    ]


# =============================================================================
# SUES-200 IMAGE DISCOVERY
# =============================================================================
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif"}


def discover_sues200(root, loc_ids, altitudes):
    """
    Discover drone images in SUES-200 structure:
      {root}/drone-view/{loc_id}/{altitude}/*.png

    Returns OrderedDict: {
        "0001_150": ["/path/to/0001/150/img1.png", ...],
        "0001_200": [...],
        ...
    }
    Keys are "{loc_id}_{altitude}" for unique identification.
    """
    drone_root = Path(root) / DRONE_DIR
    images = OrderedDict()

    for loc_num in loc_ids:
        loc_id = f"{loc_num:04d}"
        for alt in altitudes:
            alt_dir = drone_root / loc_id / alt
            if not alt_dir.exists():
                continue
            files = sorted([
                str(f) for f in alt_dir.iterdir()
                if f.is_file() and f.suffix.lower() in IMG_EXT
            ])
            if files:
                images[f"{loc_id}_{alt}"] = files

    return images


def discover_sues200_sat(root, loc_ids):
    """Discover satellite images: {root}/satellite-view/{loc_id}/0.png"""
    sat_root = Path(root) / SAT_DIR
    images = OrderedDict()
    for loc_num in loc_ids:
        loc_id = f"{loc_num:04d}"
        sat_file = sat_root / loc_id / "0.png"
        if sat_file.exists():
            images[loc_id] = [str(sat_file)]
    return images


# =============================================================================
# PREVIEW GRID
# =============================================================================
def save_preview_grid(augmenters, sample_path, output_dir):
    """Save a 2x5 visual grid showing all 10 weather conditions."""
    img = Image.open(sample_path).convert("RGB")
    if IMG_SIZE > 0:
        img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
    img_np = np.array(img)

    cell_w, cell_h = 384, 384
    cols, rows = 5, 2
    margin, label_h = 4, 28
    grid_w = cols * (cell_w + margin) + margin
    grid_h = rows * (cell_h + margin + label_h) + margin
    grid = Image.new("RGB", (grid_w, grid_h), (255, 255, 255))
    draw = ImageDraw.Draw(grid)

    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
    except Exception:
        font = ImageFont.load_default()

    for idx, (w_name, aug) in enumerate(zip(WEATHER_NAMES, augmenters)):
        row, col = divmod(idx, cols)
        x = margin + col * (cell_w + margin)
        y = margin + row * (cell_h + margin + label_h)
        out_np = img_np if aug is None else aug(image=img_np)
        cell_img = Image.fromarray(out_np).resize((cell_w, cell_h), Image.BICUBIC)
        grid.paste(cell_img, (x, y + label_h))
        draw.text((x + 4, y + 2), f"{idx}. {w_name}", fill=(0, 0, 0), font=font)

    preview_path = Path(output_dir) / "weather_samples_preview.png"
    grid.save(str(preview_path))
    print(f"  Preview saved: {preview_path}")


# =============================================================================
# WEATHER IMAGE GENERATION
# =============================================================================
def generate_weather(images_dict, augmenters, output_dir, split_label):
    """
    Apply 10 weather augmentations to discovered images.

    Args:
        images_dict: {"0001_150": [paths...], ...} from discover_sues200()
        augmenters:  list of 10 augmenters from build_weather_augmenters()
        output_dir:  base output path
        split_label: "train" or "test" (for subfolder naming)

    Output structure:
        {output_dir}/sues200_weather_{split_label}/{loc_id}_{alt}/
            {stem}-normal.jpg, {stem}-fog.jpg, ...
    """
    out_base = Path(output_dir) / f"sues200_weather_{split_label}"
    total = 0
    first_sample = None

    pbar = tqdm(sorted(images_dict.items()), desc=f"  {split_label}", unit="class")
    for key, img_paths in pbar:
        n_pick = min(MAX_IMAGES_PER_CLASS, len(img_paths))
        chosen = random.sample(img_paths, n_pick)

        class_dir = out_base / key
        class_dir.mkdir(parents=True, exist_ok=True)

        for src_path in chosen:
            if first_sample is None:
                first_sample = src_path

            img = Image.open(src_path).convert("RGB")
            if IMG_SIZE > 0:
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            img_np = np.array(img)
            stem = Path(src_path).stem

            for w_name, aug in zip(WEATHER_NAMES, augmenters):
                out_np = img_np if aug is None else aug(image=img_np)
                out_img = Image.fromarray(out_np)
                out_img.save(str(class_dir / f"{stem}-{w_name}.jpg"),
                             quality=JPEG_QUALITY)
                total += 1

        pbar.set_postfix(images=total)

    print(f"    {split_label}: {total} images across {len(images_dict)} classes")
    return total, first_sample


def generate_weather_sat(sat_dict, augmenters, output_dir, split_label):
    """Also augment satellite images for cross-view weather robustness."""
    out_base = Path(output_dir) / f"sues200_weather_{split_label}_sat"
    total = 0

    pbar = tqdm(sorted(sat_dict.items()), desc=f"  {split_label}_sat", unit="loc")
    for loc_id, img_paths in pbar:
        for src_path in img_paths:
            img = Image.open(src_path).convert("RGB")
            if IMG_SIZE > 0:
                img = img.resize((IMG_SIZE, IMG_SIZE), Image.BICUBIC)
            img_np = np.array(img)
            stem = Path(src_path).stem

            loc_dir = out_base / loc_id
            loc_dir.mkdir(parents=True, exist_ok=True)

            for w_name, aug in zip(WEATHER_NAMES, augmenters):
                out_np = img_np if aug is None else aug(image=img_np)
                Image.fromarray(out_np).save(
                    str(loc_dir / f"{stem}-{w_name}.jpg"), quality=JPEG_QUALITY)
                total += 1

        pbar.set_postfix(images=total)

    print(f"    {split_label}_sat: {total} images across {len(sat_dict)} locations")
    return total


# =============================================================================
# VERIFICATION
# =============================================================================
def verify_output(output_dir):
    """Print summary statistics of generated data."""
    print("\n" + "=" * 65)
    print("  OUTPUT VERIFICATION")
    print("=" * 65)

    root = Path(output_dir)
    if not root.exists():
        print("  No output directory found.")
        return

    grand_total = 0
    for split_dir in sorted(root.iterdir()):
        if not split_dir.is_dir() or not split_dir.name.startswith("sues200"):
            continue

        n_classes, n_images = 0, 0
        weather_counts = {w: 0 for w in WEATHER_NAMES}

        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            n_classes += 1
            for f in class_dir.iterdir():
                if f.suffix.lower() in IMG_EXT or f.suffix.lower() == ".jpg":
                    n_images += 1
                    w = f.stem.rsplit("-", 1)[-1] if "-" in f.stem else "unknown"
                    if w in weather_counts:
                        weather_counts[w] += 1

        grand_total += n_images
        print(f"\n  {split_dir.name}:")
        print(f"    Classes/Locs: {n_classes}")
        print(f"    Images:       {n_images}")
        print(f"    Per-weather:")
        for w, c in weather_counts.items():
            if c > 0:
                print(f"      {w:12s}: {c:6d}")

    print(f"\n  GRAND TOTAL: {grand_total} images")


# =============================================================================
# MAIN — Runs automatically when you hit "Run All" on Kaggle
# =============================================================================
def main():
    start = time.time()
    random.seed(SEED)
    np.random.seed(SEED)

    print("=" * 65)
    print("  EXP48: SUES-200 Synthetic Weather Data Generation")
    print(f"  Mode:   {RUN_MODE}")
    print(f"  Input:  {SUES_ROOT}")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"  Seed:   {SEED} | ImgSize: {IMG_SIZE} | Quality: {JPEG_QUALITY}")
    print("=" * 65)

    # ---- Validate dataset path ----
    drone_root = Path(SUES_ROOT) / DRONE_DIR
    sat_root = Path(SUES_ROOT) / SAT_DIR
    if not drone_root.exists():
        print(f"\n  ERROR: drone-view not found at {drone_root}")
        print(f"  Check SUES_ROOT. Your /kaggle/input/ contains:")
        input_dir = Path("/kaggle/input")
        if input_dir.exists():
            for p in sorted(input_dir.rglob("*"))[:30]:
                print(f"    {p}")
        return
    print(f"\n  drone-view found at {drone_root}")
    print(f"  satellite-view found: {sat_root.exists()}")

    # ---- Build augmenters ----
    print("\n[1/4] Building 10 weather augmentation pipelines...")
    augmenters = build_weather_augmenters()
    print(f"  {len(WEATHER_NAMES)} conditions: {', '.join(WEATHER_NAMES)}")

    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    # ---- Discover images ----
    print("\n[2/4] Discovering SUES-200 images...")

    if RUN_MODE == "quick_test":
        # Quick test: 3 train locations, 3 test locations
        test_train_locs = TRAIN_LOCS[:3]
        test_test_locs  = TEST_LOCS[:3]
        train_images = discover_sues200(SUES_ROOT, test_train_locs, ALTITUDES)
        test_images  = discover_sues200(SUES_ROOT, test_test_locs, ALTITUDES)
        train_sat    = discover_sues200_sat(SUES_ROOT, test_train_locs)
        test_sat     = discover_sues200_sat(SUES_ROOT, test_test_locs)
    else:
        train_images = discover_sues200(SUES_ROOT, TRAIN_LOCS, ALTITUDES)
        test_images  = discover_sues200(SUES_ROOT, TEST_LOCS, ALTITUDES)
        train_sat    = discover_sues200_sat(SUES_ROOT, TRAIN_LOCS)
        test_sat     = discover_sues200_sat(SUES_ROOT, TEST_LOCS)

    print(f"  Train drone: {len(train_images)} class-altitude combos")
    print(f"  Test  drone: {len(test_images)} class-altitude combos")
    print(f"  Train sat:   {len(train_sat)} locations")
    print(f"  Test  sat:   {len(test_sat)} locations")

    if not train_images and not test_images:
        print("\n  ERROR: No images found! Check SUES_ROOT path.")
        print(f"  Expected: {SUES_ROOT}/drone-view/0001/150/*.png")
        # Debug: show what's actually there
        dr = Path(SUES_ROOT) / DRONE_DIR
        if dr.exists():
            children = sorted(list(dr.iterdir()))[:10]
            print(f"  Found in {dr}: {[c.name for c in children]}")
            if children:
                sub = sorted(list(children[0].iterdir()))[:5]
                print(f"  Found in {children[0].name}/: {[s.name for s in sub]}")
        return

    # ---- Generate weather images ----
    print("\n[3/4] Generating weather-augmented images...")
    total = 0
    first_sample = None

    if train_images:
        n, sample = generate_weather(train_images, augmenters, OUTPUT_ROOT, "train")
        total += n
        if first_sample is None:
            first_sample = sample

    if test_images:
        n, sample = generate_weather(test_images, augmenters, OUTPUT_ROOT, "test")
        total += n
        if first_sample is None:
            first_sample = sample

    if train_sat:
        total += generate_weather_sat(train_sat, augmenters, OUTPUT_ROOT, "train")

    if test_sat:
        total += generate_weather_sat(test_sat, augmenters, OUTPUT_ROOT, "test")

    print(f"\n  Total generated: {total} images")

    # ---- Preview grid ----
    print("\n[4/4] Saving preview grid...")
    if first_sample:
        save_preview_grid(augmenters, first_sample, OUTPUT_ROOT)

    # ---- Save generation log ----
    log = {
        "dataset": "sues200",
        "sues_root": SUES_ROOT,
        "output_root": OUTPUT_ROOT,
        "run_mode": RUN_MODE,
        "seed": SEED,
        "img_size": IMG_SIZE,
        "max_images_per_class": MAX_IMAGES_PER_CLASS,
        "weather_conditions": WEATHER_NAMES,
        "total_generated": total,
        "timestamp": datetime.now().isoformat(),
        "platform": platform.platform(),
        "python": sys.version,
    }
    log_path = Path(OUTPUT_ROOT) / "generation_log.json"
    with open(str(log_path), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2, ensure_ascii=False)
    print(f"  Log saved: {log_path}")

    # ---- Verify ----
    verify_output(OUTPUT_ROOT)

    elapsed = time.time() - start
    print(f"\n{'=' * 65}")
    print(f"  Done! ({elapsed:.1f}s)")
    print(f"{'=' * 65}")


main()
