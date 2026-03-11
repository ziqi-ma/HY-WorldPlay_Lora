#!/bin/bash
# Crop images in a directory to 16:9 and 9:16 aspect ratios (center crop).
# Usage: bash scripts/eval/crop_images.sh <input_dir>

INPUT_DIR="$1"

if [ -z "$INPUT_DIR" ] || [ ! -d "$INPUT_DIR" ]; then
  echo "Usage: $0 <input_dir>"
  exit 1
fi

python3 - "$INPUT_DIR" <<'PYEOF'
import sys, os
from pathlib import Path
from PIL import Image

input_dir = Path(sys.argv[1])
out_16x9 = input_dir / "16x9"
out_9x16 = input_dir / "9x16"
out_16x9.mkdir(exist_ok=True)
out_9x16.mkdir(exist_ok=True)

EXTS = {".png", ".jpg", ".jpeg", ".webp"}

for img_path in sorted(input_dir.iterdir()):
    if img_path.is_dir() or img_path.suffix.lower() not in EXTS:
        continue

    print(f"Processing: {img_path.name}")
    img = Image.open(img_path)
    w, h = img.size

    # --- 16:9 center crop ---
    if w * 9 > h * 16:
        new_w, new_h = h * 16 // 9, h
    else:
        new_w, new_h = w, w * 9 // 16
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    img.crop((left, top, left + new_w, top + new_h)).save(out_16x9 / img_path.name)
    print(f"  -> 16:9: {out_16x9 / img_path.name}")

    # --- 9:16 center crop ---
    if w * 16 > h * 9:
        new_w, new_h = h * 9 // 16, h
    else:
        new_w, new_h = w, w * 16 // 9
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    img.crop((left, top, left + new_w, top + new_h)).save(out_9x16 / img_path.name)
    print(f"  -> 9:16: {out_9x16 / img_path.name}")

print(f"Done. Outputs in {out_16x9}/ and {out_9x16}/")
PYEOF
