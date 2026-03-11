#!/bin/bash
# Extract the last frame from each gen.mp4 in a directory tree.
# Usage: bash scripts/eval/extract_last_frame.sh <output_base>
# Saves last_frame.png next to each gen.mp4.

OUTPUT_BASE="$1"

if [ -z "$OUTPUT_BASE" ] || [ ! -d "$OUTPUT_BASE" ]; then
  echo "Usage: $0 <output_base>"
  exit 1
fi

python3 - "$OUTPUT_BASE" <<'PYEOF'
import sys
from pathlib import Path
from PIL import Image
import cv2

output_base = Path(sys.argv[1])

for video_path in sorted(output_base.rglob("gen.mp4")):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        print(f"Skipping {video_path} (no frames)")
        cap.release()
        continue

    cap.set(cv2.CAP_PROP_POS_FRAMES, total - 1)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        print(f"Skipping {video_path} (failed to read last frame)")
        continue

    out_path = video_path.parent / "last_frame.png"
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    Image.fromarray(frame_rgb).save(out_path)
    print(f"{video_path.parent.name}: frame {total-1} -> {out_path}")

print("Done.")
PYEOF
