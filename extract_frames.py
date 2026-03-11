#!/usr/bin/env python3
"""Extract frames from an MP4 video file into a folder."""

import argparse
import os
import cv2


def extract_frames(video_path: str, output_dir: str = None) -> int:
    """
    Extract all frames from a video file.

    Args:
        video_path: Path to the MP4 file
        output_dir: Output directory (default: video name + "_frames")

    Returns:
        Number of frames extracted
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    # Default output directory: video name without extension + "_frames"
    if output_dir is None:
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_dir = video_name + "_frames"

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_path = os.path.join(output_dir, f"frame_{frame_count:04d}.png")
        cv2.imwrite(frame_path, frame)
        frame_count += 1

    cap.release()

    # Print video info
    print(f"Video: {video_path}")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))} x {int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"Extracted {frame_count} frames to {output_dir}/")

    return frame_count


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from MP4 video")
    parser.add_argument("video", help="Path to MP4 video file")
    parser.add_argument("-o", "--output", help="Output directory (default: <video_name>_frames)")
    args = parser.parse_args()

    extract_frames(args.video, args.output)
