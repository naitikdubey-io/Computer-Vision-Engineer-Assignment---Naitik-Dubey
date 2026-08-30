"""
Step 1: Pull raw frames out of the source video with ffmpeg.

Usage:
    python src/extract_frames.py --video path/to/video.mp4 --fps 3
"""
import argparse
import subprocess
import sys
from pathlib import Path

from tool_paths import get_ffmpeg


def extract_frames(video_path: str, out_dir: str, fps: float) -> int:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    pattern = str(out / "frame_%06d.png")
    cmd = [
        get_ffmpeg(), "-y",
        "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        pattern,
    ]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(result.stderr, file=sys.stderr)
        raise RuntimeError("ffmpeg failed - is it installed and on PATH?")

    frames = sorted(out.glob("frame_*.png"))
    print(f"Extracted {len(frames)} frames to {out}")
    return len(frames)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to source video file")
    ap.add_argument("--out", default="data/frames", help="Output directory for frames")
    ap.add_argument("--fps", type=float, default=3.0,
                     help="Frames per second to sample (scoreboards don't need full framerate)")
    args = ap.parse_args()
    extract_frames(args.video, args.out, args.fps)
