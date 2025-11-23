# app/engines/wav2lip_single_image/utils.py
import os
import subprocess
from typing import Optional

def save_uploaded_bytes(data: bytes, path: str) -> str:
    """Write bytes to path and return path."""
    with open(path, "wb") as f:
        f.write(data)
    return path

def ensure_outputs_dir(path: str) -> str:
    """Ensure outputs directory exists and is writable."""
    os.makedirs(path, exist_ok=True)
    return path

def merge_audio_video_if_needed(video_in: str, audio_in: str, out_path: str) -> str:
    """
    Use ffmpeg to copy video and merge audio into out_path.
    Requires ffmpeg installed and available in PATH.
    """
    # Build ffmpeg command to map video and audio properly, re-encode audio to aac
    cmd = [
        "ffmpeg", "-y",
        "-i", video_in,
        "-i", audio_in,
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-map", "0:v:0",
        "-map", "1:a:0",
        out_path
    ]
    subprocess.run(cmd, check=True)
    return out_path

