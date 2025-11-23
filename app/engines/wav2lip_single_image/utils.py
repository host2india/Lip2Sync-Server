# app/engines/wav2lip_single_image/utils.py
import os
import cv2
import numpy as np

def save_uploaded_bytes(data: bytes, path: str) -> str:
    """
    Save bytes to disk (works for image and audio bytes).
    Returns saved path.
    """
    with open(path, "wb") as f:
        f.write(data)
    return path

def generate_temp_video_from_image(image_path: str, out_video_path: str, fps: int = 60, duration_sec: int = 4):
    """
    Generate a temporary video from a single image.
    - fps: 60 (ultra-smooth)
    - duration_sec: default 4 seconds => fps*duration_sec frames
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found or cannot be read: {image_path}")

    height, width = img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_video_path, fourcc, float(fps), (width, height))
    frames = int(fps * duration_sec)

    for _ in range(frames):
        out.write(img)
    out.release()
    return out_video_path
