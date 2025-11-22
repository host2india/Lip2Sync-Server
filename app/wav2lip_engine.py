import os
import torch
import subprocess
from pathlib import Path

MODELS_DIR = "models/wav2lip"

def synthesize_lips(input_video_path, input_audio_path, output_path="uploads/wav2lip_output.mp4"):
    """
    Runs Wav2Lip inference (actual GPU-ready).
    input_video_path: path to driving video
    input_audio_path: path to target audio
    output_path: final output video
    """

    # Ensure output folder exists
    os.makedirs("uploads", exist_ok=True)

    # Build inference command
    command = [
        "python", "inference.py",
        "--checkpoint_path", os.path.join(MODELS_DIR, "wav2lip.pth"),
        "--face", input_video_path,
        "--audio", input_audio_path,
        "--outfile", output_path
    ]

    print("Running Wav2Lip command:", " ".join(command))

    subprocess.run(command, check=True)

    return output_path
