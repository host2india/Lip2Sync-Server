import os
import subprocess
from pathlib import Path

MODELS_DIR = "models/sadtalker"

def synthesize_talking(source_image, input_audio, output_path="uploads/sadtalker_output.mp4"):
    """
    Runs SadTalker inference pipeline (placeholder - real version will be added next).
    """

    os.makedirs("uploads", exist_ok=True)

    command = [
        "python", "inference_sadtalker.py",
        "--driven_audio", input_audio,
        "--source_image", source_image,
        "--result_dir", "uploads"
    ]

    print("Running SadTalker command:", " ".join(command))
    subprocess.run(command, check=True)

    return output_path
