# app/engines/wav2lip_single_image/engine.py
import os
import uuid
import subprocess
from typing import Dict

from app.engines.wav2lip_single_image.utils import (
    save_uploaded_bytes,
    generate_temp_video_from_image,
    create_output_path,
)

class Wav2LipSingleImageEngine:
    """
    Production-ready single-image engine wrapper for Wav2Lip.
    - Converts an input image to a 60fps temp video
    - Runs the Wav2Lip infer.py with the supplied audio
    - Produces output: /workspace/outputs/<jobid>_single_image.mp4
    """

    def __init__(self, model_path: str = "/workspace/Wav2Lip", workspace: str = "/workspace"):
        self.model_path = model_path
        self.workspace = workspace
        self.temp_dir = os.path.join(self.workspace, "temp")
        self.outputs_dir = os.path.join(self.workspace, "outputs")
        os.makedirs(self.temp_dir, exist_ok=True)
        os.makedirs(self.outputs_dir, exist_ok=True)

    def run(self, image_bytes: bytes, audio_bytes: bytes) -> Dict:
        job_id = str(uuid.uuid4())

        # Paths
        image_path = os.path.join(self.temp_dir, f"{job_id}_img.png")
        audio_path = os.path.join(self.temp_dir, f"{job_id}.wav")
        temp_video = os.path.join(self.temp_dir, f"{job_id}_temp.mp4")
        output_path = os.path.join(self.outputs_dir, f"{job_id}_single_image.mp4")  # internal filename
        final_name = os.path.join(self.outputs_dir, "single_image.mp4")  # canonical final filename

        # 1) Save uploads
        save_uploaded_bytes(image_bytes, image_path)
        save_uploaded_bytes(audio_bytes, audio_path)

        # 2) Generate temp video from image at 60 fps
        generate_temp_video_from_image(image_path, temp_video, fps=60)

        # 3) Build and run inference command
        infer_script = os.path.join(self.model_path, "infer.py")
        checkpoint = os.path.join(self.model_path, "checkpoints", "wav2lip.pth")

        if not os.path.exists(infer_script):
            return {"status": "error", "details": f"infer.py not found at {infer_script}"}
        if not os.path.exists(checkpoint):
            return {"status": "error", "details": f"checkpoint not found at {checkpoint}"}

        command = [
            "python3",
            infer_script,
            "--checkpoint_path", checkpoint,
            "--face", temp_video,
            "--audio", audio_path,
            "--outfile", output_path
        ]

        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            return {"status": "error", "details": str(e)}

        # 4) Move/rename output to canonical final filename (overwrite if exists)
        try:
            if os.path.exists(final_name):
                os.remove(final_name)
            os.rename(output_path, final_name)
        except Exception:
            # If rename fails, fall back to returning the specific job output path
            return {"status": "success", "output_path": output_path}

        return {"status": "success", "output_path": final_name}
