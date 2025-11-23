# app/engines/wav2lip_single_image/engine.py
import os
import uuid
import subprocess
from typing import Dict, Optional

from app.engines.wav2lip_single_image.utils import (
    save_uploaded_bytes,
    ensure_outputs_dir,
    merge_audio_video_if_needed,
)

class Wav2LipSingleImageEngine:
    """
    Wav2Lip Single Image Engine:
      - Input: single face image + audio (wav/mp3)
      - Output: talking video file named `single_image.mp4` in outputs dir
      - Expects model folder structure:
          models/wav2lip/
             infer.py
             checkpoints/wav2lip.pth
             checkpoints/wav2lip_gan.pth (optional)
             checkpoints/s3fd.pth
    """

    def __init__(self, model_path: str = "models/wav2lip", workspace: Optional[str] = None):
        env_ws = os.environ.get("WORKSPACE")
        if workspace is None:
            workspace = env_ws if env_ws else "."
        self.model_path = model_path
        self.workspace = workspace

        # temp + outputs inside workspace so Pod can mount /workspace
        self.temp_dir = os.path.join(self.workspace, "temp_single_image")
        self.outputs_dir = os.path.join(self.workspace, "outputs")
        os.makedirs(self.temp_dir, exist_ok=True)
        ensure_outputs_dir(self.outputs_dir)

    def run(self, image_bytes: bytes, audio_bytes: Optional[bytes] = None) -> Dict:
        """
        Run the single-image pipeline:
          - Save uploads to temp
          - If audio provided, create a merged input video using a short "still image -> video" step
            (we will generate a temporary silent video from the image and then merge the provided audio).
          - Call model infer.py with appropriate args to generate talking video.
          - Move/rename to canonical outputs/single_image.mp4 (overwrite if exists)
        Returns: dict with status and path or error details.
        """
        job_id = str(uuid.uuid4())

        # file paths
        img_in = os.path.join(self.temp_dir, f"{job_id}_in_image.png")
        audio_in = None
        if audio_bytes:
            audio_in = os.path.join(self.temp_dir, f"{job_id}_in_audio.wav")

        # intermediate video created from image (image -> looping video)
        image_video = os.path.join(self.temp_dir, f"{job_id}_image_video.mp4")
        job_output = os.path.join(self.outputs_dir, f"{job_id}_single_image.mp4")
        final_output = os.path.join(self.outputs_dir, "single_image.mp4")

        # Save uploaded bytes
        try:
            save_uploaded_bytes(image_bytes, img_in)
            if audio_bytes:
                save_uploaded_bytes(audio_bytes, audio_in)
        except Exception as e:
            return {"status": "error", "details": f"failed to save uploads: {e}"}

        # Create a short video from the image ( ffmpeg -loop 1 -i img -c:v libx264 -t <duration> ... )
        # Duration: if audio provided use audio duration implicitly by merging; otherwise create 5s video.
        # We'll generate a 1-second silent video placeholder and let model handle audio; but when audio_in present,
        # we create a longer video with same duration as audio (fallback to 10s).
        try:
            # if audio_in is present, create a video long enough for audio; ffmpeg will respect audio length when merging later
            if audio_in:
                # generate silent video for merging: produce a long video (we will merge audio into it)
                # using ffmpeg: loop image, set framerate 25, output `image_video`
                cmd_img_vid = [
                    "ffmpeg", "-y", "-loop", "1", "-i", img_in,
                    "-c:v", "libx264", "-t", "60", "-pix_fmt", "yuv420p",
                    "-vf", "scale=640:-2",
                    image_video
                ]
                subprocess.run(cmd_img_vid, check=True)
                # merge audio into that video (so Face input is video with audio)
                merged_input = os.path.join(self.temp_dir, f"{job_id}_merged_input.mp4")
                merge_audio_video_if_needed(image_video, audio_in, merged_input)
                face_input = merged_input
            else:
                # if no audio provided, just produce a short video (5s) to feed model
                cmd_img_vid = [
                    "ffmpeg", "-y", "-loop", "1", "-i", img_in,
                    "-c:v", "libx264", "-t", "5", "-pix_fmt", "yuv420p",
                    "-vf", "scale=640:-2",
                    image_video
                ]
                subprocess.run(cmd_img_vid, check=True)
                face_input = image_video
        except subprocess.CalledProcessError as e:
            return {"status": "error", "details": f"ffmpeg failed while creating image video: {e}"}
        except Exception as e:
            return {"status": "error", "details": f"image video creation error: {e}"}

        # Validate model existence
        infer_script = os.path.join(self.model_path, "infer.py")
        checkpoint = os.path.join(self.model_path, "checkpoints", "wav2lip.pth")
        if not os.path.exists(infer_script):
            return {"status": "error", "details": f"infer.py not found at {infer_script}"}
        if not os.path.exists(checkpoint):
            return {"status": "error", "details": f"checkpoint not found at {checkpoint}"}

        # Build command (prefer model-specific single-image flags if available; fallback to --face/--audio)
        command = [
            "python3",
            infer_script,
            "--checkpoint_path", checkpoint,
            "--face", face_input,
            "--outfile", job_output
        ]

        # If audio_bytes exists and we didn't already attach it into face_input, pass --audio (but above we merged)
        # (No-op here - we already merged audio into face_input)
        try:
            subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            # capture stderr would be useful but avoid exposing too much; include returncode
            return {"status": "error", "details": f"inference failed: returncode {e.returncode}"}

        # Move to canonical filename
        try:
            if os.path.exists(final_output):
                os.remove(final_output)
            os.rename(job_output, final_output)
        except Exception:
            # fallback to returning job_output path
            return {"status": "success", "output_path": job_output}

        return {"status": "success", "output_path": final_output}

