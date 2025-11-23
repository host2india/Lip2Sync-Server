# Wav2Lip Single Image ENGINE (single_image.mp4 output)

Converts a single face image + audio into a talking video (single_image.mp4).

## API
POST `/api/sync/single_image`
- form fields:
  - `image` (required) — face image file (png/jpg)
  - `audio` (optional) — wav/mp3
- returns: `single_image.mp4` (video/mp4)

## Model expectations
Place model files under repo (or pod workspace) in:

models/wav2lip/
infer.py
checkpoints/
wav2lip.pth
wav2lip_gan.pth # optional
s3fd.pth

markdown
Copy code

## Notes for Pod / Runpod
- Set `WORKSPACE` env var (inside pod) to `/workspace` (default code uses `WORKSPACE` or `.`).
- Set `WAV2LIP_MODEL_PATH` if models are located elsewhere.
- ffmpeg must be installed in the container (used for merge/image→video steps).
- Final canonical output path: `<WORKSPACE>/outputs/single_image.mp4`.

## Testing (local)
Assuming server is running on port 3000:
curl -X POST "http://localhost:3000/api/sync/single_image"
-F "image=@/path/to/face.png"
-F "audio=@/path/to/audio.wav"
--output out_single_image.mp4

pgsql
Copy code
If `audio` is omitted, a short silent video is created.

## Troubleshooting
- If you see `infer.py not found` or `checkpoint not found` — ensure `WAV2LIP_MODEL_PATH` points to the folder containing `infer.py` and `checkpoints/`.
- If ffmpeg errors appear, verify `ffmpeg` is installed and available in PATH.
- Use `WORKSPACE=/workspace` inside pod to ensure outputs are written to the mounted workspace.

