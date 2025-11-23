from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import FileResponse, JSONResponse
import shutil, os, uuid
from pathlib import Path
from .wav2lip_engine import synthesize_lips
from .sadtalker_engine import synthesize_talking

app = FastAPI(title="Lip2Sync API")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

def save_upload(upload: UploadFile, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    ext = Path(upload.filename).suffix or ""
    unique = f"{uuid.uuid4().hex}{ext}"
    dest = dest_dir / unique
    with dest.open("wb") as f:
        shutil.copyfileobj(upload.file, f)
    return dest

@app.get("/")
def root():
    return {"status":"ok","service":"Lip2Sync-Server"}

# Minimal Wav2Lip endpoint
# Accepts either a video or an image as 'source' and an audio file 'audio'
@app.post("/api/wav2lip")
async def api_wav2lip(source: UploadFile = File(...), audio: UploadFile = File(...)):
    src_path = save_upload(source, UPLOAD_DIR)
    audio_path = save_upload(audio, UPLOAD_DIR)
    out_path = UPLOAD_DIR / f"wav2lip_{uuid.uuid4().hex}.mp4"
    try:
        output = synthesize_lips(str(src_path), str(audio_path), str(out_path))
        return FileResponse(output, media_type="video/mp4", filename=Path(output).name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Minimal SadTalker endpoint
# Accepts a single source image and an audio file
@app.post("/api/sadtalker")
async def api_sadtalker(source: UploadFile = File(...), audio: UploadFile = File(...)):
    src_path = save_upload(source, UPLOAD_DIR)
    audio_path = save_upload(audio, UPLOAD_DIR)
    out_path = UPLOAD_DIR / f"sadtalker_{uuid.uuid4().hex}.mp4"
    try:
        output = synthesize_talking(str(src_path), str(audio_path), str(out_path))
        return FileResponse(output, media_type="video/mp4", filename=Path(output).name)
    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# Register SadTalker Single Image Endpoint
from app.routes.sadtalker_single import router as sadtalker_single_router
app.include_router(sadtalker_single_router)
