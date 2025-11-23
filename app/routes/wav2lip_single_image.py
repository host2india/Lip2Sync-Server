# app/routes/wav2lip_single_image.py
from fastapi import APIRouter, UploadFile, File
from fastapi.responses import FileResponse, JSONResponse
from app.engines.wav2lip_single_image.engine import Wav2LipSingleImageEngine

router = APIRouter()
engine = Wav2LipSingleImageEngine(model_path="Wav2Lip", workspace=".")

@router.post("/sync/single_image")
async def sync_single_image(image: UploadFile = File(...), audio: UploadFile = File(...)):
    # Read bytes
    image_bytes = await image.read()
    audio_bytes = await audio.read()

    result = engine.run(image_bytes, audio_bytes)

    if result.get("status") == "success":
        return FileResponse(result["output_path"], media_type="video/mp4", filename="single_image.mp4")
    return JSONResponse({"status": "error", "details": result.get("details", "unknown")}, status_code=500)
