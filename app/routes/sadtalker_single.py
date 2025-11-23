from fastapi import APIRouter, UploadFile, File
from app.engines.sadtalker.engine import SadTalkerEngine

router = APIRouter(prefix="/v1/sadtalker", tags=["SadTalker"])

engine = SadTalkerEngine()

@router.post("/single")
async def sadtalker_single(
    image: UploadFile = File(...),
    audio: UploadFile = File(...)
):
    """
    Single image → full animated head using SadTalker.
    """
    output_path = await engine.infer_from_uploads(image, audio)
    return {"output_video": output_path}
