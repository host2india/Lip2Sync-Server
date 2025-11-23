# import the route
from app.routes.wav2lip_single_image import router as wav2lip_single_image_router

# then include router (near other app.include_router calls)
app.include_router(wav2lip_single_image_router, prefix="/api")
