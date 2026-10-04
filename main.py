from fastapi import FastAPI, UploadFile, File
import cv2
import numpy as np
from PIL import Image
import io

# 1. API එක ආරම්භ කිරීම
app = FastAPI(title="DeepProof AI Verification API")

# 2. Photo එක Upload කරන Endpoint එක
@app.post("/scan")
async def scan_media(file: UploadFile = File(...)):
    try:
        # Photo එකේ Data ටික කියවා ගැනීම
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        img_np = np.array(image)
        
        # Photo එක කළු-සුදු (Grayscale) කිරීම (ලේසියෙන් process කරන්න)
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        
        # Laplacian Variance එකෙන් Photo එකේ දාර (Edges/Sharpness) මැනීම
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Score එක 100 ට අඩු නම්, ඒක AI හදපු හෝ Blurry එකක් කියලා අනුමාන කරනවා
        is_synthetic = bool(laplacian_var < 100)
        
        # Result එක JSON විදියට එළියට දෙනවා
        return {
            "status": "success",
            "filename": file.filename,
            "is_deepfake": is_synthetic,
            "metrics": {
                "sharpness_score": round(laplacian_var, 2)
            },
            "message": "Warning: Fake or Blurry" if is_synthetic else "Authentic"
        }
    except Exception as e:
        return {"status": "error", "message": str(e)}