from fastapi import FastAPI, File, UploadFile
import requests
import os

app = FastAPI(title="DeepProof AI API")

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    image_bytes = await file.read()
    
    try:
        # Vercel එකට ගැලපෙන සරල requests ක්‍රමය
        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=30)
        result = response.json()
    except Exception as e:
        return {"status": "error", "message": f"Connection Error: {str(e)}"}
    
    if isinstance(result, list) and len(result) > 0:
        best_match = result[0]
        is_deepfake = True if best_match.get('label', '').lower() == 'fake' else False
        confidence = best_match.get('score', 0) * 100
        
        return {
            "status": "success",
            "filename": file.filename,
            "is_deepfake": is_deepfake,
            "ai_confidence": f"{confidence:.2f}%",
            "message": "Warning: Deepfake Detected!" if is_deepfake else "Authentic Image"
        }
    else:
        return {"status": "error", "message": "AI System Loading. Try again in 10 seconds.", "details": result}