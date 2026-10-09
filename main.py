from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import requests
import os

app = FastAPI(title="DeepProof AI API")

# Frontend එකට කතා කරන්න අවසර දීම (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

HF_API_KEY = os.getenv("HF_API_KEY")
API_URL = "https://router.huggingface.co/hf-inference/models/google/vit-base-patch16-224"

@app.get("/")
def home():
    return {"status": "success", "message": "DeepProof AI API is running!"}

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    if not HF_API_KEY:
        return {"status": "error", "message": "HF_API_KEY is not configured in Vercel."}
        
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"status": "error", "message": "Please upload an image file."}
        
    image_bytes = await file.read()
    if not image_bytes:
        return {"status": "error", "message": "Uploaded image is empty."}
        
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": file.content_type
    }
    
    try:
        response = requests.post(API_URL, headers=headers, data=image_bytes, timeout=60)
        result = response.json()
    except Exception as e:
        return {"status": "error", "message": f"Connection Error: {str(e)}"}
        
    if response.status_code != 200:
        return {"status": "error", "http_status": response.status_code, "details": result}
        
    if not isinstance(result, list) or len(result) == 0:
        return {"status": "error", "message": "AI returned an unexpected result.", "raw_result": result}
        
    best_match = max(result, key=lambda x: x.get("score", 0))
    label = str(best_match.get("label", "")).lower()
    confidence = float(best_match.get("score", 0)) * 100
    
    fake_keywords = ["fake", "artificial", "ai", "generated"]
    real_keywords = ["real", "human", "original", "authentic"]
    
    if any(keyword in label for keyword in fake_keywords):
        is_deepfake = True
        prediction = "FAKE"
        message = "Warning: AI-generated / Fake image detected!"
    elif any(keyword in label for keyword in real_keywords):
        is_deepfake = False
        prediction = "REAL"
        message = "Image appears to be Authentic/Human."
    else:
        is_deepfake = None
        prediction = best_match.get("label", "UNKNOWN")
        message = "Model returned an unknown label."
        
    return {
        "status": "success",
        "filename": file.filename,
        "prediction": prediction,
        "is_deepfake": is_deepfake,
        "ai_confidence": f"{confidence:.2f}%",
        "message": message,
        "raw_model_result": result
    }
