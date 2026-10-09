from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import requests
import os
import base64
import json

app = FastAPI(title="DeepProof AI API")

# Frontend එකට කතා කරන්න අවසර දීම (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vercel එකේ සේව් කරපු Gemini API Key එක ගැනීම
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

@app.get("/")
def home():
    return {"status": "success", "message": "DeepProof AI API is running with Google Gemini!"}

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    # API Key එක තියෙනවද කියලා චෙක් කිරීම
    if not GEMINI_API_KEY:
        return {"status": "error", "message": "GEMINI_API_KEY is not configured in Vercel."}
        
    if not file.content_type or not file.content_type.startswith("image/"):
        return {"status": "error", "message": "Please upload an image file."}
        
    image_bytes = await file.read()
    if not image_bytes:
        return {"status": "error", "message": "Uploaded image is empty."}
        
    # Google Gemini එකට යවන්න ෆොටෝ එක Base64 (Text) විදියට හරවා ගැනීම
    base64_image = base64.b64encode(image_bytes).decode('utf-8')
    
    # Gemini 1.5 Flash Model URL එක
    gemini_url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={GEMINI_API_KEY}"
    
    # AI එකට දෙන උපදෙස් (Prompt)
    prompt_text = """Analyze this image carefully. Is it a real authentic photograph or an AI-generated image (Deepfake)? 
    Return ONLY a valid JSON object strictly matching this format:
    {
        "prediction": "FAKE" or "REAL",
        "confidence": 98.5,
        "message": "A short sentence explaining why you think it is real or fake."
    }"""
    
    # Gemini API එකට යවන දත්ත සැකසීම
    payload = {
        "contents": [{
            "parts": [
                {"text": prompt_text},
                {
                    "inline_data": {
                        "mime_type": file.content_type,
                        "data": base64_image
                    }
                }
            ]
        }],
        "generationConfig": {
            "response_mime_type": "application/json" # අනිවාර්යයෙන්ම JSON උත්තරයක් ඉල්ලීම
        }
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        # Gemini එකට Request එක යැවීම
        response = requests.post(gemini_url, headers=headers, json=payload, timeout=60)
        result = response.json()
    except Exception as e:
        return {"status": "error", "message": f"Connection Error: {str(e)}"}
        
    if response.status_code != 200:
        return {"status": "error", "http_status": response.status_code, "details": result}
        
    try:
        # Gemini එවපු JSON උත්තරේ කොටස වෙන් කරගෙන කියවීම
        ai_text_response = result["candidates"][0]["content"]["parts"][0]["text"]
        ai_data = json.loads(ai_text_response)
        
        prediction = ai_data.get("prediction", "UNKNOWN")
        confidence = float(ai_data.get("confidence", 0))
        message = ai_data.get("message", "Scanned successfully.")
        
        is_deepfake = True if prediction == "FAKE" else False
        
        # Frontend එකට කලින් විදියටම රිසල්ට් එක යැවීම (එතකොට HTML වෙනස් කරන්න ඕන නෑ)
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": prediction,
            "is_deepfake": is_deepfake,
            "ai_confidence": f"{confidence:.2f}%",
            "message": message,
            "raw_model_result": ai_data
        }
        
    except Exception as e:
        return {"status": "error", "message": "AI returned an unexpected format.", "raw_result": result}