from fastapi import FastAPI, File, UploadFile
import uvicorn
import requests
import os

app = FastAPI(title="DeepProof AI API")

# Render එකේ සඟවලා තියෙන API Key එක මෙතනින් කෝඩ් එකට ගන්නවා
HF_API_KEY = os.getenv("HF_API_KEY")
# Hugging Face එකේ තියෙන Deepfake Detection AI මොඩල් එකේ ලින්ක් එක
API_URL = "https://api-inference.huggingface.co/models/dima806/deepfake_vs_real_image_detection"

@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    headers = {"Authorization": f"Bearer {HF_API_KEY}"}
    image_bytes = await file.read()
    
    # AI මොළයට ෆොටෝ එක යැවීම
    response = requests.post(API_URL, headers=headers, data=image_bytes)
    result = response.json()
    
    # AI එකෙන් එන උත්තරේ විශ්ලේෂණය කිරීම
    if isinstance(result, list) and len(result) > 0:
        best_match = result[0]
        # AI එක කියන විදියට මේක Fake ද (Deepfake) කියලා බලමු
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
        return {"status": "error", "message": "AI System Loading or Error. Try again in 10 seconds."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)