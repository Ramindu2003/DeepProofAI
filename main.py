from fastapi import FastAPI, File, UploadFile
import requests
import os

app = FastAPI(title="DeepProof AI API")

HF_API_KEY = os.getenv("HF_API_KEY")

API_URL = "https://router.huggingface.co/hf-inference/models/dima806/deepfake_vs_real_image_detection"


@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):

    if not HF_API_KEY:
        return {
            "status": "error",
            "message": "HF_API_KEY is not configured in Vercel."
        }

    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/octet-stream"
    }

    image_bytes = await file.read()

    try:
        response = requests.post(
            API_URL,
            headers=headers,
            data=image_bytes,
            timeout=60
        )

        # HTTP error එකක් නම් details ගන්න
        if response.status_code != 200:
            return {
                "status": "error",
                "http_status": response.status_code,
                "message": response.text
            }

        result = response.json()

    except Exception as e:
        return {
            "status": "error",
            "message": f"Connection Error: {str(e)}"
        }

    if isinstance(result, list) and len(result) > 0:

        # Highest score එක තියෙන prediction එක
        best_match = max(
            result,
            key=lambda x: x.get("score", 0)
        )

        label = best_match.get("label", "").lower()
        confidence = best_match.get("score", 0) * 100

        is_deepfake = label == "fake"

        return {
            "status": "success",
            "filename": file.filename,
            "prediction": "FAKE" if is_deepfake else "REAL",
            "is_deepfake": is_deepfake,
            "ai_confidence": f"{confidence:.2f}%",
            "message": (
                "Warning: AI-generated / Fake image detected!"
                if is_deepfake
                else "Image appears to be Real."
            )
        }

    return {
        "status": "error",
        "message": "AI returned an unexpected response.",
        "details": result
    }