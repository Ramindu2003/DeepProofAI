from fastapi import FastAPI, File, UploadFile
import requests
import os

app = FastAPI(title="DeepProof AI API")

# Hugging Face API Key
HF_API_KEY = os.getenv("HF_API_KEY")

# Deepfake detection model
API_URL = "https://router.huggingface.co/hf-inference/models/dima806/deepfake_vs_real_image_detection"


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "DeepProof AI API is running!"
    }


@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):

    # Check API key
    if not HF_API_KEY:
        return {
            "status": "error",
            "message": "HF_API_KEY is not configured in Vercel."
        }

    # Check file type
    if not file.content_type or not file.content_type.startswith("image/"):
        return {
            "status": "error",
            "message": "Please upload an image file."
        }

    # Read image
    image_bytes = await file.read()

    if not image_bytes:
        return {
            "status": "error",
            "message": "Uploaded image is empty."
        }

    # Hugging Face headers
    headers = {
        "Authorization": f"Bearer {HF_API_KEY}",
        "Content-Type": "application/octet-stream"
    }

    try:

        # Send image to Hugging Face
        response = requests.post(
            API_URL,
            headers=headers,
            data=image_bytes,
            timeout=60
        )

        # Convert response to JSON
        try:
            result = response.json()
        except Exception:
            return {
                "status": "error",
                "message": "Hugging Face returned an invalid response.",
                "http_status": response.status_code,
                "raw_response": response.text
            }

    except requests.exceptions.Timeout:
        return {
            "status": "error",
            "message": "Hugging Face request timed out. Please try again."
        }

    except requests.exceptions.ConnectionError as e:
        return {
            "status": "error",
            "message": "Could not connect to Hugging Face.",
            "details": str(e)
        }

    except Exception as e:
        return {
            "status": "error",
            "message": "Unexpected connection error.",
            "details": str(e)
        }

    # HTTP error from Hugging Face
    if response.status_code != 200:

        return {
            "status": "error",
            "http_status": response.status_code,
            "message": "Hugging Face API returned an error.",
            "details": result
        }

    # Make sure response is a list
    if not isinstance(result, list) or len(result) == 0:

        return {
            "status": "error",
            "message": "AI returned an unexpected result.",
            "raw_result": result
        }

    # Find highest confidence prediction
    best_match = max(
        result,
        key=lambda x: x.get("score", 0)
    )

    # Get label and score
    label = str(best_match.get("label", "")).lower()
    confidence = float(best_match.get("score", 0)) * 100

    # Determine Fake / Real
    if label == "fake":
        is_deepfake = True
        prediction = "FAKE"
        message = "Warning: Fake / AI-generated image detected!"

    elif label == "real":
        is_deepfake = False
        prediction = "REAL"
        message = "Image appears to be Real."

    else:
        # Unknown label
        is_deepfake = None
        prediction = best_match.get("label", "UNKNOWN")
        message = "Model returned an unknown label."

    # Final response
    return {
        "status": "success",

        "filename": file.filename,

        "prediction": prediction,

        "is_deepfake": is_deepfake,

        "ai_confidence": f"{confidence:.2f}%",

        "message": message,

        # IMPORTANT:
        # This lets us see exactly what Hugging Face returned.
        "raw_model_result": result
    }