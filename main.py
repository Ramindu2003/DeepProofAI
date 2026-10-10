
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import os
import base64
import json
import requests

app = FastAPI(title="DeepProof AI API")

# Production website URL එක දන්නා පසු allow_origins සීමා කරන්න.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash").strip()
MAX_IMAGE_SIZE = 10 * 1024 * 1024

ALLOWED_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}


def error_response(message, status_code=400):
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": message,
        },
    )


@app.get("/")
def home():
    return {
        "status": "success",
        "message": "DeepProof AI API is running!",
        "model": GEMINI_MODEL,
    }


@app.get("/health")
def health():
    return {"status": "success", "message": "API is healthy."}


@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):

    if not GEMINI_API_KEY:
        return error_response(
            "GEMINI_API_KEY is missing in Vercel Environment Variables.",
            500,
        )

    if not file.filename:
        return error_response("Please select an image.")

    mime_type = (file.content_type or "").lower().split(";")[0].strip()

    if mime_type in ("image/jpg", "image/pjpeg"):
        mime_type = "image/jpeg"

    if mime_type not in ALLOWED_TYPES:
        return error_response(
            "Please upload a JPG, JPEG, PNG, or WEBP image."
        )

    image_bytes = await file.read(MAX_IMAGE_SIZE + 1)

    if not image_bytes:
        return error_response("The uploaded image is empty.")

    if len(image_bytes) > MAX_IMAGE_SIZE:
        return error_response(
            "Image is too large. Maximum size is 10 MB."
        )

    base64_image = base64.b64encode(
        image_bytes
    ).decode("ascii")

    gemini_url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{GEMINI_MODEL}:generateContent"
    )

    prompt = """
Analyze this image for visual signs that may suggest
AI generation or image manipulation.

Visual analysis alone cannot prove authenticity.
Do not claim forensic certainty. If the evidence is
unclear, explain that in the message.

Return only a JSON object with these fields:
{
  "prediction": "FAKE" or "REAL",
  "confidence": 50.0,
  "message": "Brief explanation with uncertainty noted."
}

Confidence must be a number from 0 to 100.
It is the model's estimate, not a calibrated probability.
"""

    payload = {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {
                        "inline_data": {
                            "mime_type": mime_type,
                            "data": base64_image,
                        }
                    },
                ]
            }
        ],
        "generationConfig": {
            "response_mime_type": "application/json",
            "temperature": 0.1,
            "max_output_tokens": 300,
        },
    }

    try:
        response = requests.post(
            gemini_url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY,
            },
            json=payload,
            timeout=50,
        )
    except requests.Timeout:
        return error_response(
            "Gemini API timed out. Please try again.",
            504,
        )
    except requests.RequestException:
        return error_response(
            "Could not connect to Gemini API.",
            502,
        )

    try:
        result = response.json()
    except ValueError:
        return error_response(
            "Gemini returned an invalid response.",
            502,
        )

    if not response.ok:
        details = result.get("error", {}).get(
            "message", "Unknown Gemini API error."
        )

        if response.status_code == 404:
            message = (
                f"Model '{GEMINI_MODEL}' is unavailable for this API key. "
                "Check the available models and GEMINI_MODEL setting. "
                f"Details: {details}"
            )
        elif response.status_code in (401, 403):
            message = (
                "Gemini rejected the API key or access. "
                f"Check the key and API permissions. Details: {details}"
            )
        elif response.status_code == 429:
            message = (
                "Gemini quota or rate limit reached. "
                f"Details: {details}"
            )
        else:
            message = (
                f"Gemini API error ({response.status_code}): {details}"
            )

        return error_response(message, 502)

    try:
        candidates = result.get("candidates", [])

        if not candidates:
            return error_response(
                "Gemini did not return an assessment.",
                502,
            )

        parts = candidates[0].get(
            "content", {}
        ).get("parts", [])

        text_result = next(
            (
                part["text"]
                for part in parts
                if part.get("text")
            ),
            "",
        )

        if not text_result:
            return error_response(
                "Gemini returned an empty result.",
                502,
            )

        text_result = text_result.strip()

        if text_result.startswith("```"):
            text_result = text_result.split("\n", 1)[-1]
            if text_result.endswith("```"):
                text_result = text_result[:-3].strip()

        data = json.loads(text_result)

        prediction = str(
            data.get("prediction", "")
        ).strip().upper()

        if prediction not in ("FAKE", "REAL"):
            return error_response(
                "Invalid prediction returned by Gemini.",
                502,
            )

        try:
            confidence = float(data.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        confidence = max(0.0, min(100.0, confidence))
        message = str(
            data.get("message", "Assessment completed.")
        )

        # Keep the response field names used by your frontend.
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": prediction,
            "is_deepfake": prediction == "FAKE",
            "ai_confidence": f"{confidence:.2f}%",
            "message": message,
            "disclaimer": (
                "AI visual estimate only; not forensic verification."
            ),
        }

    except (ValueError, TypeError, KeyError, IndexError):
        return error_response(
            "Unexpected Gemini response format. Try again.",
            502,
        )

    except Exception:
        return error_response(
            "An unexpected server error occurred.",
            500,
        )
