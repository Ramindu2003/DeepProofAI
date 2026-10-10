```python
import os
import json
import time
import base64
import logging

import requests
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

app = FastAPI(title="DeepProof AI API")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("deepproof")

# Configure your production website URL in Vercel
# as FRONTEND_URL. For example:
# https://your-website.vercel.app
frontend_url = os.getenv("FRONTEND_URL", "").strip()

allowed_origins = (
    [frontend_url.rstrip("/")]
    if frontend_url
    else ["*"]
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "").strip()

# Set GEMINI_MODEL in Vercel to a model supported
# by your Gemini API key.
GEMINI_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash"
).strip()

MAX_IMAGE_SIZE = 10 * 1024 * 1024
MAX_RETRIES = 3

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
    return {
        "status": "success",
        "message": "API is healthy.",
    }


def call_gemini(gemini_url, payload):
    """Call Gemini and retry temporary failures."""
    for attempt in range(MAX_RETRIES):
        response = requests.post(
            gemini_url,
            headers={
                "Content-Type": "application/json",
                "x-goog-api-key": GEMINI_API_KEY,
            },
            json=payload,
            timeout=40,
        )

        # Retry temporary server errors and rate limits.
        retryable = (
            response.status_code in (429, 500, 502, 503, 504)
        )

        if not retryable or attempt == MAX_RETRIES - 1:
            return response

        # Respect Retry-After when supplied; otherwise use backoff.
        retry_after = response.headers.get("Retry-After")

        try:
            delay = float(retry_after) if retry_after else 2 ** attempt
        except (TypeError, ValueError):
            delay = 2 ** attempt

        # Avoid waiting excessively in a serverless request.
        time.sleep(min(max(delay, 1), 8))

    return response


@app.post("/scan")
async def scan_image(file: UploadFile = File(...)):
    if not GEMINI_API_KEY:
        return error_response(
            "GEMINI_API_KEY is missing in Vercel Environment Variables.",
            500,
        )

    if not file.filename:
        return error_response("Please select an image.")

    mime_type = (
        (file.content_type or "")
        .lower()
        .split(";")[0]
        .strip()
    )

    if mime_type in ("image/jpg", "image/pjpeg"):
        mime_type = "image/jpeg"

    if mime_type not in ALLOWED_TYPES:
        return error_response(
            "Please upload a JPG, JPEG, PNG, or WEBP image."
        )

    try:
        image_bytes = await file.read(MAX_IMAGE_SIZE + 1)
    finally:
        await file.close()

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
Assess whether this image may have been AI-generated
or digitally manipulated.

Important limitations:
- Visual inspection alone cannot prove authenticity.
- Do not claim that the image is definitely real or fake.
- Do not invent forensic evidence.
- If evidence is unclear, say so in the message.
- Confidence is only the model's estimate, not a
  calibrated probability.

Return a JSON object with exactly these fields:
{
  "prediction": "FAKE" or "REAL" or "INCONCLUSIVE",
  "confidence": 50.0,
  "message": "Brief explanation and limitations."
}

Use FAKE when visual evidence suggests AI generation
or manipulation. Use REAL only when the image appears
consistent with an ordinary photograph. Use INCONCLUSIVE
when there is insufficient evidence.

Confidence must be a number from 0 to 100.
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
        response = call_gemini(gemini_url, payload)

    except requests.Timeout:
        return error_response(
            "Gemini API timed out. Please try again.",
            504,
        )

    except requests.RequestException:
        logger.exception("Gemini connection error")
        return error_response(
            "Could not connect to Gemini API. Please try again.",
            502,
        )

    try:
        result = response.json()
    except ValueError:
        logger.error("Gemini returned a non-JSON response")
        return error_response(
            "Gemini returned an invalid response.",
            502,
        )

    if not response.ok:
        details = result.get("error", {}).get(
            "message",
            "Unknown Gemini API error.",
        )

        logger.warning(
            "Gemini API returned HTTP %s",
            response.status_code,
        )

        if response.status_code == 503:
            return error_response(
                "Gemini is temporarily busy. Please try again later.",
                503,
            )

        if response.status_code == 429:
            return error_response(
                "Gemini quota or rate limit reached. Please try later.",
                429,
            )

        if response.status_code == 404:
            return error_response(
                "Gemini model unavailable. Check GEMINI_MODEL "
                "in Vercel Environment Variables.",
                502,
            )

        if response.status_code in (401, 403):
            return error_response(
                "Gemini rejected the API key or access. "
                "Check the Vercel environment variable.",
                502,
            )

        if response.status_code in (500, 502, 504):
            return error_response(
                "Gemini is temporarily unavailable. Please try again.",
                502,
            )

        # Avoid exposing API details or secrets to website visitors.
        logger.error("Gemini API error details: %s", details[:500])

        return error_response(
            f"Gemini API request failed (HTTP {response.status_code}).",
            502,
        )

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
        ).strip()

        if not text_result:
            return error_response(
                "Gemini returned an empty result.",
                502,
            )

        if text_result.startswith("```"):
            text_result = text_result.split("\n", 1)[-1]
            if text_result.endswith("```"):
                text_result = text_result[:-3].strip()

        data = json.loads(text_result)

        prediction = str(
            data.get("prediction", "")
        ).strip().upper()

        if prediction not in (
            "FAKE",
            "REAL",
            "INCONCLUSIVE",
        ):
            return error_response(
                "Invalid prediction returned by Gemini.",
                502,
            )

        try:
            confidence = float(data.get("confidence", 0))
        except (TypeError, ValueError):
            confidence = 0.0

        if not 0 <= confidence <= 100:
            confidence = max(0.0, min(100.0, confidence))

        message = str(
            data.get("message", "Assessment completed.")
        )[:1000]

        # Preserve the response fields expected by the existing frontend.
        return {
            "status": "success",
            "filename": file.filename,
            "prediction": prediction,
            "is_deepfake": prediction == "FAKE",
            "ai_confidence": f"{confidence:.2f}%",
            "message": message,
            "disclaimer": (
                "Visual AI estimate only. This result does not "
                "prove whether an image is authentic."
            ),
        }

    except (ValueError, TypeError, KeyError, IndexError):
        logger.exception("Could not parse Gemini response")
        return error_response(
            "Unexpected Gemini response format. Please try again.",
            502,
        )

    except Exception:
        logger.exception("Unexpected error in scan_image")
        return error_response(
            "An unexpected server error occurred.",
            500,
        )
```