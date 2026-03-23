"""
Fashion Florence API — HuggingFace Space wrapping anushreeberlia/fashion-florence.

Endpoints:
  POST /analyze  — accepts image file, returns structured fashion tags as JSON
  GET  /health   — readiness check (model loaded?)
"""

import io
import json
import logging

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MODEL_NAME = "anushreeberlia/fashion-florence"
PROMPT = "Analyze this clothing item image and return structured fashion tags as JSON."

app = FastAPI(title="Fashion Florence API")

device = "cuda:0" if torch.cuda.is_available() else "cpu"
dtype = torch.float16 if torch.cuda.is_available() else torch.float32

logger.info("Loading %s on %s (%s)...", MODEL_NAME, device, dtype)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=dtype, trust_remote_code=True,
).to(device)
processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
logger.info("Model loaded.")


@app.get("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME, "device": device}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Empty file")

    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
        inputs = processor(
            text=PROMPT, images=image, return_tensors="pt",
        ).to(device, dtype)

        with torch.no_grad():
            generated_ids = model.generate(
                input_ids=inputs["input_ids"],
                pixel_values=inputs["pixel_values"],
                max_new_tokens=256,
                num_beams=3,
            )

        result_text = processor.batch_decode(
            generated_ids, skip_special_tokens=True
        )[0]

        if PROMPT in result_text:
            result_text = result_text.split(PROMPT, 1)[-1].strip()

        text = result_text.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            text = "\n".join(lines[1:-1])

        return json.loads(text)

    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail=f"Model returned invalid JSON: {text}")
    except Exception as e:
        logger.error("Inference error: %s", e)
        raise HTTPException(status_code=500, detail=str(e))
