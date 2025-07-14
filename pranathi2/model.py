# =========================
# model.py
# =========================

# This script contains the implementation logic for each of the three models:
# Gemini 2.5 Pro, Gemini 2.5 Flash, and ChartGemma

import google.generativeai as genai
import base64, time
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForImageTextToText

# ............................ Gemini Configuration ............................
def configure_gemini(api_key):
    genai.configure(api_key=api_key)

# Helper: encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# ............................ Gemini 2.5 Pro Prediction ............................
def predict_with_gemini_pro(image_path, prompt):
    model = genai.GenerativeModel("gemini-2.5-pro")
    image_data = {
        "inline_data": {
            "mime_type": "image/png",
            "data": encode_image(image_path)
        }
    }
    start = time.time()
    response = model.generate_content([prompt, image_data])
    end = time.time()
    return response.text.strip(), round(end - start, 2)

# ............................ Gemini 2.5 Flash Prediction ............................
def predict_with_gemini_flash(image_path, prompt):
    model = genai.GenerativeModel("gemini-2.5-flash")
    image_data = {
        "inline_data": {
            "mime_type": "image/png",
            "data": encode_image(image_path)
        }
    }
    start = time.time()
    response = model.generate_content([prompt, image_data])
    end = time.time()
    return response.text.strip(), round(end - start, 2)

# ............................ ChartGemma Prediction ............................

# Load ChartGemma model and processor once
processor_cg = AutoProcessor.from_pretrained("Chart_Gemma_3B_Processor")
model_cg = AutoModelForImageTextToText.from_pretrained("Chart_Gemma_3B_Model")
model_cg = model_cg.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

def predict_with_chartgemma(image_path, prompt):
    image = Image.open(image_path).convert("RGB")
    inputs = processor_cg(images=image, text=prompt, return_tensors="pt").to(model_cg.device)
    start = time.time()
    with torch.no_grad():
        output_ids = model_cg.generate(**inputs, max_new_tokens=20)
    end = time.time()
    prediction = processor_cg.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    return prediction, round(end - start, 2)

