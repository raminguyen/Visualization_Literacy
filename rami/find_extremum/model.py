# chartgemma.py
# gemini.py
# models/llama_runner.py

from transformers import AutoProcessor, AutoModelForImageTextToText
import google.generativeai as genai
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForPreTraining, BitsAndBytesConfig
from predict_tasks import *

from transformers import (
    AutoProcessor,
    AutoModelForImageTextToText,
    AutoModelForPreTraining,
    AutoModelForVision2Seq,
    BitsAndBytesConfig
)

import os

import os
import base64
import requests

from dotenv import load_dotenv
import os

load_dotenv()  # 🔑 Loads variables from .env into environment

chatgpt_api_key = os.getenv("CHATGPT_API_KEY")
google_api_key = os.getenv("GOOGLE_API_KEY")


def _encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# === Chart Gemma ===

def chart_gemma(model_path: str, processor_path: str):
    
    """
    Load ChartGemma model and processor.

    Args:
        model_path (str): Path or name of the pretrained model.
        processor_path (str): Path or name of the processor.

    Returns:
        processor, model: Loaded HuggingFace processor and model.
    """

    # Step_1: Save Model Weight##

    # from transformers import AutoProcessor, AutoModelForImageTextToText

    # model = AutoModelForImageTextToText.from_pretrained("ahmed-masry/chartgemma")
    # processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

    # model.save_pretrained("Chart_Gemma_3B_Model")
    # processor.save_pretrained("Chart_Gemma_3B_Processor")

    # print ('It is done. Yeah!')

    
    processor = AutoProcessor.from_pretrained(processor_path)

    model = AutoModelForImageTextToText.from_pretrained(model_path)

    print("✅ Model and processor loaded.")

    return processor, model

 # === Gemini Model ===

Google_API_KEY = "AIzaSyDLJP6EfL4OWkQ4SwvWzkaP7IDQvfxZ0gs"  

#Rami: done for gemini 2.5 pro

def configure_gemini():
    """Configure the Gemini API."""
    genai.configure(api_key=google_api_key)

def load_gemini_pro():
    """Load Gemini 2.5 Pro model."""
    configure_gemini()
    model = genai.GenerativeModel("gemini-2.5-pro")
    print("✅ Gemini 2.5 Pro model loaded.")
    return model

#Rami: done for gemini 2.5 flash

def load_gemini_flash():
    """Load Gemini 1.5 Flash model."""
    configure_gemini()
    model = genai.GenerativeModel("gemini-2.5-flash")
    print("✅ Gemini 1.5 Flash model loaded.")
    return model
    
# models/chatgpt_runner.py


import base64
import requests


# ================================
# 1. Encode image as base64
# ================================
def _encode_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

# ================================
# 2. Main GPT-4o function
# ================================

def load_gpt_4o(prompt, image_path, model_name="gpt-4o"):
    """
    Query GPT-4o with a prompt and an image.

    Args:
        prompt (str): The text prompt.
        image_path (str): Path to PNG image.
        model_name (str): Model name, default is 'gpt-4o'.

    Returns:
        str: Model's answer or detailed "ERROR: ..." message.
    """
    endpoint = "https://api.openai.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {chatgpt_api_key}",
        "Content-Type": "application/json"
    }

    try:
        image_b64 = _encode_image_base64(image_path)

        payload = {
            "model": model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ],
            "max_tokens": 400
        }

        response = requests.post(endpoint, headers=headers, json=payload)
        result = response.json()

        if "error" in result:
            print("❌ ChatGPT Error:", result["error"]["message"])
            return f"ERROR: {result['error']['message']}"

        return result["choices"][0]["message"]["content"].strip()

    except Exception as e:
        print("❌ Request failed:", e)
        return f"ERROR: {str(e)}"

# ================================
# 3. Loader (for your model.py)
# ================================
def load_chatgpt():
    print("✅ GPT-4o function loaded.")
    return load_gpt_4o
