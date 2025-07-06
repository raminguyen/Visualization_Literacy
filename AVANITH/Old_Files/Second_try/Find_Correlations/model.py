# === model.py ===
# All model-loading and utility functions for Determine Range predictions.

import os
import google.generativeai as genai
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image


# --------- Load .env keys ----------
def load_api_keys():
    """
    Loads both Gemini API keys from .env and .env1 files.
    Returns: Tuple of (api_key1, api_key2)
    """
    load_dotenv(dotenv_path=".env")
    api_key1 = os.getenv("api_key")

    load_dotenv(dotenv_path=".env1")
    api_key2 = os.getenv("api_key2")

    if not api_key1 or not api_key2:
        raise ValueError("❌ API keys not found in .env/.env1")
    
    return api_key1, api_key2


# --------- Load Gemini model ----------
def load_gemini_model(api_key: str, model_version: str = "gemini-2.5-pro"):
    """
    Initializes a Gemini model using the provided API key and version.
    """
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_version)


# --------- Load ChartGemma model ----------
def load_chartgemma_model(model_path: str, processor_path: str):
    """
    Loads ChartGemma model and processor from local directory.
    """
    model = AutoModelForImageTextToText.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    return model, processor


# --------- Load Image ----------
def load_image(image_path: str):
    """
    Opens and converts an image to RGB.
    """
    return Image.open(image_path).convert("RGB")
