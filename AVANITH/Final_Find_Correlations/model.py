# === model.py ===
# All model-loading and utility functions for Determine Range predictions.

import os
import google.generativeai as genai
from openai import OpenAI
from dotenv import load_dotenv
from transformers import AutoProcessor, AutoModelForImageTextToText
from PIL import Image

# --------- Load .env keys ----------
def load_api_keys():
    """
    Loads Gemini and ChatGPT API keys from .env and .env1 files.
    """
    load_dotenv(dotenv_path=".env")
    api_key1 = os.getenv("api_key")
    api_key2 = os.getenv("api_key2")
    api_key3 = os.getenv("api_key3")  # ChatGPT

    if not api_key1 or not api_key2 or not api_key3:
        raise ValueError("❌ One or more API keys missing in .env/.env1")

    return api_key1, api_key2, api_key3


# --------- Load Gemini model ----------
def load_gemini_model(api_key: str, model_version: str = "gemini-2.5-pro"):
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(model_version)


# --------- Load ChartGemma model ----------
def load_chartgemma_model(model_path: str, processor_path: str):
    model = AutoModelForImageTextToText.from_pretrained(model_path)
    processor = AutoProcessor.from_pretrained(processor_path)
    return model, processor


# --------- Load ChatGPT client ----------
def load_chatgpt_client(api_key: str):
    return OpenAI(api_key=api_key)


# --------- Load Image ----------
def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")
