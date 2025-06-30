# gemini.py

import google.generativeai as genai

API_KEY = "."  

def configure_gemini():
    """Configure the Gemini API."""
    genai.configure(api_key=API_KEY)

def load_gemini_pro():
    """Load Gemini 2.5 Pro model."""
    configure_gemini()
    model = genai.GenerativeModel("gemini-2.5-pro")
    print("✅ Gemini 2.5 Pro model loaded.")
    return model

def load_gemini_flash():
    """Load Gemini 1.5 Flash model."""
    configure_gemini()
    model = genai.GenerativeModel("gemini-2.0-flash")
    print("✅ Gemini 1.5 Flash model loaded.")
    return model
