# =========================
# predict_tasks.py
# =========================

# This script is a helper used to run predictions on a single task configuration.
# It loads the image and prompt, runs the selected model, and returns predictions.

from model import (
    predict_with_gemini_pro,
    predict_with_gemini_flash,
    predict_with_chartgemma,
    configure_gemini
)

PROMPT = "How much did the price of a barrel of oil fall from April to September in 2015? Options: $4, $15, $17, $45, Omit"
GROUND_TRUTH = "$15"

# Main prediction wrapper

def run_task(image_path, model_name, api_key=""):
    if model_name.startswith("gemini"):
        configure_gemini(api_key)

    model_func = {
        "gemini_pro": predict_with_gemini_pro,
        "gemini_flash": predict_with_gemini_flash,
        "chartgemma": predict_with_chartgemma
    }[model_name]

    prediction, duration = model_func(image_path, PROMPT)
    is_correct = GROUND_TRUTH in prediction

    return {
        "image": image_path,
        "prediction": prediction,
        "time": duration,
        "correct": is_correct
    }
