# run_chartgemma_determine_value.py

from predict_tasks import prediction
from model import *
import sys
import re
import os

config = {
    "MODEL_NAME": "ChartGemma-2.92B",
    "MODEL_PATH": "Chart_Gemma_3B_Model",
    "PROCESSOR_PATH": "Chart_Gemma_3B_Processor",
    "RUNS_PER_IMAGE": 5,
    "SLEEP": 5,
    "IMAGE_FOLDER": "/Users/ramihuunguyen/Documents/PhD/Visualization_Literacy/Visualization_Literacy/rami/retrieve_value/retrieve_value_images",
    "OUTPUT_CSV": "/Users/ramihuunguyen/Documents/PhD/Visualization_Literacy/Visualization_Literacy/rami_retrieve_value/prediction/chartgemma_retrieve_value_output.csv",
    
    "PROMPTS": {
        "Q1": "What was the price of a barrel of oil in February 2015? Options: $57.36, $47.82, $50.24, $39.72, Omit."
    },
    "CHART_TITLES": {
        "Q1": "Oil prices spike between April and June"
    },
    "GROUND_TRUTH": {
        "Q1": {
            "answer": "$50.24",
            "question_type": "Retrieve Value"
        }
    }
}


# ✅ Dynamically assign OUTPUT_CSV for the only task
model_tag = re.sub(r'\W+', '_', config["MODEL_NAME"].lower())
os.makedirs(os.path.dirname(config["OUTPUT_CSV"]), exist_ok=True)

# === Run Chart Gemma Model ===
def run_chartgemma():
    processor, model = chart_gemma(config["MODEL_PATH"], config["PROCESSOR_PATH"])
    print("✅ ChartGemma loaded.")
    prediction(processor, model, config)


# === Run Gemini Pro Model ===
def run_gemini_pro():
    config["MODEL_NAME"] = "Gemini 2.5 Pro"
    config["OUTPUT_CSV"] = re.sub(r'\W+', '_', config["MODEL_NAME"].lower()) + "_determine_retrieve_value.csv"
    config["SLEEP"] = 5
    model = load_gemini_pro()
    print("✅ Gemini 2.5 Pro model loaded.")
    prediction(None, model, config)


# === Run Gemini Flash Model ===
def run_gemini_flash():
    config["MODEL_NAME"] = "Gemini 2.5 Flash"
    config["OUTPUT_CSV"] = re.sub(r'\W+', '_', config["MODEL_NAME"].lower()) + "_determine_retrieve_value.csv"
    config["SLEEP"] = 5
    model = load_gemini_flash()
    print("✅ Gemini 2.5 Flash model loaded.")
    prediction(None, model, config)


# === Run GPT-4o Model ===
def run_gpt_4o():
    config["MODEL_NAME"] = "chatgpt_4o"
    config["OUTPUT_CSV"] = re.sub(r'\W+', '_', config["MODEL_NAME"].lower()) + "_determine_retrieve_value.csv"
    model = load_chatgpt()
    print("✅ GPT-4o model loaded.")
    prediction(None, model, config)


# === Main Entry Point ===
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide a model to run: chart_gemma | gemini_2_5_pro | gemini_2_5_flash | gpt_4o")
        sys.exit(1)

    model_to_run = sys.argv[1].lower()

    if model_to_run == "chart_gemma":
        run_chartgemma()
    elif model_to_run == "gemini_2_5_pro":
        run_gemini_pro()
    elif model_to_run == "gemini_2_5_flash":
        run_gemini_flash()
    elif model_to_run == "gpt_4o":
        run_gpt_4o()
    else:
        print(f"❌ Unknown model: {model_to_run}")
