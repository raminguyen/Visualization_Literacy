# run_chartgemma_find_extremum.py

from predict_tasks import prediction
from model import *
import sys
import re
import os


# === Configuration ===
config = {
    "MODEL_NAME": "ChartGemma-2.92B",
    "MODEL_PATH": "Chart_Gemma_3B_Model",
    "PROCESSOR_PATH": "Chart_Gemma_3B_Processor",
    "RUNS_PER_IMAGE": 5,
    "SLEEP": 5,

    "IMAGE_FOLDER": "/Users/ramihuunguyen/Documents/PhD/Visualization_Literacy/Visualization_Literacy/rami/find_extremum/find_extreum_images",

    # Default output path (will be dynamically replaced per model)
    "OUTPUT_DIR": "/Users/ramihuunguyen/Documents/PhD/Visualization_Literacy/Visualization_Literacy/rami/find_extremum/prediction",

    "PROMPTS": {
        "Q1": "In which month was the price of a barrel of oil the lowest in 2015? Options: March, May, July, December, Omit."
    },
    "CHART_TITLES": {
        "Q1": "line_data.csv"
    },
    "GROUND_TRUTH": {
        "Q1": {
            "answer": "December",
            "question_type": "Find Extremum"
        }
    }
}

# Ensure prediction output directory exists
os.makedirs(config["OUTPUT_DIR"], exist_ok=True)


def set_output_csv(model_name):
    tag = re.sub(r'\W+', '_', model_name.lower())
    config["OUTPUT_CSV"] = os.path.join(config["OUTPUT_DIR"], f"{tag}_find_extremum_output.csv")


# === Run Chart Gemma Model ===
def run_chartgemma():
    config["MODEL_NAME"] = "ChartGemma-2.92B"
    set_output_csv(config["MODEL_NAME"])
    processor, model = chart_gemma(config["MODEL_PATH"], config["PROCESSOR_PATH"])
    print("✅ ChartGemma loaded.")
    prediction(processor, model, config)

# === Run Gemini Pro Model ===
def run_gemini_pro():
    config["MODEL_NAME"] = "Gemini 2.5 Pro"
    set_output_csv(config["MODEL_NAME"])
    model = load_gemini_pro()
    print("✅ Gemini 2.5 Pro model loaded.")
    prediction(None, model, config)

# === Run Gemini Flash Model ===
def run_gemini_flash():
    config["MODEL_NAME"] = "Gemini 2.5 Flash"
    set_output_csv(config["MODEL_NAME"])
    model = load_gemini_flash()
    print("✅ Gemini 2.5 Flash model loaded.")
    prediction(None, model, config)

# === Run GPT-4o Model ===
def run_gpt_4o():
    config["MODEL_NAME"] = "chatgpt_4o"
    set_output_csv(config["MODEL_NAME"])
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
