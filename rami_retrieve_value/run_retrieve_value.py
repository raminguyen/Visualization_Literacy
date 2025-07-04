# run_chartgemma_determine_value.py

from predict_tasks import prediction
from model import*
import sys
import re 

import re

config = {
    "MODEL_NAME": "ChartGemma-2.92B",
    "MODEL_PATH": "Chart_Gemma_3B_Model",
    "PROCESSOR_PATH": "Chart_Gemma_3B_Processor",
    "IMAGE_FOLDER": "/Users/ramihuunguyen/Documents/PhD/Visualization_Literacy/retrieve_value/retrieve_value",

    "OUTPUT_CSV": "chartgemma_retrieve_value_output.csv",

    "RUNS_PER_IMAGE": 5,

    "PROMPTS": {
        "Q": "What was the price of a barrel of oil in February 2015? Options: $57.36, $47.82, $50.24, $39.72, Omit."
    },
    "CHART_TITLES": {
        "Q": "Oil prices spike between April and June"
    },
    "GROUND_TRUTH": {
        "Q": {
            "answer": "$50.24",
            "question_type": "Retrieve Value"
        }
    }
}



# 🔧 Set dynamic OUTPUT_CSV filename based on model name
model_tag = re.sub(r'\W+', '_', config["MODEL_NAME"].lower())
config["OUTPUT_CSV"] = f"{model_tag}_determine_retrieve_value.csv"


 # === Run Chart Gemma Model ===

def run_chartgemma():
   
    processor, model = chart_gemma(config["MODEL_PATH"], config["PROCESSOR_PATH"])
    print("✅ ChartGemma loaded.")
    prediction(processor, model, config)
    

# === Run Gemini Models ===

def run_gemini_pro():
    config["MODEL_NAME"] = "Gemini 2.5 Pro"
    config["OUTPUT_CSV"] = re.sub(r'\W+', '_', config["MODEL_NAME"].lower()) + "_determine_retrieve_value.csv"
    config["SLEEP"] = 5 # sleep 2 seconds between runs

    model = load_gemini_pro()
    print("✅ Gemini 2.5 Pro model loaded.")
    prediction(None, model, config)

# === Run Gemini Flash Model ===

def run_gemini_flash():
    config["MODEL_NAME"] = "Gemini 2.5 Flash"
    config["OUTPUT_CSV"] = re.sub(r'\W+', '_', config["MODEL_NAME"].lower()) + "_determine_retrieve_value.csv"
    config["SLEEP"] = 5  # sleep 2 seconds between runs

    model = load_gemini_flash()
    print("✅ Gemini 2.5 Flash model loaded.")
    prediction(None, model, config)

# === Run GPT-4o Model ===

def run_gpt_4o():
    config["MODEL_NAME"] = "chatgpt_4o"  # use lowercase with underscore
    config["OUTPUT_CSV"] = re.sub(r'\W+', '_', config["MODEL_NAME"].lower()) + "_determine_retrieve_value.csv"

    model = load_chatgpt()  # ✅ get the function, don’t call it yet
    print("📂 Current working directory:", os.getcwd())
    print("📄 Intended output CSV:", config["OUTPUT_CSV"])
    print("✅ GPT-4o model loaded.")
    prediction(None, model, config)  


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide a model to run: chartgemma | gemini_pro | gemini_flash")
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