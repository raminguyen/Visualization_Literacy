# run_chartgemma_determine_value.py

from chart_gemma import chart_gemma
from predict_tasks import run_determine_value_prediction
from gemini import load_gemini_pro, load_gemini_flash
import sys
import re 

config = {
    # === Model Info ===
    "MODEL_NAME": "ChartGemma-2.92B",
    "MODEL_PATH": "Chart_Gemma_3B_Model",
    "PROCESSOR_PATH": "Chart_Gemma_3B_Processor",

    # === Paths ===
    "GROUND_TRUTH_CSV": "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/Visualization_Literacy/all_dataset/answer.csv",
    "IMAGE_FOLDER": "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/Visualization_Literacy/chart_gemma/test_images",  # change to your image folder
    "OUTPUT_CSV": "chartgemma_determine_value_output.csv",

    # === Settings ===
    "RUNS_PER_IMAGE": 1,

    # === Prompts ===
    "PROMPTS": {
        "Q1": "What was the price of a barrel of oil in February 2015? Options: $57.36, $47.82, $50.24, $39.72, Omit.",
        "Q2": "What was the average price of a pound of coffee beans in September 2013? Options: $4.9, $5.0, $5.1, $5.2, Omit."
    },

    # === Titles ===
    "CHART_TITLES": {
        "Q1": "Oil prices spike between April and June",
        "Q2": "Coffee bean price dropped from 2013 high to 2014 low"
    },

    # === Ground Truth ===
    "GROUND_TRUTH": {
        "Q1": {
            "answer": "$50.24",
            "question_type": "Retrieve Value"
        },
        "Q2": {
            "answer": "$5.1",
            "question_type": "Retrieve Value"
        }
    }
}


# 🔧 Set dynamic OUTPUT_CSV filename based on model name
model_tag = re.sub(r'\W+', '_', config["MODEL_NAME"].lower())
config["OUTPUT_CSV"] = f"{model_tag}_determine_retrieve_value.csv"

def run_chartgemma():
   
    processor, model = chart_gemma(config["MODEL_PATH"], config["PROCESSOR_PATH"])
    print("✅ ChartGemma loaded.")
    run_determine_value_prediction(processor, model, config)

def run_gemini_pro():
    config["MODEL_NAME"] = "Gemini 2.5 Pro"
    config["OUTPUT_CSV"] = re.sub(r'\W+', '_', config["MODEL_NAME"].lower()) + "_determine_retrieve_value.csv"

    model = load_gemini_pro()
    print("✅ Gemini Pro model loaded.")
    run_determine_value_prediction(None, model, config)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("⚠️ Please provide a model to run: chartgemma | gemini_pro | gemini_flash")
        sys.exit(1)

    model_to_run = sys.argv[1].lower()

    if model_to_run == "chart_gemma":
        run_chartgemma()
    elif model_to_run == "gemini_2_5_pro":
        run_gemini_pro()
    else:
        print(f"❌ Unknown model: {model_to_run}")