import os
import time
import dotenv
import torch
from PIL import Image
import csv
from datetime import datetime

from transformers import AutoProcessor, AutoModelForImageTextToText
import google.generativeai as genai

# =============================
# Load API Keys
# =============================
dotenv.load_dotenv('.env')
API_KEYS = [os.getenv("api_key"), os.getenv("api_key2")]

# =============================
# Load Config
# =============================
config = {
    "MODEL_NAME": {
        "chartgemma": "ChartGemma-2.92B",
        "gemini_2_5_pro": "gemini-2.5-pro",
        "gemini_flash": "gemini-2.5-flash"
    },
    "MODEL_PATH": "Chart_Gemma_3B_Model",
    "PROCESSOR_PATH": "Chart_Gemma_3B_Processor",
    "GROUND_TRUTH": "$37.04 - $60.95",
    "PROMPT":" What was the price range of a barrel of oil in 2015? Options: $35 - $65, $48.36 - $60.95, $37.04 - $48.36, $37.04 - $60.95, Omit."
,
    "QUESTION_TYPE": "Determine Range",
    "QUESTION_DESC": "What was the price range of a barrel of oil in 2015?",
    "IMAGE_PATHS": [
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_black.png"
    ],
    "CHART_TITLE": "Monthly Oil Price History in 2015",
    "RUNS": 5,
    "DELAY_SECONDS": 5
}


# =============================
# ChartGemma Prediction
# =============================
def run_chartgemma():
    processor = AutoProcessor.from_pretrained(config["PROCESSOR_PATH"])
    model = AutoModelForImageTextToText.from_pretrained(config["MODEL_PATH"]).to("cuda" if torch.cuda.is_available() else "cpu")

    output_file = "chartgemma_determine_range.csv"
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Image", "Question Type", "Prompt", "Question Description", "Model Name", "Color",
                         "Chart Title", "Image Path", "Ground Truth", "Prediction", "Answer Time"])

        for img_path in config["IMAGE_PATHS"]:
            image = Image.open(img_path).convert("RGB")
            color = img_path.split("_")[-1].replace(".png", "")
            filename = os.path.basename(img_path)

            for i in range(config["RUNS"]):
                start = time.time()
                inputs = processor(images=image, text= config["PROMPT"], return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=50)
                prediction = processor.batch_decode(outputs, skip_special_tokens=True)[0]
                duration = round(time.time() - start, 2)

                writer.writerow([
                    f"{filename}_run{i+1}", filename, config["QUESTION_TYPE"], config["PROMPT"],
                    config["QUESTION_DESC"], config["MODEL_NAME"]["chartgemma"], color,
                    config["CHART_TITLE"], img_path, config["GROUND_TRUTH"], prediction, duration
                ])

# =============================
# Gemini Prediction
# =============================
def gemini_generate(prompt, image_path, key, model_version):
    genai.configure(api_key=key)
    model = genai.GenerativeModel(model_version)

    try:
        image = Image.open(image_path).convert("RGB")
        response = model.generate_content([prompt, image])
        return response.text.strip()
    except Exception as e:
        return f"Error: {e}"

def run_gemini(model_type):
    output_file = f"{model_type}_determine_range.csv"
    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Image", "Question Type", "Prompt", "Question Description", "Model Name", "Color",
                         "Chart Title", "Image Path", "Ground Truth", "Prediction", "Answer Time"])

        model_version = config["MODEL_NAME"][model_type]

        for img_path in config["IMAGE_PATHS"]:
            color = img_path.split("_")[-1].replace(".png", "")
            filename = os.path.basename(img_path)

            for i in range(config["RUNS"]):
                key = API_KEYS[i % len(API_KEYS)]
                start = time.time()
                prediction = gemini_generate(config["PROMPT"], img_path, key, model_version)
                duration = round(time.time() - start, 2)
                time.sleep(config["DELAY_SECONDS"])

                writer.writerow([
                    f"{filename}_run{i+1}", filename, config["QUESTION_TYPE"], config["PROMPT"],
                    config["QUESTION_DESC"], model_version, color,
                    config["CHART_TITLE"], img_path, config["GROUND_TRUTH"], prediction, duration
                ])

def run_gemini_pro():
    run_gemini("gemini_2_5_pro")

def run_gemini_flash():
    run_gemini("gemini_flash")
