import os
import time
import dotenv
import torch
import base64
from PIL import Image
import csv

from transformers import AutoProcessor, AutoModelForImageTextToText
import google.generativeai as genai
from openai import OpenAI

# =============================
# Load API Keys
# =============================
dotenv.load_dotenv('.env')
API_KEYS = [os.getenv("api_key"), os.getenv("api_key2")]
CHATGPT_API_KEY = os.getenv("api_key3")

# =============================
# Load Config
# =============================
config = {
    "MODEL_NAME": {
        "chartgemma": "ChartGemma-2.92B",
        "gemini_2_5_pro": "gemini-2.5-pro",
        "gemini_flash": "gemini-2.5-flash",
        "chatgpt": "gpt-4o"
    },
    "MODEL_PATH": "Chart_Gemma_3B_Model",
    "PROCESSOR_PATH": "Chart_Gemma_3B_Processor",
    "GROUND_TRUTH": "$37.04 - $60.95",
    "PROMPT": "Over the course of the second half of 2015, the price of a barrel of oil was __. Options: rising, falling, staying, Omit. Choose only one word as the answer without explanation.",
    "QUESTION_TYPE": "Find Correlations/Trends",
    "QUESTION_DESC": "Over the course of the second half of 2015, the price of a barrel of oil was __.",
    "IMAGE_PATHS": [
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_black.png",
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_burlywood.png",
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_chartreuse.png",
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_lightgray.png",
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_lightpink.png",
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_mediumvioletred.png",
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_navy.png",
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_red.png",
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_saddlebrown.png",
        r"C:\Users\kanam\OneDrive\Desktop\Visualization_Literacy\Visualization_Literacy\Avanith\Test\line_data.csv_line_Monthly_Oil_Price_History_in_2015_tomato.png"
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

    output_file = "Final_chartgemma__Find_Correlations.csv"
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
                inputs = processor(images=image, text=config["PROMPT"], return_tensors="pt").to(model.device)
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
    output_file = f"Final_{model_type}_Find_Correlations.csv"
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

# =============================
# ChatGPT Vision Prediction
# =============================
def run_chatgpt_vision():
    import base64
    from openai import OpenAI

    output_file = "Final_chatgpt_Find_Correlations.csv"
    client = OpenAI(api_key=CHATGPT_API_KEY)

    with open(output_file, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Image", "Question Type", "Prompt", "Question Description", "Model Name", "Color",
                         "Chart Title", "Image Path", "Ground Truth", "Prediction", "Answer Time"])

        for img_path in config["IMAGE_PATHS"]:
            color = img_path.split("_")[-1].replace(".png", "")
            filename = os.path.basename(img_path)

            with open(img_path, "rb") as image_file:
                b64_image = base64.b64encode(image_file.read()).decode("utf-8")

            for i in range(config["RUNS"]):
                start = time.time()
                try:
                    response = client.chat.completions.create(
                        model=config["MODEL_NAME"]["chatgpt"],
                        messages=[
                            {
                                "role": "user",
                                "content": [
                                    {"type": "text", "text": config["PROMPT"]},
                                    {
                                        "type": "image_url",
                                        "image_url": {
                                            "url": f"data:image/png;base64,{b64_image}"
                                        }
                                    }
                                ]
                            }
                        ],
                        max_tokens=10
                    )
                    prediction = response.choices[0].message.content.strip()
                except Exception as e:
                    prediction = f"Error: {e}"

                duration = round(time.time() - start, 2)
                time.sleep(config["DELAY_SECONDS"])
#this script is needed cause this is what is saving the data in csv
                writer.writerow([
                    f"{filename}_run{i+1}", filename, config["QUESTION_TYPE"], config["PROMPT"],
                    config["QUESTION_DESC"], config["MODEL_NAME"]["chatgpt"], color,
                    config["CHART_TITLE"], img_path, config["GROUND_TRUTH"], prediction, duration
                ])
