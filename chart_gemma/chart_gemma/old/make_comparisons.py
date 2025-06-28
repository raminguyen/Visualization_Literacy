# === Step 1: CONFIGURATION SECTION ===
MODEL_NAME = "ChartGemma-2.92B"
MODEL_PATH = "Chart_Gemma_3B_Model"
PROCESSOR_PATH = "Chart_Gemma_3B_Processor"
GROUND_TRUTH_CSV = "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/Visualization_Literacy/all_dataset/ans.csv"
IMAGE_FOLDER = "/hpcstor6/scratch01/h/huuthanhvy.nguyen001/Visualization_Literacy/line_chart_dataset/grouped_by_type/Make_Comparisons"
OUTPUT_CSV = "chartgemma_Make_Comparisons_output.csv"  # ✅ Image folder-based name

PROMPT = """
<image> Look at each month carefully. About how much did the price of a barrel of oil fall from April to September in 2015? 
Which option is correct $4, $15, $17, $45, Omit?"
"""

RUNS_PER_IMAGE = 5  # ✅ Always run 5 times per image

# === Step 2: IMPORTS ===
import os
import time
import torch
import pandas as pd
from PIL import Image
import difflib
from transformers import AutoProcessor, AutoModelForImageTextToText
from chartgemma import chart_gemma

# === Step 3: Load Model & Processor ===
processor, model = chart_gemma(MODEL_PATH, PROCESSOR_PATH)
print("✅ Model and processor loaded.")

# === Step 4: Load Ground Truth CSV & Clean Question ===
ans_df = pd.read_csv(GROUND_TRUTH_CSV)
clean_question = PROMPT.replace("<image>", "").strip()

# === Step 5: Fuzzy Match to Find Ground Truth Answer ===
question_list = ans_df['question'].astype(str).tolist()
best_match = difflib.get_close_matches(clean_question, question_list, n=1, cutoff=0.5)

if best_match:
    match_row = ans_df[ans_df['question'] == best_match[0]].iloc[0]
    ground_truth = match_row['answer']
    question_type = match_row['question_type']
else:
    ground_truth = "UNKNOWN"
    question_type = "UNKNOWN"
    print("⚠️ Warning: Question not found via fuzzy matching.")

# === Step 6: Get Image Files (no filtering prefix here) ===
image_files = sorted([
    f for f in os.listdir(IMAGE_FOLDER)
    if f.endswith(".png")
])

# === Step 7: Run Inference and Save Output ===
first_row = True

for i, filename in enumerate(image_files, start=1):
    image_path = os.path.join(IMAGE_FOLDER, filename)
    color = filename.split("_")[-1].replace(".png", "")
    image = Image.open(image_path).convert("RGB")

    for run in range(1, RUNS_PER_IMAGE + 1):
        print(f"\n🚀 Run {run} on Image {i}/{len(image_files)}: {filename}")

        start_time = time.time()
        inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(model.device)

        with torch.no_grad():
            generated_ids = model.generate(**inputs, max_new_tokens=100)

        prediction_time = round(time.time() - start_time, 2)
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        print(f"✅ Prediction: {prediction} (⏱️ {prediction_time}s)")

        row = {
            "Index": f"img{i}_run{run}",
            "Question Type": question_type,
            "Question Description": clean_question,
            "Prompt": PROMPT,
            "Model Name": MODEL_NAME,
            "Chart Title": "Monthly Oil Prices in 2015",
            "Color": color,
            "Image Path": image_path,
            "Ground Truth": ground_truth,
            "Prediction": prediction,
            "Answer Time": prediction_time
        }

        pd.DataFrame([row]).to_csv(OUTPUT_CSV, mode='a', index=False, header=first_row)
        first_row = False
