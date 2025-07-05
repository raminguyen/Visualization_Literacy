import os
import time
import pandas as pd
from PIL import Image
import torch

def prediction(processor, model, config):
    # ================================
    # 0. Load prompt, ground truth, and chart title from config
    # ================================
    prompt = list(config["PROMPTS"].values())[0]
    clean_question = prompt.replace("<image>", "").strip()
    chart_title = list(config.get("CHART_TITLES", {}).values())[0] if "CHART_TITLES" in config else "UNKNOWN"
    ground_truth = list(config.get("GROUND_TRUTH", {}).values())[0].get("answer", "UNKNOWN")
    question_type = list(config.get("GROUND_TRUTH", {}).values())[0].get("question_type", "UNKNOWN")
    model_name_lower = config["MODEL_NAME"].lower()

    image_files = sorted([f for f in os.listdir(config["IMAGE_FOLDER"]) if f.endswith(".png")])
    first_row = True

    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(config["IMAGE_FOLDER"], filename)

        # ================================
        # 1. Load image and get color label
        # ================================
        image = Image.open(img_path).convert("RGB")
        color = filename.split("_")[-1].replace(".png", "")

        # ================================
        # 2. Prepare prompt (same for all)
        # ================================
        if "gemini" in config["MODEL_NAME"].lower() and not prompt.strip().startswith("#"):
            prompt += " Choose the accurate answer. No explanation."
        if "gpt_4o" in config["MODEL_NAME"].lower() and not prompt.strip().startswith("#"):
            prompt += " Choose the accurate answer. No explanation."

        # ================================
        # 3. Prediction loop
        # ================================
        for run in range(1, config["RUNS_PER_IMAGE"] + 1):
            print(f"\n🔁 {config['MODEL_NAME']} | Image {i}, Run {run}: {filename}")
            start_time = time.time()

            try:
                # ================================
                # ChartGemma
                # ================================
                if "chartgemma" in model_name_lower:
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=20)
                    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                # ================================
                # ChatGPT (gpt_4o)
                # ================================
                elif "gpt_4o" in model_name_lower or "chatgpt" in model_name_lower:
                    pred = model(prompt, img_path)

                # ================================
                # Gemini models
                # ================================
                elif "gemini" in model_name_lower:
                    response = model.generate_content([prompt, image])
                    pred = response.text.strip()

                # ================================
                # Unknown model fallback
                # ================================
                else:
                    pred = "MODEL_TYPE_NOT_RECOGNIZED"

            except Exception as e:
                pred = f"ERROR: {str(e)}"

            pred_time = round(time.time() - start_time, 2)

            color = os.path.splitext(filename)[0].split("_")[-1]  # e.g., from Q1_img1_Red.png → "Red"

            # ================================
            # 4. Save result to CSV
            # ================================
            pd.DataFrame([{
                "Index": f"img{i}_run{run}",
                "Question Type": question_type,
                "Prompt": prompt,
                "Question Description": clean_question,
                "Model Name": config["MODEL_NAME"],
                "Color": color,
                "Chart Title": chart_title,
                "Image Path": img_path,
                "Ground Truth": ground_truth,
                "Prediction": pred,
                "Answer Time": pred_time,
            }]).to_csv(config["OUTPUT_CSV"], mode="a", index=False, header=first_row)

            first_row = False

            # ================================
            # 5. Print debug info
            # ================================
            print("📄 Writing to CSV:", config["OUTPUT_CSV"])
            print("🖼️ Image:", filename)
            print("📌 Prompt:", prompt)
            print("📌 Prediction:", pred)
            print(f"✅ Prediction saved (Time: {pred_time}s)")
