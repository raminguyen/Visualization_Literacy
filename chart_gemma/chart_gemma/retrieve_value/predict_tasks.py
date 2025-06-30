import os
import time
import pandas as pd
from PIL import Image
import torch

def run_determine_value_prediction(processor, model, config):
    PROMPTS = config["PROMPTS"]
    ans_df = pd.read_csv(config["GROUND_TRUTH_CSV"])
    image_files = sorted([f for f in os.listdir(config["IMAGE_FOLDER"]) if f.endswith(".png")])
    first_row = True

    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(config["IMAGE_FOLDER"], filename)

        # ================================
        #1. Load image and get color label
        # ================================
        image = Image.open(img_path).convert("RGB")
        color = filename.split("_")[-1].replace(".png", "")

        # ================================
        #2. Get prompt key (Q1, Q2, etc.)
        # ================================
        prompt_key = next((k for k in PROMPTS if filename.startswith(k)), None)
        if not prompt_key:
            print(f"⚠️ Skipping: {filename} (no Q1/Q2 match)")
            continue

        # ================================
        # 3. Prepare prompt
        # ================================
        base_prompt = PROMPTS[prompt_key]
        prompt = base_prompt

        if "gemini" in config["MODEL_NAME"].lower() and not prompt.strip().startswith("#"):
            prompt += " Choose the accurate answer. No explanation."

        clean_question = prompt.replace("<image>", "").strip()

        # ================================
        # 4. Get ground truth and type from config
        # ================================
        if prompt_key in config.get("GROUND_TRUTH", {}):
            ground_truth = config["GROUND_TRUTH"][prompt_key]["answer"]
            question_type = config["GROUND_TRUTH"][prompt_key]["question_type"]
        else:
            ground_truth = "UNKNOWN"
            question_type = "UNKNOWN"

        # ================================
        #5. Prediction loop
        # ================================
        for run in range(1, config["RUNS_PER_IMAGE"] + 1):
            print(f"\n🔁 {config['MODEL_NAME']} | Image {i}, Run {run}: {filename}")
            start_time = time.time()

            # ================================
            # ChartGemma or similar models
            # ================================

            if processor is not None:
                inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    generated_ids = model.generate(**inputs, max_new_tokens=20)
                pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

            # ================================
            # Gemini models
            # ================================
            else:
                try:
                    response = model.generate_content([prompt, image])
                    pred = response.text.strip()
                except Exception as e:
                    pred = f"ERROR: {str(e)}"

            pred_time = round(time.time() - start_time, 2)

            # ================================
            # Save result
            # ================================
            pd.DataFrame([{
                "Index": f"img{i}_run{run}",
                "Question Type": question_type,
                "Prompt": prompt,
                "Question Description": clean_question,
                "Model Name": config["MODEL_NAME"],
                "Color": color,
                "Chart Title": config.get("CHART_TITLES", {}).get(prompt_key, "UNKNOWN"),
                "Image Path": img_path,
                "Ground Truth": ground_truth,
                "Prediction": pred,
                "Answer Time": pred_time,
            }]).to_csv(config["OUTPUT_CSV"], mode="a", index=False, header=first_row)

            first_row = False
