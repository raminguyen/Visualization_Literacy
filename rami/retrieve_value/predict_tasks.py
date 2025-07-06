import os
import time
import pandas as pd
from PIL import Image
import torch

def prediction(processor, model, config):
    model_name_lower = config["MODEL_NAME"].lower()
    image_files = sorted([f for f in os.listdir(config["IMAGE_FOLDER"]) if f.endswith(".png")])
    first_row = True

    # Initialize counters outside the loop
    total_rows = 0
    model_counts = {}
    color_counts = {}

    for i, filename in enumerate(image_files, 1):
        img_path = os.path.join(config["IMAGE_FOLDER"], filename)
        image = Image.open(img_path).convert("RGB")
        color = filename.split("_")[-1].replace(".png", "")

        # === Reset prompt per image ===
        base_prompt = list(config["PROMPTS"].values())[0]
        prompt = base_prompt.strip()
        clean_question = base_prompt.replace("<image>", "").strip()
        chart_title = list(config.get("CHART_TITLES", {}).values())[0] if "CHART_TITLES" in config else "UNKNOWN"
        ground_truth = list(config.get("GROUND_TRUTH", {}).values())[0].get("answer", "UNKNOWN")
        question_type = list(config.get("GROUND_TRUTH", {}).values())[0].get("question_type", "UNKNOWN")

        if "gemini" in model_name_lower and not prompt.startswith("#"):
            prompt += " Choose the accurate answer. No explanation."
        if "gpt_4o" in model_name_lower and not prompt.startswith("#"):
            prompt += " Choose the accurate answer. No explanation."

        # === Prediction loop ===
        for run in range(1, config["RUNS_PER_IMAGE"] + 1):
            print(f"\n🔁 {config['MODEL_NAME']} | Image {i}, Run {run}: {filename}")
            start_time = time.time()

            try:
                if "chartgemma" in model_name_lower:
                    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)
                    with torch.no_grad():
                        generated_ids = model.generate(**inputs, max_new_tokens=20)
                    pred = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

                elif "gpt_4o" in model_name_lower or "chatgpt" in model_name_lower:
                    pred = model(prompt, img_path)

                elif "gemini" in model_name_lower:
                    response = model.generate_content([prompt, image])
                    pred = response.text.strip()

                else:
                    pred = "MODEL_TYPE_NOT_RECOGNIZED"

            except Exception as e:
                pred = f"ERROR: {str(e)}"

            pred_time = round(time.time() - start_time, 2)

            # === Save to CSV ===
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
            time.sleep(config.get("SLEEP", 5))

            # === Track totals ===
            total_rows += 1
            model_counts[config["MODEL_NAME"]] = model_counts.get(config["MODEL_NAME"], 0) + 1
            color_counts[color] = color_counts.get(color, 0) + 1

    # ================================
    # 6. Final summary saved to CSV
    # ================================

    # Prepare summary DataFrame
    summary_data = {
        "Summary Type": ["Total Predictions"] + [f"Color: {c}" for c in color_counts],
        "Model Name": [config["MODEL_NAME"]] + ["" for _ in color_counts],
        "Count": [total_rows] + [color_counts[c] for c in color_counts]
    }
    summary_df = pd.DataFrame(summary_data)

    # Add empty row for spacing
    empty_row = pd.DataFrame([["", "", ""]], columns=summary_df.columns)

    # Append empty line + summary block to same CSV file
    with open(config["OUTPUT_CSV"], "a") as f:
        f.write("\n")  # Blank line separator
    summary_df.to_csv(config["OUTPUT_CSV"], mode="a", index=False)

    print(f"📄 Summary block saved at the bottom of {config['OUTPUT_CSV']}")
