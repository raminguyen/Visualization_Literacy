from transformers import AutoProcessor, AutoModelForVision2Seq
from PIL import Image
import torch, os, time, csv

print(" Starting ChartGemma Debug Script")

# Debug check: Make sure folders and images exist
debug_folders = ["Q1", "Q2"]
for folder in debug_folders:
    print(f" Checking folder: {folder}")
    if not os.path.exists(folder):
        print(f" Folder not found: {folder}")
    else:
        images = [f for f in os.listdir(folder) if f.endswith(".png")]
        print(f" {len(images)} images found in {folder}: {images[:2]}...")

# === Load ChartGemma model ===
model_id = "google/chart-gemma-2b"
device = "cuda" if torch.cuda.is_available() else "cpu"
processor = AutoProcessor.from_pretrained(model_id)
model = AutoModelForVision2Seq.from_pretrained(model_id).to(device)

# === Define folders and corresponding questions ===
folders_with_questions = {
    "Q1": "About how much did the price of a barrel of oil fall from April to September in 2015?",
    "Q2": "What was the price of a barrel of oil in February 2015?"
}
print(" Folders to scan:", list(folders_with_questions.keys()))

# === Output CSV ===
output_csv = "chartgemma_all_results.csv"

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Folder", "Question", "Model", "Prediction", "Answer Time"])

    for folder, question in folders_with_questions.items():
        for image_name in sorted(os.listdir(folder)):
            if not image_name.endswith(".png"):
                continue

            image_path = os.path.join(folder, image_name)

            if not os.path.exists(image_path):
                print(f" Image not found: {image_path}")
                continue

            for run in range(5):
                try:
                    print(f" {folder} | {image_name} | Run {run+1}")
                    image = Image.open(image_path).convert("RGB")
                    inputs = processor(images=image, text=question, return_tensors="pt").to(device)

                    start = time.time()
                    with torch.no_grad():
                        output = model.generate(**inputs, max_new_tokens=50)
                    end = time.time()

                    answer = processor.decode(output[0], skip_special_tokens=True)
                    writer.writerow([image_name, folder, question, "ChartGemma", answer, round(end - start, 2)])

                except Exception as e:
                    writer.writerow([image_name, folder, question, "ChartGemma", f"Error: {e}", 0])
                    print(f" Error on {image_path}: {e}")

print(" Done! Results written to chartgemma_all_results.csv")
