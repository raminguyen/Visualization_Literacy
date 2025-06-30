import os
import csv
import time
from PIL import Image
import google.generativeai as genai
import pandas as pd

# === Configuration ===
API_KEY=""
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")

# === Paths ===
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, "task2_modified_images")
output_csv = os.path.join(script_dir, "output", "task2_gemini_results.csv")

# === Color Mapping ===
color_mapping = {
    "Q161_img1.png": "red",
    "Q161_img2.png": "navy",
    "Q161_img3.png": "chartreuse",
    "Q161_img4.png": "orange",
    "Q161_img5.png": "medium violet",
    "Q161_img6.png": "saddle brown",
    "Q161_img7.png": "black",
    "Q161_img8.png": "light pink",
    "Q161_img9.png": "light gray",
    "Q161_img10.png": "tomato"
}

# === Prompt ===
prompt = "About how much did the price of a barrel of oil fall from April to September in 2015? Options: $4, $15, $17, $45, Omit. Answer only. No explanation."

# === Determine starting index and write header if needed ===
csv_exists = os.path.exists(output_csv)
is_empty = os.stat(output_csv).st_size == 0 if csv_exists else True

if not csv_exists or is_empty:
    index_start = 1
    with open(output_csv, "w", newline='', encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["index", "image", "run", "color", "response"])
        writer.writeheader()
else:
    try:
        df_existing = pd.read_csv(output_csv)
        index_start = df_existing["index"].max() + 1 if "index" in df_existing.columns else 1
    except Exception as e:
        print(" Warning: Could not read existing CSV properly. Starting from index 1.")
        index_start = 1

# === Load images ===
image_list = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
print(f" Found {len(image_list)} modified images.")
print(f" Writing to: {output_csv}")

# === Run Gemini and append results ===
with open(output_csv, "a", newline='', encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["index", "image", "run", "color", "response"])
    index = index_start

    for image_name in image_list:
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path)
        color = color_mapping.get(image_name, "unknown")

        print(f"\n Running: {image_name} (color: {color})")
        for run in range(1, 6):
            print(f"   Run {run}/5")
            try:
                response = model.generate_content([prompt, image])
                answer = response.text.strip()
            except Exception as e:
                answer = f"Error: {e}"
                print("   Error:", e)

            row = {
                "index": index,
                "image": image_name,
                "run": run,
                "color": color,
                "response": answer
            }
            writer.writerow(row)
            index += 1
            time.sleep(5)

print("\n Task 2 results appended to CSV successfully.")
