import os
import csv
import time 
from PIL import Image
from PIL import ImageOps
import google.generativeai as genai
import pandas as pd

API_KEY = ""

if not API_KEY or API_KEY.strip() == "":
    print(" ERROR: Gemini API Key not found.")
    exit()

print(" SCRIPT IS RUNNING...")

# Setup Gemini
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("models/gemini-2.5-flash")


# Input folder
script_dir = os.path.dirname(os.path.abspath(__file__))
image_folder = os.path.join(script_dir, "task1_original_images")


if not os.path.exists(image_folder):
    print(" ERROR: Folder not found:", image_folder)
    exit()

image_list = sorted([f for f in os.listdir(image_folder) if f.endswith(".png")])
print(f" Found {len(image_list)} images.")

# Output setup
output_folder = "pranathi/output"
os.makedirs(output_folder, exist_ok=True)
output_csv = os.path.join(output_folder, "task1_gemini_results.csv")

prompt = "About how much did the price of a barrel of oil fall from April to September in 2015? Options: $4, $15, $17, $45, Omit. Answer only. No explanation."

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

rows = []  #  Store rows here to print later

# Run the queries and save results
with open(output_csv, "w", newline='', encoding="utf-8") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["index", "image", "run", "color", "response"])
    writer.writeheader()

    index = 1
    for image_name in image_list:
        image_path = os.path.join(image_folder, image_name)
        image = Image.open(image_path)
        cropped_image = ImageOps.crop(image, (0, 50, 0, 0))  # (left, top, right, bottom)
        color = color_mapping.get(image_name, "unknown")

        print(f"\n Running: {image_name}")
        for run in range(1, 6):
            print(f"    Run {run}/5")
            try:
                response = model.generate_content([prompt, cropped_image])
                answer = response.text.strip()
                valid_answers = {"$15", "15", "$4", "4", "$17", "17", "$45", "45", "Omit", "omit"}
                if answer not in valid_answers:
                    answer = "Omit"
            except Exception as e:
                answer = f"Error: {e}"
                print(" Error:", e)

            row = {
                "index": index,
                "image": image_name,
                "run": run,
                "color": color,
                "response": answer
            }
            writer.writerow(row)
            rows.append(row)
            index += 1
            time.sleep(5) # one run takes 4 sec for gemini so i take 5 second break 

print(f"\n FINISHED. Results saved at: {output_csv}")

# Display results as a table
print("\nResult Table Preview:")
df = pd.DataFrame(rows)
print(df.to_string(index=False))

