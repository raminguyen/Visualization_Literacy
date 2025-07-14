
import google.generativeai as genai
import base64, os, time, csv

print("🚀 Starting Gemini 2.5 Pro script...")

# Authenticate (user must replace this with their real API key)
genai.configure(api_key="YOUR_API_KEY_HERE")

# Use Gemini 2.5 Pro Vision
model = genai.GenerativeModel("gemini-1.5-pro-vision")

# Questions
folders_with_questions = {
    "Q1": "About how much did the price of a barrel of oil fall from April to September in 2015?",
    "Q2": "What was the price of a barrel of oil in February 2015?"
}

output_csv = "gemini_pro_all_results.csv"

def encode_image(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode("utf-8")

with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Image", "Folder", "Question", "Model", "Prediction", "Answer Time"])

    for folder, question in folders_with_questions.items():
        if not os.path.exists(folder):
            print(f"❌ Folder not found: {folder}")
            continue

        image_files = sorted([f for f in os.listdir(folder) if f.endswith(".png")])
        print(f"📂 Found {len(image_files)} images in {folder}")

        for image_name in image_files:
            image_path = os.path.join(folder, image_name)

            for run in range(5):
                try:
                    print(f"🧠 Processing {image_name} | Folder: {folder} | Run {run+1}")
                    image_data = {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": encode_image(image_path)
                        }
                    }

                    start = time.time()
                    response = model.generate_content([question, image_data])
                    end = time.time()

                    answer = response.text.strip()
                    writer.writerow([image_name, folder, question, "Gemini 2.5 Pro", answer, round(end - start, 2)])

                except Exception as e:
                    writer.writerow([image_name, folder, question, "Gemini 2.5 Pro", f"Error: {e}", 0])
                    print(f"❌ Error with {image_path}: {e}")

print("✅ Done! Results saved to gemini_pro_all_results.csv")
