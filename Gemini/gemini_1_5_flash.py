# gemini_vision.py

import google.generativeai as genai
from PIL import Image
from config import API_KEY

class GeminiVisionRunner:
    def __init__(self, model_name="gemini-2.5-pro-preview-06-05", image_path=None, prompt=None, runs=3, output_file="gemini_output.txt"):
        self.api_key = API_KEY
        self.model_name = model_name
        self.image_path = image_path
        self.prompt = prompt
        self.runs = runs
        self.output_file = output_file

        self._configure()
        self.model = genai.GenerativeModel(self.model_name)

    def _configure(self):
        genai.configure(api_key=self.api_key)

    def run(self):
        if not self.image_path or not self.prompt:
            raise ValueError("Image path and prompt must be provided.")

        image = Image.open(self.image_path)

        with open(self.output_file, "w", encoding="utf-8") as f:
            for i in range(1, self.runs + 1):
                print(f"🔁 Run #{i}")
                response = self.model.generate_content([self.prompt, image])
                result = response.text.strip()
                print("🧠 Gemini's Answer:", result)
                print("-" * 50)

                # Save to file
                f.write(f"🔁 Run #{i}\n")
                f.write(f"{result}\n")
                f.write("-" * 50 + "\n")
