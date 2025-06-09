import os
import base64
import requests
from dotenv import load_dotenv
import csv

# Load API key from .env file
load_dotenv()

API_KEY = os.getenv("chatgpt_api_key")
assert API_KEY is not None, "❌ API key not found in .env"

class ChatGPTModel:
    def __init__(self, model_name="gpt-4o"):
        self.model_name = model_name
        self.api_key = API_KEY
        self.endpoint = "https://api.openai.com/v1/chat/completions"

    def _encode_image_base64(self, image_path):
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode("utf-8")

    def _build_payload(self, question, image_b64):
        return {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}}
                    ]
                }
            ],
            "max_tokens": 400
        }

    def query(self, question, image_path):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        image_b64 = self._encode_image_base64(image_path)
        payload = self._build_payload(question, image_b64)

        response = requests.post(self.endpoint, headers=headers, json=payload)
        result = response.json()

        if "error" in result:
            print("❌ Error:", result["error"]["message"])
            return None

        return result["choices"][0]["message"]["content"]

    import csv

    def query_multiple_times(self, question, image_path, times=3):
        results = []
        for i in range(times):
            print(f"\n🧠 Run {i+1}/{times}")
            answer = self.query(question, image_path)
            if answer:
                print(answer)
                results.append({'run': i + 1, 'answer': answer})

        # Save all to a single CSV file
        with open("linechart_chatgpt_answers.csv", "w", newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=["run", "answer"])
            writer.writeheader()
            writer.writerows(results)

        return results


