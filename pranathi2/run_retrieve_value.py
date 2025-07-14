import os, csv
from model import (
    predict_with_gemini_pro,
    predict_with_gemini_flash,
    predict_with_chartgemma,
    configure_gemini
)
from dotenv import load_dotenv
load_dotenv()

# Configuration 🔧
FOLDER = r"C:\Users\prana\OneDrive\Desktop\VL\Visualization_Literacy\Make_comparison\images"
OUTPUT_FILES = {
    "gemini_pro": "gemini_2_5_pro_q1_results.csv",
    "gemini_flash": "gemini_2_5_flash_q1_results.csv",
    "chartgemma": "chartgemma_q1_results.csv"
}
PROMPT = "How much did the price of a barrel of oil fall from April to September in 2015? Options: $4, $15, $17, $45, Omit"
GROUND_TRUTH = "$15"
CHART_TITLE = "Make Comparisons"
QUESTION_TYPE = "Make Comparison"
API_KEY = os.getenv("GEMINI_API_KEY")  

# ............................ Run Prediction for one model ............................
def run_model(model_name):
    print(f" Starting model run: {model_name}")
    if model_name.startswith("gemini"):
        if not API_KEY or API_KEY.strip() == "":
            print(" Missing Gemini API key! Please set GEMINI_API_KEY in your .env file.")
            return
        configure_gemini(API_KEY)

    model_func = {
        "gemini_pro": predict_with_gemini_pro,
        "gemini_flash": predict_with_gemini_flash,
        "chartgemma": predict_with_chartgemma
    }.get(model_name)

    output_path = OUTPUT_FILES[model_name]

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "Image", "Run", "Prompt", "Chart Title", "Question Type",
            "Ground Truth", "Model Name", "Prediction", "Answer Time (s)", "Correct"
        ])

        image_files = sorted([f for f in os.listdir(FOLDER) if f.lower().endswith(".png")])
        print(f" Found {len(image_files)} images in {FOLDER}")
        for image_name in image_files:
            image_path = os.path.join(FOLDER, image_name)
            for run in range(5):
                try:
                    prediction, duration = model_func(image_path, PROMPT)
                    correct = "Correct" if GROUND_TRUTH in prediction else "Incorrect"
                    writer.writerow([
                        image_name, run + 1, PROMPT, CHART_TITLE, QUESTION_TYPE,
                        GROUND_TRUTH, model_name.replace("_", " ").title(), prediction, duration, correct
                    ])
                    print(f" {image_name} | Run {run + 1}: {prediction} ({correct})")
                    if model_name.startswith("gemini"):
                        import time
                        time.sleep(15)
                except Exception as e:
                    writer.writerow([
                        image_name, run + 1, PROMPT, CHART_TITLE, QUESTION_TYPE,
                        GROUND_TRUTH, model_name.replace("_", " ").title(), f"Error: {e}", 0, "Error"])
                    print(f" Error: {e}")

# ............................ Run Models Based on Command Line Arg ............................
if __name__ == "__main__":
    import sys
    #  Custom call like: python run_retrieve_value.py chartgemma
    if len(sys.argv) == 2:
        model = sys.argv[1].lower()
        if model in OUTPUT_FILES:
            print(" Script starting...")
            run_model(model)
            print("\n Model run completed. Results saved.")
        else:
            print(f" Unknown model: {model}. Use: chartgemma, gemini_pro, gemini_flash")
        sys.exit()

    #  Default: run all if no model specified
    print(" No model specified. Running all models by default...")
    for model in ["gemini_pro", "gemini_flash", "chartgemma"]:
        print(f"\n Running predictions with: {model.upper()}")
        run_model(model)
    print("\n All models completed. Results saved.")
