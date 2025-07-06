# run_find_correlations.py

import sys
from predict_determine_range import run_chartgemma, run_gemini_pro, run_gemini_flash, run_chatgpt_vision


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a model to run: chartgemma | gemini_pro | gemini_flash | chatgpt")
        sys.exit(1)

    model_to_run = sys.argv[1].lower()

    if model_to_run == "chartgemma":
        run_chartgemma()
    elif model_to_run == "gemini_pro":
        run_gemini_pro()
    elif model_to_run == "gemini_flash":
        run_gemini_flash()
    elif model_to_run == "chatgpt":
        run_chatgpt_vision()
    else:
        print(f"Unknown model: {model_to_run}")
