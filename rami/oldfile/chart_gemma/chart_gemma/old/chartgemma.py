# #Step_1: Save Model Weight

# from transformers import AutoProcessor, AutoModelForImageTextToText

# model = AutoModelForImageTextToText.from_pretrained("ahmed-masry/chartgemma")
# processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

# model.save_pretrained("Chart_Gemma_3B_Model")
# processor.save_pretrained("Chart_Gemma_3B_Processor")

# print ('It is done. Yeah!')

# === chartgemma_loader.py ===

from transformers import AutoProcessor, AutoModelForImageTextToText

def chart_gemma(model_path: str, processor_path: str):
    """
    Load ChartGemma model and processor.

    Args:
        model_path (str): Path or name of the pretrained model.
        processor_path (str): Path or name of the processor.

    Returns:
        processor, model: Loaded HuggingFace processor and model.
    """

    processor = AutoProcessor.from_pretrained(processor_path)

    model = AutoModelForImageTextToText.from_pretrained(model_path)

    print("✅ Model and processor loaded.")

    return processor, model
