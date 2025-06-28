# chartgemma.py

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
