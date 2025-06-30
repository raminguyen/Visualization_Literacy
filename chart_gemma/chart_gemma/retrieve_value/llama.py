# models/llama_runner.py
import torch
import numpy as np
from PIL import Image
from transformers import AutoProcessor, AutoModelForPreTraining, BitsAndBytesConfig
from predictors.predict_task import run_prediction_llama

def run_llama_3_2(config):
    model_id = "meta-llama/Llama-3.2-11B-Vision-Instruct"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    model = AutoModelForPreTraining.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=bnb_config,
        low_cpu_mem_usage=True,
        cache_dir="/hpcstor6/scratch01/h/huuthanhvy.nguyen001/cache"
    )

    processor = AutoProcessor.from_pretrained(model_id)
    
    run_prediction_llama(model, processor, config)
