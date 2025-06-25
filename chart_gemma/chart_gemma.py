# #Step_1: Save Model Weight

# from transformers import AutoProcessor, AutoModelForImageTextToText

# model = AutoModelForImageTextToText.from_pretrained("ahmed-masry/chartgemma")
# processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

# model.save_pretrained("Chart_Gemma_3B_Model")
# processor.save_pretrained("Chart_Gemma_3B_Processor")

# print ('It is done. Yeah!')


""" Step 2: Load Chart_Gemma_2.92B Model"""

from transformers import AutoProcessor, AutoModelForImageTextToText

from generalfunction import *

#Load the processor and model from saved folders
processor = AutoProcessor.from_pretrained("Chart_Gemma_3B_Processor")

model = AutoModelForImageTextToText.from_pretrained("Chart_Gemma_3B_Model")

print("Model and processor loaded successfully.")

"""Step 3: Run one example of line chart"""

from generalfunction import *
import pandas as pd
import torch


image_path = "/Users/ramihuunguyen/Documents/PhD/Visualization_Literacy/line_chart_dataset/Make_Comparisons_Images.png"

# Load the image once
image = Image.open(image_path).convert("RGB")

# Add <image> token to the question
question = "<image> Look at each month carefully. About how much did the price of a barrel of oil fall from April to September in 2015? Options: $4, $15, $17, $45, Omit. Answer only. No explanation."

# Run 3 times
for i in range(3):
    inputs = processor(images=image, text=question, return_tensors="pt").to(model.device)

    with torch.no_grad():
        generated_ids = model.generate(**inputs, max_new_tokens=20)

    answer = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    print(f"Run {i+1} answer:", answer)