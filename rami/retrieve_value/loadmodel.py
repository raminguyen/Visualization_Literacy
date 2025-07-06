#Step_1: Save Model Weight##

from transformers import AutoProcessor, AutoModelForImageTextToText

model = AutoModelForImageTextToText.from_pretrained("ahmed-masry/chartgemma")
processor = AutoProcessor.from_pretrained("ahmed-masry/chartgemma")

model.save_pretrained("Chart_Gemma_3B_Model")
processor.save_pretrained("Chart_Gemma_3B_Processor")

print ('It is done. Yeah!')