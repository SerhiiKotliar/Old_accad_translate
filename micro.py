import torch
from transformers import AutoModelForSeq2SeqLM

model_name = "google/byt5-small"

model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

state_dict = torch.load("D:/Projects/Python/Конкурсы/Old_accad_translate/models/part_0.pt", map_location="cpu")
model.load_state_dict(state_dict)
save_path = "D:/Projects/Python/Конкурсы/Old_accad_translate/models/restored_model"

model.save_pretrained(save_path)


print("Модель восстановлена")