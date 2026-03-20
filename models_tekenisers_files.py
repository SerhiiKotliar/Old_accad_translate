import os
import torch
import shutil
import zipfile

# -----------------------------
# Пути
# -----------------------------
save_path = "/kaggle/working/akkadian_bt5_augmented"  # исходная папка модели и токенизера
download_dir = "/kaggle/working/akkadian_bt5_for_download"
os.makedirs(download_dir, exist_ok=True)

MAX_SIZE = 1 * 1024**3  # 1 ГБ

# -----------------------------
# 1. Сохраняем модель по частям (только zip)
# -----------------------------
model_file = os.path.join(save_path, "pytorch_model.bin")
state_dict = torch.load(model_file, map_location="cpu")

part_num = 0
current_part = {}
current_size = 0

for name, tensor in state_dict.items():
    tensor_size = tensor.element_size() * tensor.numel()
    if current_size + tensor_size > MAX_SIZE:
        # сохраняем и архивируем часть
        part_zip = os.path.join(download_dir, f"model_part_{part_num}.zip")
        with zipfile.ZipFile(part_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
            temp_path = os.path.join(download_dir, f"temp_part_{part_num}.pt")
            torch.save(current_part, temp_path)
            zipf.write(temp_path, arcname=f"model_part_{part_num}.pt")
        os.remove(temp_path)
        print(f"Saved {part_zip}")
        part_num += 1
        current_part = {}
        current_size = 0

    current_part[name] = tensor
    current_size += tensor_size

# последняя часть
if current_part:
    part_zip = os.path.join(download_dir, f"model_part_{part_num}.zip")
    with zipfile.ZipFile(part_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
        temp_path = os.path.join(download_dir, f"temp_part_{part_num}.pt")
        torch.save(current_part, temp_path)
        zipf.write(temp_path, arcname=f"model_part_{part_num}.pt")
    os.remove(temp_path)
    print(f"Saved {part_zip}")

# -----------------------------
# 2. Сохраняем токенизер (один zip)
# -----------------------------
tokenizer_files = ["spiece.model", "tokenizer_config.json", "special_tokens_map.json", "config.json"]

tokenizer_zip = os.path.join(download_dir, "tokenizer.zip")
with zipfile.ZipFile(tokenizer_zip, "w", compression=zipfile.ZIP_DEFLATED) as zipf:
    for f in tokenizer_files:
        zipf.write(os.path.join(save_path, f), arcname=f)
print(f"Tokenizer archived: {tokenizer_zip}")

print(f"Все файлы готовы для скачивания из {download_dir}!")

# восстановление модели и токенизатора из файлов в целое в Кэгл
# -----------------------------------------------------------------------------------
import os
import torch
import zipfile
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# -----------------------------
# Пути
# -----------------------------
input_dir = "/kaggle/input/your_uploaded_folder"  # замените на папку Input с вашими zip
rebuild_dir = "/kaggle/working/akkadian_bt5_rebuilt"
os.makedirs(rebuild_dir, exist_ok=True)

# -----------------------------
# 1. Восстанавливаем модель из частей
# -----------------------------
model_parts = sorted([f for f in os.listdir(input_dir) if f.startswith("model_part_") and f.endswith(".zip")])

full_state_dict = {}

for part_zip in model_parts:
    part_path = os.path.join(input_dir, part_zip)
    with zipfile.ZipFile(part_path, "r") as zipf:
        zipf.extractall(rebuild_dir)
    # в zip внутри есть model_part_*.pt
    pt_files = [f for f in os.listdir(rebuild_dir) if f.startswith("model_part_") and f.endswith(".pt")]
    for pt_file in pt_files:
        state_dict_part = torch.load(os.path.join(rebuild_dir, pt_file), map_location="cpu")
        full_state_dict.update(state_dict_part)
        os.remove(os.path.join(rebuild_dir, pt_file))  # удаляем временный pt

# -----------------------------
# 2. Загружаем конфиг модели
# -----------------------------
config_path = os.path.join(input_dir, "config.json")  # для Seq2SeqLM обычно config.json
# скопируем конфиг в rebuild_dir
rebuild_config_path = os.path.join(rebuild_dir, "config.json")
if not os.path.exists(rebuild_config_path):
    import shutil
    shutil.copy(config_path, rebuild_config_path)

# -----------------------------
# 3. Создаем модель и загружаем state_dict
# -----------------------------
model = AutoModelForSeq2SeqLM.from_pretrained(rebuild_dir, local_files_only=True)
model.load_state_dict(full_state_dict)
model.eval()
print("Модель успешно восстановлена оффлайн!")

# -----------------------------
# 4. Восстанавливаем токенизер
# -----------------------------
tokenizer_zip = os.path.join(input_dir, "tokenizer.zip")
with zipfile.ZipFile(tokenizer_zip, "r") as zipf:
    zipf.extractall(rebuild_dir)

tokenizer = AutoTokenizer.from_pretrained(rebuild_dir, use_fast=False)
print("Токенизер успешно восстановлен оффлайн!")

# -----------------------------
# Теперь можно использовать модель и токенизер
# -----------------------------
# Пример инференса
# inputs = tokenizer("Текст на аккадском", return_tensors="pt")
# outputs = model.generate(**inputs)
# print(tokenizer.batch_decode(outputs, skip_special_tokens=True))