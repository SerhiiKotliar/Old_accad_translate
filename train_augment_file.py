import pandas as pd
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq
)

# -----------------------------
# 1. загрузка аугментированного корпуса
# -----------------------------
df = pd.read_csv("train_augmented.csv")

df = df.dropna(subset=["transliteration", "translation"])
df["transliteration"] = df["transliteration"].astype(str)
df["translation"] = df["translation"].astype(str)

# создаем колонки для модели
df["input_text"] = "translate Akkadian to English: " + df["transliteration"]
df["target_text"] = df["translation"]

dataset = Dataset.from_pandas(df[["input_text", "target_text"]])
dataset = dataset.train_test_split(test_size=0.1)

train_dataset = dataset["train"]
val_dataset = dataset["test"]

# -----------------------------
# 2. загрузка модели и токенизатора
# -----------------------------
model_name = "google/mt5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# -----------------------------
# 3. токенизация
# -----------------------------
max_input_length = 128
max_target_length = 128

def preprocess(examples):
    model_inputs = tokenizer(
        examples["input_text"],
        max_length=max_input_length,
        truncation=True,
        padding="max_length"
    )

    labels = tokenizer(
        text_target=examples["target_text"],
        max_length=max_target_length,
        truncation=True,
        padding="max_length"
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)

# -----------------------------
# 4. DataCollator
# -----------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# -----------------------------
# 5. параметры обучения
# -----------------------------
training_args = Seq2SeqTrainingArguments(
    output_dir="./akkadian_mt5_aug",
    learning_rate=3e-5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=8,
    num_train_epochs=5,
    weight_decay=0.01,
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_steps=500,
    predict_with_generate=True,
    save_total_limit=2,
)

# -----------------------------
# 6. Trainer
# -----------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator
)

# -----------------------------
# 7. обучение
# -----------------------------
trainer.train()

# -----------------------------
# 8. сохранение модели
# -----------------------------
trainer.save_model("akkadian_mt5_model_aug")
tokenizer.save_pretrained("akkadian_mt5_model_aug")