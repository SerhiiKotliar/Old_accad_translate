

# with open(r"G:\Visual Studio 2010\Projects\Python\Old_accad_translate\data\test.csv", "r", encoding="utf-8") as f:
#     text = f.read()
#
# print(text)
#
# with open(r"C:\Users\user\Downloads\submission (5).csv", "r", encoding="utf-8") as f:
#     text = f.read()
#
# print(text)

import pandas as pd

# загрузка файлов
df_main = pd.read_csv("G:/Visual Studio 2010/Projects/Python/Old_accad_translate/data/train.csv")
df_aux = pd.read_csv("G:/Visual Studio 2010/Projects/Python/Old_accad_translate/data/Sentences_Oare_FirstWord_LinNum.csv")
missing_in_main = set(df_aux.text_uuid) - set(df_main.oare_id)
print("Есть в aux, но нет в main:", len(missing_in_main))
missing_in_aux = set(df_main.oare_id) - set(df_aux.text_uuid)
print("Есть в main, но нет в aux:", len(missing_in_aux))
print(df_main.oare_id.head(10))
print(df_aux.text_uuid.head(10))
print(type(df_main.oare_id.iloc[0]))
print(type(df_aux.text_uuid.iloc[0]))
df_main["oare_id"] = df_main["oare_id"].astype(str).str.lower().str.strip()
df_aux["text_uuid"] = df_aux["text_uuid"].astype(str).str.lower().str.strip()

missing_in_main = set(df_aux.text_uuid) - set(df_main.oare_id)
print("Есть в aux, но нет в main:", len(missing_in_main))
missing_in_aux = set(df_main.oare_id) - set(df_aux.text_uuid)
print("Есть в main, но нет в aux:", len(missing_in_aux))
result_rows = []

# группируем вспомогательный файл по тексту
grouped = df_aux.sort_values("first_word_number").groupby("text_uuid")

for _, row in df_main.iterrows():
    text_id = row["oare_id"]
    text = str(row["transliteration"])

    if text_id not in grouped.groups:
        continue

    # слова текста
    words = text.split()

    aux_group = grouped.get_group(text_id).sort_values("first_word_number")

    starts = aux_group["first_word_number"].tolist()
    translations = aux_group["translation"].tolist()
    sent_ids = aux_group["sentence_uuid"].tolist()

    # добавляем конец текста
    starts.append(len(words) + 1)

    for i in range(len(starts) - 1):
        start = starts[i] - 1   # индексация с 0
        end = starts[i + 1] - 1

        sentence_words = words[start:end]
        sentence_text = " ".join(sentence_words)

        result_rows.append({
            "id": sent_ids[i],
            "transliteration": sentence_text,
            "translation": translations[i]
        })

# сохраняем
df_result = pd.DataFrame(result_rows)
df_result.to_csv("G:/Visual Studio 2010/Projects/Python/Old_accad_translate/data/train_add.csv", index=False, encoding="utf-8")