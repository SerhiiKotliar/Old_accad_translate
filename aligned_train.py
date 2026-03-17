import pandas as pd
import numpy as np
import re

# ======================
# 1. Загрузка данных
# ======================
# thiscompteca = "D:/Projects/Python/Конкурсы/Old_accad_translate"
thiscompteca = "G:/Visual Studio 2010/Projects/Python/Old_accad_translate"

# df_main = pd.read_csv("G:/Visual Studio 2010/Projects/Python/Old_accad_translate/data/train.csv")
df_main = pd.read_csv(thiscompteca+'/data/train.csv')
df_aux = pd.read_csv(thiscompteca+'/data/Sentences_Oare_FirstWord_LinNum.csv')
# df_aux = pd.read_csv("G:/Visual Studio 2010/Projects/Python/Old_accad_translate/data/Sentences_Oare_FirstWord_LinNum.csv")

# ======================
# 2. Очистка данных
# ======================
def normalize(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)  # убираем пунктуацию
    return text

df_main["translation_norm"] = df_main["translation"].apply(normalize)
df_aux["translation_norm"] = df_aux["translation"].apply(normalize)

# ======================
# 3. Группировка aux по текстам
# ======================
grouped_aux = df_aux.groupby("text_uuid")

result = []

# ======================
# 4. Основной цикл (ВАЖНО: идём от aux!)
# ======================
for text_uuid, group in grouped_aux:

    aux_sentences = group["translation_norm"].tolist()

    best_match = None
    best_score = 0

    # ищем лучший текст из train
    for _, main_row in df_main.iterrows():
        full_text = main_row["translation_norm"]

        score = sum(1 for sent in aux_sentences if sent and sent in full_text)

        if score > best_score:
            best_score = score
            best_match = main_row

    # фильтр — отсекаем плохие совпадения
    if best_match is None or best_score < 2:
        continue

    # ======================
    # 5. Разбиение транслитерации
    # ======================
    words = str(best_match["transliteration"]).split()

    group = group.sort_values("first_word_number")
    starts = group["first_word_number"].tolist()
    starts.append(len(words) + 1)

    for i, (_, row) in enumerate(group.iterrows()):
        start = starts[i] - 1
        end = starts[i + 1] - 1

        sentence_words = words[start:end]
        sentence_text = " ".join(sentence_words).strip()

        result.append({
            "oare_id": row["sentence_uuid"],
            "transliteration": sentence_text,
            "translation": row["translation"]
        })

# ======================
# 6. В DataFrame
# ======================
df_result = pd.DataFrame(result)

# ======================
# 7. Удаление мусора
# ======================
df_result = df_result.replace(r'^\s*$', np.nan, regex=True)

df_result = df_result.dropna(subset=[
    "oare_id",
    "transliteration",
    "translation"
])

# ======================
# 8. Удаление дублей (на всякий случай)
# ======================
df_result = df_result.drop_duplicates(subset=["oare_id"])

# ======================
# 9. Сохранение
# ======================
df_result.to_csv("aligned_sentences.csv", index=False, encoding="utf-8")

# ======================
# 10. Контроль
# ======================
print("Итог строк:", len(df_result))
print("Уникальных предложений:", df_result["oare_id"].nunique())