import pandas as pd
import random

# -----------------------------
# 1. загрузка исходного корпуса
# -----------------------------
df = pd.read_csv("train_combined.csv")

# очистка
df = df.dropna(subset=["transliteration", "translation"])
df["transliteration"] = df["transliteration"].astype(str)
df["translation"] = df["translation"].astype(str)

# -----------------------------
# 2. функция генерации вариаций транслитерации
# -----------------------------
def augment_transliteration(text, n_variants=3):
    """
    Создает n_variants вариаций аккадской транслитерации.
    Вариации:
      - меняем дефисы на пробелы и наоборот
      - переставляем окончания (например, -um ↔ -u)
      - иногда меняем регистр букв
    """
    variants = set()
    variants.add(text)  # оставляем оригинал

    for _ in range(n_variants):
        t = text

        # 1) дефис <-> пробел
        if "-" in t:
            t = t.replace("-", " ")
        elif " " in t:
            t = t.replace(" ", "-")

        # 2) иногда меняем окончания -um/-u/-a
        t = t.replace("-um", "-u") if random.random() < 0.3 else t
        t = t.replace("-a", "-ā") if random.random() < 0.2 else t

        # 3) случайное переключение регистра
        t = "".join(c.upper() if random.random() < 0.1 else c for c in t)

        variants.add(t)

    return list(variants)

# -----------------------------
# 3. создание расширенного корпуса
# -----------------------------
rows = []
for _, row in df.iterrows():
    translit_variants = augment_transliteration(row["transliteration"], n_variants=3)
    for t_var in translit_variants:
        rows.append({
            "oare_id": len(rows),
            "transliteration": t_var,
            "translation": row["translation"]
        })

df_augmented = pd.DataFrame(rows, columns=["oare_id", "transliteration", "translation"])

# -----------------------------
# 4. сохранение в CSV
# -----------------------------
df_augmented.to_csv("train_augmented.csv", index=False, encoding="utf-8")

print(f"Исходных примеров: {len(df)}, после аугментации: {len(df_augmented)}")