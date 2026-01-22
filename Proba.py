import pandas as pd
import re

# Регулярка строки транслитерации
# Разделители блоков, которые игнорируем
SEPARATOR_RE = re.compile(r'^-+$')

# TRANSLIT_LINE_RE — разрешённые символы для транслитерации
TRANSLIT_LINE_RE = re.compile(
    r"^[A-Za-zŠšḫḪṣṢṭṬʾʿ0-9\-ℵ \[\]\.\!⅀⅁ᲟᲠ–]+$"
)

# Морфемные разделители (дефис или ℵ)
MORPHEME_SEP_RE = re.compile(r"[-ℵ]")

# Стоп-слова для фильтрации английского текста
EN_WORD_RE = re.compile(
    r"\b(if|when|is|going|to|in|at|and|palace|textiles|old|assyrian|procedures)\b",
    re.I
)

def extract_transliteration(text) -> list:
    """
    Извлекает блоки транслитерации из текста.
    Склеивает строки, оканчивающиеся на - или ℵ с последующей.
    Возвращает список блоков.
    """
    # 1️⃣ если список, объединяем
    if isinstance(text, list):
        text = "\n".join(text)

    # 2️⃣ разбиваем по \n
    raw_lines = text.splitlines()

    # 3️⃣ склеиваем строки с оканчивающимися на - или ℵ
    lines = []
    buffer = ""
    for line in raw_lines:
        line = line.rstrip()
        if not line:
            if buffer:
                lines.append(buffer)
                buffer = ""
            continue

        # если буфер пустой, начинаем новый
        if not buffer:
            buffer = line
        else:
            buffer += " " + line  # Добавляем пробел между склеенными строками

        # если строка не оканчивается на - или ℵ → добавляем как полную строку
        if not line.endswith(("-", "ℵ")):
            lines.append(buffer)
            buffer = ""

    # добавляем остаток
    if buffer:
        lines.append(buffer)

    # 4️⃣ формируем блоки транслитерации
    blocks = []
    current = []

    for line in lines:
        # пропускаем старые разделители
        if SEPARATOR_RE.match(line):
            continue

        # фильтруем строки
        if (TRANSLIT_LINE_RE.match(line) and
            MORPHEME_SEP_RE.search(line) and
            not EN_WORD_RE.search(line)):
            current.append(line)
        else:
            if current:
                blocks.append("\n".join(current).strip())
                current = []

    if current:
        blocks.append("\n".join(current).strip())

    return blocks


# Завантаження даних з CSV-файлу
thiscompteca = "D:/Projects/Python/Конкурсы/Old_accad_translate/"
csv_file_path = thiscompteca + '/data/publications.csv'
df_trnl = pd.read_csv(csv_file_path)

# Обработка данных
df_trnl = df_trnl.drop_duplicates()
idx = df_trnl[df_trnl['has_akkadian']].index
df_trnl.loc[idx, df_trnl.columns[2]] = (
    df_trnl.loc[idx, df_trnl.columns[2]]
    .str.replace("\\n", "\n", regex=False)
)

# Извлекаем все блоки транслитерации
all_blocks = []
for i in idx:
    text = df_trnl.at[i, df_trnl.columns[2]]
    blocks = extract_transliteration(text)
    all_blocks.extend(blocks)

# Формируем финальный текст с разделителями
# Между блоками в пределах одного текста - пустая строка
# Между разными текстами - линия из 40 дефисов
separator = "\n" + "-" * 40 + "\n"
result_text = separator.join(all_blocks)

# Выводим результат
print(result_text)

# # Если нужно сохранить в файл
# with open(thiscompteca + '/transliteration_blocks.txt', 'w', encoding='utf-8') as f:
#     f.write(result_text)