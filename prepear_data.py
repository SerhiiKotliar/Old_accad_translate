#%%
from unittest import case

import pandas as pd
# import numpy as np
import re
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv1D
# from keras.utils import to_categorical
# from sklearn.metrics import accuracy_score
# import shutil
#%%
DETERMINATIVE_MAP = {
    # боги
    "ᴰ": "{d}",

    # звёзды
    "ᴹᵁᴸ": "{mul}",

    # земля / место
    "ᴷᴵ": "{ki}",

    # человек
    "ᴸᵁ₂": "{lu₂}",   # если lu₂ записано с надстрочной ₂
    "ᴸᵁ": "{lu₂}",    # частый OCR-вариант без ₂

    # здания
    "ᴱ₂": "{e₂}",
    "ᴱ": "{e₂}",

    # населённые пункты
    "ᵁᴿᵁ": "{uru}",

    # страны / горы
    "ᴷᵁᴿ": "{kur}",

    # женский род
    "ᴹᴵ": "{mi}",

    # мужской род
    "ᴹ": "{m}",

    # дерево / деревянное
    "ᴳᴵŠ": "{geš}",
    "ᴳᴵŠ": "{ĝeš}",   # при необходимости нормализации

    # ткани
    "ᵀᵁᴳ₂": "{tug₂}",
    "ᵀᵁᴳ": "{tug₂}",

    # таблички
    "ᴰᵁᴮ": "{dub}",

    # река / канал
    "ᴵᴰ₂": "{id₂}",
    "ᴵᴰ": "{id₂}",

    # птицы
    "ᴹᵁŠᴱᴺ": "{mušen}",
    "ᴹᵁŠ": "{mušen}",

    # камень
    "ᴺᴬ₄": "{na₄}",
    "ᴺᴬ": "{na₄}",

    # кожа
    "ᴷᵁŠ": "{kuš}",

    # растения
    "ᵁ₂": "{u₂}",
    "ᵁ": "{u₂}",
}
SUBSCRIPT_DIGITS = str.maketrans({
    "₀": "0",
    "₁": "1",
    "₂": "2",
    "₃": "3",
    "₄": "4",
    "₅": "5",
    "₆": "6",
    "₇": "7",
    "₈": "8",
    "₉": "9",
})

# --- ASCII → Unicode (фонетическая нормализация)
CHAR_MAP = {
    's"': 'š', 'S"': 'Š',
    's,': 'ṣ', 'S,': 'Ṣ',
    't,': 'ṭ', 'T,': 'Ṭ',
    'h,': 'ḫ', 'H,': 'Ḫ',
    "'": 'ʾ', "`": 'ʿ',
    "’": 'ʾ', "‘": 'ʿ',
}

#%%
def extract_quoted_substring(text: str, start_pos: int):
    """
    Ищет в строке text, начиная С позиции start_pos,
    подстроку вида: ' "текст"'.

    Возвращает:
        (substring, is_longer_than_30, closing_quote_pos)

    Если ничего не найдено — (None, None, None)
    """
    translate = False
    open_seq = ' "'

    # поиск начинается С start_pos
    open_pos = text.find(open_seq, start_pos)

    if open_pos == -1:
        return None, None, start_pos

    # позиция открывающей кавычки "
    quote_start = open_pos + 1

    # ищем закрывающую кавычку "
    quote_end = text.find('"', quote_start + 1)

    if quote_end == -1:
        return None, None, start_pos

    # подстрока между кавычками
    substring = text[quote_start + 1 : quote_end]

    if len(substring) > 30:
        translate = True
         # 1. найти открывающую скобку
        open_pos = text.find("(", quote_end)
        # if open_pos == -1:
        #     return None, None, None
        # 2. проверить расстояние от закрывающей кавычки до открывающей скобки
        if open_pos - quote_end > 3:
            translate = False

    return substring, translate, quote_end

#%%
def extract_parenthesized_substring(text: str, start_pos: int):
    """
    С позиции start_pos ищет '('.
    Возвращает:
        (substring, flag, close_pos)
    """

    # 1. найти открывающую скобку
    open_pos = text.find("(", start_pos)
    if open_pos == -1:
        return None, None, start_pos

    # 2. проверить расстояние
    if open_pos - start_pos > 3:
        close_pos_tz = text.find(";", open_pos + 1)
        # скобка найдена, но слишком далеко
        close_pos_s = text.find(")", open_pos + 1)
        if close_pos_tz != -1 and close_pos_s != -1:
            close_pos = min(close_pos_tz, close_pos_s)
        else:
            if close_pos_tz == -1:
                close_pos = close_pos_s
            if close_pos_s == -1:
                close_pos = close_pos_tz
        if close_pos == -1:
            return None, None, start_pos
        return text[open_pos + 1 : close_pos], False, close_pos

    # 3. найти закрывающую скобку
    close_pos_tz = text.find(";", open_pos + 1)
    # скобка найдена, но слишком далеко
    close_pos_s = text.find(")", open_pos + 1)
    if close_pos_tz != -1 and close_pos_s != -1:
        close_pos = min(close_pos_tz, close_pos_s)
    else:
        if close_pos_tz == -1:
            close_pos = close_pos_s
        if close_pos_s == -1:
            close_pos = close_pos_tz
    if close_pos == -1:
        return None, None, start_pos

    substring = text[open_pos + 1 : close_pos]

    # 4. условия
    is_long = len(substring) > 30
    dash_count = substring.count("-")

    flag = is_long and dash_count >= 6

    return substring, flag, close_pos

#%%
def extract_letter_space_digit_colon_space(text: str, start_search_pos: int):
    # 1. Основной шаблон
    pattern = re.compile(
        r'[A-Za-z]{3,7} \d{4}: \d{1}'
    )

    match = pattern.search(text, start_search_pos)
    if not match:
        return None, None, start_search_pos
    # 2. Проверка 5 символов после группы
    pos = match.end()
    limit = min(pos + 5, len(text))

    start_pos = pos
    for i in range(pos, limit):
        if text[i].isdigit() or text[i] == '-':
            start_pos = i + 1
        else:
            start_pos = i
            break

    # 3. Поиск одинарной открывающей кавычки
    quote_pos = text.find("'", start_pos)
    if quote_pos == -1:
        return None, None, start_pos

    diff = quote_pos - start_pos
    if diff >= 1000:
        return None, None, start_pos

    # 4. Проверка подстроки
    substr = text[start_pos:quote_pos]

    dash_count = substr.count('-')
    aleph_count = substr.count('ℵ')

    dash_required = diff / 10.5

    if dash_count >= dash_required or aleph_count >= 2:
        transliter_txt = substr
        return transliter_txt, True, quote_pos - 1

    return None, None, start_pos

#%%
def extract_single_quotes(text: str, start_pos: int):
    if start_pos < 0 or start_pos >= len(text):
        return None, None, start_pos

    # 1. Поиск закрывающей одинарной кавычки
    quote_pos = text.find("'", start_pos)
    if quote_pos == -1:
        return None, None, start_pos

    # 2. Проверка расстояния
    if quote_pos - start_pos > 1200:
        return None, None, start_pos

    # 3. Извлечение подстроки
    translate_txt = text[start_pos:quote_pos]

    # 4. Возврат результата
    return translate_txt, True, quote_pos - 1

#%%
def normalize_akkadian_determinatives(text: str) -> str:
    for sup, norm in DETERMINATIVE_MAP.items():
        text = text.replace(sup, norm)
    return text

#%%
def normalize_subscripts(text: str) -> str:
    return text.translate(SUBSCRIPT_DIGITS)

#%%
def normalize_gaps(text: str) -> str:
    # порядок замен важен!
    replacements = [
        (r"\[\s*…\s*…\s*\]", "<big_gap>"),  # [… …]
        (r"\[x\]", "<gap>"),               # [x]
        (r"…", "<big_gap>"),               # …
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    return text

#%%
# def normalize_assyrian_line(text: str) -> str:
#     # 1. Удаляем редакторскую метку Pl-/
#     text = re.sub(r"^Pl-/\s*", "", text)
#
#     # 2. Удаляем перенос строки (\)
#     text = text.replace("\\", "")
#
#     # 3. Удаляем запятую (маркер переноса строки)
#     text = text.replace(",", "")
#
#     # 4. Обрабатываем квадратные скобки
#     def replace_supplied(match):
#         content = match.group(1).strip()
#
#         # лакуны — не трогаем
#         if content == "x":
#             return "[x]"
#         if re.fullmatch(r"[.\s…]+", content):
#             return f"[{content}]"
#
#         # восстановленный текст → склеиваем
#         return "-" + content
#
#     text = re.sub(r"\[([^\]]+)\]", replace_supplied, text)
#
#     # 5. Убираем лишние пробелы
#     text = re.sub(r"\s+", " ", text).strip()
#
#     return text

#%%
def normalize_for_mt(text: str) -> str:
    # 0. Базовая очистка (translate-таблица уже применяется снаружи)
    a = text
    chars_to_remove = "!?/:.<>˹˺[]ℵ⅀⅁"
    table = str.maketrans("", "", chars_to_remove)
    # 1. ASCII → Unicode
    for old, new in CHAR_MAP.items():
        a = a.replace(old, new)

    # удаление ненужных символов
    a = a.translate(table)

    # 2. Надстрочные детерминативы
    a = normalize_akkadian_determinatives(a)

    # 3. Подстрочные цифры
    a = normalize_subscripts(a)

    # 4. Удаляем редакторские маркеры
    a = re.sub(r"^Pl-/\s*", "", a)   # Pl-/
    a = a.replace("\\", "")          # перенос строки
    a = a.replace(",", "")           # маркер переноса строки

    # 5. Квадратные скобки: восстановление vs лакуны
    def handle_brackets(match):
        content = match.group(1).strip()

        # лакуны
        if content == "x":
            return "<gap>"
        if re.fullmatch(r"[.…\s]+", content):
            return "<big_gap>"

        # восстановленный текст → включаем в слово
        return "-" + content

    a = re.sub(r"\[([^\]]+)\]", handle_brackets, a)

    # 6. Финальная нормализация пробелов
    a = re.sub(r"\s+", " ", a).strip()

    return a

#%%
def align_and_mark_sentences(translit_text: str, translation_sentences: list, marker="<sent>") -> str:
    """
    Точная выравнивающая функция для вставки маркеров конца предложений в транслитерацию.

    Args:
        translit_text: Нормализованная транслитерация (str)
        translation_sentences: Список английских предложений (list of str)
        marker: Спец-токен конца предложения (default "<sent>")

    Returns:
        Строка транслитерации с маркерами конца предложений
    """
    translit_tokens = translit_text.split()
    translation_lengths = [len(sent.split()) for sent in translation_sentences]
    total_translit = len(translit_tokens)
    total_translation = sum(translation_lengths)

    # Вычисляем пропорцию токенов транслитерации на токен перевода
    tokens_per_translation_token = total_translit / total_translation

    marked_tokens = []
    idx = 0

    for length in translation_lengths:
        # Сколько токенов транслитерации примерно для этого предложения
        num_tokens = max(1, round(length * tokens_per_translation_token))
        sent_tokens = translit_tokens[idx: idx + num_tokens]
        marked_tokens.extend(sent_tokens)
        marked_tokens.append(marker)
        idx += num_tokens

    # Добавляем остаток токенов, если есть
    if idx < total_translit:
        marked_tokens.extend(translit_tokens[idx:])
        marked_tokens.append(marker)

    return " ".join(marked_tokens)

#%%

# def process_text_and_build_csv_rows(text: str):
#     """
#     Обрабатывает текст и возвращает список строк CSV
#     (без заголовка)
#     """
#     translate_str, accad_str = ''
#     next_pos, close_pos =0
#     extract_function_translate = [extract_quoted_substring, extract_single_quotes]
#     extract_function_accad_txt = [extract_parenthesized_substring, extract_letter_space_digit_colon_space]
#     str_txt = [translate_str, accad_str]
#     pos_num = [next_pos, close_pos]
#
#     csv_rows = []
#     start_pos = 0
#
#     while start_pos < len(text):
#         # поиск по двойным кавычкам
#         translate_str, flag, next_pos = extract_quoted_substring(text, start_pos)
#         # if translate_str is None:
#         #     break
#
#         if flag:
#             # print(translate_str)
#             # поиск по круглым скобкам
#             accad_str, flag2, close_pos = extract_parenthesized_substring(
#                 text, next_pos)
#
#             if flag2:
#                 # 1. Очистка перевода
#                 t = translate_str.replace("\n", " ")
#
#                 # 2. Очистка аккадского
#                 a = accad_str.replace("\n", " ")
#                 a = normalize_for_mt(a)
#
#                 # 3. Токенизация перевода
#                 t_sentences = sent_tokenize(t)
#
#                 # 4. Выравнивание + маркеры
#                 a = align_and_mark_sentences(a, t_sentences, marker="<sent>")
#
#                 # 5. Склеиваем перевод обратно
#                 t = " ".join(t_sentences)
#
#                 # 6. CSV-экранирование (ОДИН РАЗ!)
#                 a = a.replace('"', '""')
#                 t = t.replace('"', '""')
#
#                 csv_rows.append(f'"{a}","{t}"\n')
#                 start_pos = close_pos + 1
#             else:
#                 start_pos = next_pos + 1
#         # elif flag == False:
#         #     start_pos = next_pos + 1
#         else:
#             # поиск по буквам пробел цифрам двоеточию пробелу цифрам
#             accad_str, flag3, next_pos = extract_letter_space_digit_colon_space(text, start_pos)
#             if flag3:
#                 # поиск по одинарным кавычкам
#                 translate_str, flag4, close_pos = extract_parenthesized_substring(text, next_pos)
#                 if flag4:
#                     # 1. Очистка перевода
#                     t = translate_str.replace("\n", " ")
#
#                     # 2. Очистка аккадского
#                     a = accad_str.replace("\n", " ")
#                     a = normalize_for_mt(a)
#
#                     # 3. Токенизация перевода
#                     t_sentences = sent_tokenize(t)
#
#                     # 4. Выравнивание + маркеры
#                     a = align_and_mark_sentences(a, t_sentences, marker="<sent>")
#
#                     # 5. Склеиваем перевод обратно
#                     t = " ".join(t_sentences)
#
#                     # 6. CSV-экранирование (ОДИН РАЗ!)
#                     a = a.replace('"', '""')
#                     t = t.replace('"', '""')
#
#                     csv_rows.append(f'"{a}","{t}"\n')
#                     start_pos = close_pos + 1
#                 else:
#                     start_pos = next_pos + 1
#             # else:
#             #     # start_pos = next_pos + 1
#             #     return  []
#     return csv_rows
def process_text_and_build_csv_rows(text: str):
    """
    Обрабатывает текст и возвращает список строк CSV
    (без заголовка)
    """
    translate_str, accad_str = '', ''
    next_pos = 0
    close_pos = 0
    extract_function_1 = [extract_quoted_substring, extract_letter_space_digit_colon_space]
    extract_function_2 = [extract_parenthesized_substring, extract_single_quotes]
    str_txt = [translate_str, accad_str]
    str_txt_1 = [accad_str, translate_str]
    pos_num = [next_pos, close_pos]
    pos_num_1 = [close_pos, next_pos]
    len_arr = len(str_txt)
    i = 0
    csv_rows = []
    start_pos = 0

    # while start_pos < len(text):
    while i < len_arr:
        # поиск по двойным кавычкам потом по буквам пробелам цифрам
        str_txt[i % len_arr], flag, pos_num[i % len_arr] = extract_function_1[i % len_arr](text, start_pos)
        # if translate_str is None:
        #     break

        if flag:
            # print(translate_str)
            # поиск по круглым скобкам потом по одинарным кавычкам
            str_txt_1[i % len_arr], flag2, pos_num_1[i % len_arr] = extract_function_2[i % len_arr](text, pos_num[i % len_arr])
            if flag2:
                match i:
                    case 0:
                        translate_str = str_txt[i % len_arr]
                        accad_str = str_txt_1[i % len_arr]
                    case 1:
                        translate_str = str_txt_1[i % len_arr]
                        accad_str = str_txt[i % len_arr]
                # 1. Очистка перевода
                t = translate_str.replace("\n", " ")

                # 2. Очистка аккадского
                a = accad_str.replace("\n", " ")
                a = normalize_for_mt(a)

                # 3. Токенизация перевода
                t_sentences = sent_tokenize(t)

                # 4. Выравнивание + маркеры
                a = align_and_mark_sentences(a, t_sentences, marker="<sent>")

                # 5. Склеиваем перевод обратно
                t = " ".join(t_sentences)

                # 6. CSV-экранирование (ОДИН РАЗ!)
                a = a.replace('"', '""')
                t = t.replace('"', '""')

                csv_rows.append(f'"{a}","{t}"\n')
                start_pos = pos_num_1[i % len_arr] + 1
            else:
                start_pos = pos_num[i % len_arr] + 1
        else:
            start_pos = pos_num[i % len_arr] + 1
        if start_pos >= len(text):
            i += 1
            start_pos = 0
    return csv_rows

#%%
# ----------------------------
# Функция разбивки перевода на предложения
# ----------------------------
def naive_sent_tokenize(text):
    """
    Разделяет текст на предложения по точкам, восклицательным и вопросительным знакам.
    Работает для английского перевода.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
#%%
import csv
from io import StringIO

def parse_csv_line(line: str):
    reader = csv.reader(StringIO(line))
    accad_str, translate_str = next(reader)
    return accad_str, translate_str
#%%
# ----------------------------
# Выравнивание и разбивка транслитерации по <sent>
# ----------------------------
def split_accad_and_translate(csv_lines, marker="<sent>"):
    rows = []
    global_id = 0

    for line in csv_lines:
        accad_str, translate_str = parse_csv_line(line)

        accad_sentences = [s.strip() for s in accad_str.split(marker) if s.strip()]
        translate_sentences = naive_sent_tokenize(translate_str)

        min_len = min(len(accad_sentences), len(translate_sentences))
        accad_sentences = accad_sentences[:min_len]
        translate_sentences = translate_sentences[:min_len]

        for accad, trans in zip(accad_sentences, translate_sentences):
            rows.append({
                "id": global_id,
                "accad_str": accad,
                "translate": trans
            })
            global_id += 1

    return pd.DataFrame(rows, columns=["id", "accad_str", "translate"])

#%%
def print_file_head(path, n=5, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(f"{i}: {line.rstrip()}")

#%%
# Завантаження даних з CSV-файлу
# thiscompteca = "D:/Projects/Python/Конкурсы/Old_accad_translate/"
thiscompteca = "G:/Visual Studio 2010/Projects/Python/Old_accad_translate/"
csv_file_path = thiscompteca+'/data/publications.csv'
df_trnl = pd.read_csv(csv_file_path)

# print(df_trnl[df_trnl['has_akkadian']].head(20))  # Перші 5 строк даних
# print(df_trnl.shape)  # Dataset Shape
# print(df_trnl.info())  # Dataset Information
# print(df_trnl.describe())   # Statistics
# print(df_trnl.isnull().sum())  # Missing Values
print('\n')

idx = df_trnl[df_trnl['has_akkadian']].head(5).index
# idx = df_trnl[df_trnl['has_akkadian']].index
df_trnl.loc[idx, df_trnl.columns[2]] = (
    df_trnl.loc[idx, df_trnl.columns[2]]
    .str.replace("\\n", "\n", regex=False)
)
# num = 0
all_rows = []
for i in idx:
    # print(f"index = {i}")
    # print("Назва файлу:", df_trnl.iat[i, 0])
    # print("Сторінка з текстом, що містить переклад:", df_trnl.iat[i, 1])
    # print("Текст всієї статті:\n", df_trnl.iat[i, 2])
    # print("-" * 50)
    list_row = []
    list_row = process_text_and_build_csv_rows(df_trnl.iat[i, 2])
    # if list_row != []:
    all_rows.extend(list_row)
    # num += 1

# for i in idx[:10]:  # первые 10 для проверки
#     text = df_trnl.iat[i, 2]
#     rows = process_text_and_build_csv_rows(text)
#     print(f"Строка {i}: найдено {len(rows)} фрагментов")



new_df = split_accad_and_translate(all_rows)
new_df.to_csv('translate_from_publication.csv', index=False, quoting=csv.QUOTE_ALL)
# print("Примеры строк:")
# print(new_df)
# print(len(idx))
# print(num)
print('\n')

#%%
# Завантаження даних з CSV-файлу
# thiscompteca = "C:/Users/arecs/Мій диск (2armnot@gmail.com)/Питон/Конкурси/Old_Assyrian/"
csv_file_path = thiscompteca+'/data/published_texts.csv'
df_txt = pd.read_csv(csv_file_path)
num_row = 0
for num_row in range(df_txt.shape[0]):
    if num_row > 3:
        break
    for num_col in range(df_txt.shape[1]):
        print(df_txt.iat[num_row, num_col])
    print('-' * 50)

# print(df_txt.head())  # Перші 5 строк даних
# print(df_txt.shape)  # Dataset Shape
# print(df_txt.info())  # Dataset Information
# print(df_txt.describe())   # Statistics
# print(df_txt.isnull().sum())  # Missing Values
#%%
# Завантаження даних з CSV-файлу
# thiscompteca = "C:/Users/arecs/Мій диск (2armnot@gmail.com)/Питон/Конкурси/Old_Assyrian/"
csv_file_path = thiscompteca+'/data/bibliography.csv'
df_txt = pd.read_csv(csv_file_path)


# print(df_txt.head())  # Перші 5 строк даних
# print(df_txt.shape)  # Dataset Shape
print(df_txt.info())  # Dataset Information
# print(df_txt.describe())   # Statistics
# print(df_txt.isnull().sum())  # Missing Values

num_row = 0
for num_row in range(df_txt.shape[0]):
    # if num_row > 10:
    #     break
    for num_col in range(df_txt.shape[1]):
        if df_txt.iat[num_row, 2] == 'Mogens Trolle Larsen':
            print(df_txt.iat[num_row, num_col])
            if num_col == df_txt.shape[1] - 1:
                print('-' * 50)
#%%
# Завантаження даних з CSV-файлу
# thiscompteca = "C:/Users/arecs/Мій диск (2armnot@gmail.com)/Питон/Конкурси/Old_Assyrian/"
csv_file_path = thiscompteca+'/data/train.csv'
df_txt = pd.read_csv(csv_file_path)
num_row = 0
for num_row in range(df_txt.shape[0]):
    if num_row > 5:
        break
    for num_col in range(df_txt.shape[1]):
        print(df_txt.iat[num_row, num_col])
    print('-' * 50)

# print(df_txt.head())  # Перші 5 строк даних
# print(df_txt.shape)  # Dataset Shape
# print(df_txt.info())  # Dataset Information
# print(df_txt.describe())   # Statistics
# print(df_txt.isnull().sum())  # Missing Values
#%%
