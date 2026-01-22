import pandas as pd
import re

# Разделители блоков
SEPARATOR_RE = re.compile(r'^-+$')

# Разрешенные символы для транслитерации
TRANSLIT_LINE_RE = re.compile(
    r"^[A-Za-zŠšḫḪṣṢṭṬʾʿ0-9\-ℵ \[\]\.\!⅀⅁ᲟᲠ–]+$"
)

# Морфемные разделители (дефис или ℵ)
MORPHEME_SEP_RE = re.compile(r"[-ℵ]")

# Стоп-слова для фильтрации английского и немецкого текста - расширенный список
FOREIGN_WORD_RE = re.compile(
    r"\b("
    # Немецкие слова
    r"Jetzt|ist|gerade|ein|Brief|des|an|und|der|die|das|von|mit|"
    r"für|auf|aus|bei|nach|über|unter|zwischen|durch|wegen|"
    # Английские слова
    r"desk|bound|commercial|manager|who|conducted|"
    r"this|must|have|been|invented|institution|"
    r"if|when|is|going|to|in|at|and|palace|textiles|old|assyrian|procedures|"
    r"the|a|an|of|for|with|from|by|on|as|or|but|not|so|then|also|"
    r"that|which|what|where|why|how|"
    r"he|she|it|we|they|"
    r"was|were|be|being|been|"
    r"will|would|can|could|should|may|might|must|"
    r"about|above|after|against|among|around|before|behind|below|beneath|beside|between|beyond|"
    r"during|except|inside|outside|since|through|throughout|toward|under|until|upon|within|without"
    r")\b",
    re.I
)

# Явные признаки аккадской транслитерации
AKKADIAN_INDICATOR_RE = re.compile(
    r"[ŠšḪḥṢṣṬṭʾʿ⅀⅁ᲟᲠ]|"  # Аккадские специальные символы
    r"\[.*?\]|"  # Квадратные скобки
    r"\(.*?\)|"  # Круглые скобки
    r"\{.*?\}|"  # Фигурные скобки
    r"\b[A-Z][a-zšḫṭṣ]+-[a-zšḫṭṣ]+\b|"  # Слова с дефисом, начинающиеся с заглавной
    r"\b[a-zšḫṭṣ]+-[a-zšḫṭṣ]+\b|"  # Слова с дефисом из строчных
    r"\b\d+[rv]\b|"  # Номера строк: 14r, 15v и т.д.
    r"x\+|x\-|x\?|x=\d+|"  # Фрагменты табличек
    r"\.\.\.|…|"  # Многоточия
    r"\d+['ˈ]|"  # Числа с апострофом
    r"–[^ ]"  # Длинное тире не после пробела
)

# Признаки, что это НЕ транслитерация (пропускать такие строки)
NOT_TRANSLIT_RE = re.compile(
    r"\b[A-Z][a-z]{3,} [A-Z][a-z]{3,}\b|"  # Два заглавных слова подряд (имя собственное)
    r"\b[a-z]{4,} [a-z]{4,} [a-z]{4,}\b|"  # Три длинных слова подряд (предложение)
    r"^\d+ [A-Z][a-z]|"  # Начинается с цифры и заглавной буквы
    r"[a-z]{5,}-[a-z]{4,}[^šḫṭṣʾʿ]|"  # Длинные английские слова с дефисом
    r", |; |: |\. [A-Z]|"  # Знаки пунктуации с пробелом
    r"\b(?:[A-Za-z]+ ){3,}[A-Za-z]+\b"  # Более 3 слов подряд
)


def extract_transliteration(text) -> list:
    """
    Извлекает блоки транслитерации из текста.
    Склеивает строки, оканчивающиеся на - или ℵ с последующей.
    Возвращает список блоков.
    """
    if isinstance(text, list):
        text = "\n".join(text)

    raw_lines = text.splitlines()
    lines = []
    buffer = ""

    for line in raw_lines:
        line = line.rstrip()
        if not line:
            if buffer:
                lines.append(buffer)
                buffer = ""
            continue

        if not buffer:
            buffer = line
        else:
            buffer += " " + line

        if not line.endswith(("-", "ℵ")):
            lines.append(buffer)
            buffer = ""

    if buffer:
        lines.append(buffer)

    # Формируем блоки транслитерации
    blocks = []
    current = []

    for line in lines:
        # Пропускаем разделители
        if SEPARATOR_RE.match(line):
            continue

        line_trimmed = line.strip()

        # Пропускаем пустые строки
        if not line_trimmed:
            continue

        # Проверка 1: Соответствует ли базовому формату транслитерации?
        has_basic_format = (
                TRANSLIT_LINE_RE.match(line_trimmed) and
                MORPHEME_SEP_RE.search(line_trimmed)
        )

        if not has_basic_format:
            if current:
                blocks.append("\n".join(current).strip())
                current = []
            continue

        # Проверка 2: Содержит ли иностранные слова?
        has_foreign_words = FOREIGN_WORD_RE.search(line_trimmed)

        # Проверка 3: Содержит ли явные признаки НЕ транслитерации?
        is_not_translit = NOT_TRANSLIT_RE.search(line_trimmed)

        # Проверка 4: Содержит ли признаки аккадской транслитерации?
        has_akkadian_indicators = AKKADIAN_INDICATOR_RE.search(line_trimmed)

        # Логика принятия решения:
        # 1. Должен быть базовый формат
        # 2. Не должен содержать иностранных слов ИЛИ должен иметь аккадские индикаторы
        # 3. Не должен быть явно НЕ транслитерацией
        is_transliteration = (
                has_basic_format and
                (not has_foreign_words or has_akkadian_indicators) and
                not is_not_translit
        )

        # Особый случай: если есть аккадские индикаторы, принимаем даже с некоторыми иностранными словами
        if has_akkadian_indicators and has_basic_format and not is_not_translit:
            is_transliteration = True

        if is_transliteration:
            current.append(line_trimmed)
        else:
            if current:
                blocks.append("\n".join(current).strip())
                current = []

    if current:
        blocks.append("\n".join(current).strip())

    return blocks


# Завантаженние и обработка данных
thiscompteca = "D:/Projects/Python/Конкурсы/Old_accad_translate/"
csv_file_path = thiscompteca + '/data/publications.csv'
df_trnl = pd.read_csv(csv_file_path)

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
    if blocks:
        all_blocks.extend(blocks)

# Формируем финальный текст с разделителями
if all_blocks:
    separator = "\n" + "-" * 40 + "\n"
    result_text = separator.join(all_blocks)

    print("Найдено блоков транслитерации:", len(all_blocks))

    # Отладочный вывод: покажем проблемные строки которые были отфильтрованы
    print("\nПримеры отфильтрованных строк (для проверки):")
    test_strings = [
        "desk-bound commercial manager who conducted",
        "This naruqqu-institution must have been invented",
        "14 Jetzt ist gerade ein Brief des Iriba 15 an Litib-libbaSu 16 und ein Brief"
    ]

    for test_str in test_strings:
        print(f"\nПроверка строки: '{test_str}'")
        has_basic = TRANSLIT_LINE_RE.match(test_str) and MORPHEME_SEP_RE.search(test_str)
        has_foreign = bool(FOREIGN_WORD_RE.search(test_str))
        not_translit = bool(NOT_TRANSLIT_RE.search(test_str))
        akkadian = bool(AKKADIAN_INDICATOR_RE.search(test_str))

        print(f"  Базовый формат: {has_basic}")
        print(f"  Иностранные слова: {has_foreign}")
        print(f"  NOT транслитерация: {not_translit}")
        print(f"  Аккадские индикаторы: {akkadian}")
        print(
            f"  Результат: {'ПРИНЯТО' if (has_basic and (not has_foreign or akkadian) and not not_translit) else 'ОТФИЛЬТРОВАНО'}")

    print("\n" + "=" * 60 + "\n")

    # Выводим первые 3 блока для проверки
    for i, block in enumerate(all_blocks[:3], 1):
        print(f"Блок транслитерации {i}:")
        print(block[:300] + "..." if len(block) > 300 else block)
        print("-" * 50)

    # Сохраняем в файл
    with open(thiscompteca + '/transliteration_blocks.txt', 'w', encoding='utf-8') as f:
        f.write(result_text)
else:
    print("Блоки транслитерации не найдены")