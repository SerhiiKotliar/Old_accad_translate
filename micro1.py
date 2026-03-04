# file: renumber_trust_source.py

import re
from typing import Dict, List, Tuple, Match, Pattern
import nltk
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
Pattern_search_translate = r"(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+)|(?P<number>\d+))"


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
    "§": 'S', "⅀": "š",
    "$": "š", "∫": "š",
    "ß": "š", "ʃ": "š",
    "–": "-", "—": "-",
}


def is_clean_akkadian_translation(text: str) -> bool:
    """
    Возвращает True, если текст похож на чистый перевод с аккадского
    без транслитерации и научных комментариев.
    """

    if not text or not text.strip():
        return False

    # 1. Признаки аккадской транслитерации
    transliteration_patterns = [
        r'\b[a-z]+-[a-z]+\b',           # a-na, i-na, ša-ma
        # r'[ŠšḪḫṬṭṢṣĀāĒēĪīŪū]',          # диакритика
        r'\b[A-Z]+\.[A-Z.]+\b',         # KÙ.BABBAR
    ]

    for pattern in transliteration_patterns:
        if re.search(pattern, text):
            return False

    # 2. Научные ссылки
    scholarly_patterns = [
        r'\b\d{4}\b',                   # год (1985)
        r'\b(cf\.?|see|vgl\.?)\b',      # cf., see
        r'\b[Kk][Tt]\s*n/?k\b',         # KT n/k
        r'\b[Kk][Bb][Oo]\b',            # KBo
    ]

    for pattern in scholarly_patterns:
        if re.search(pattern, text):
            return False

    # 3. Слишком много заглавных слов (как в каталогах)
    uppercase_words = re.findall(r'\b[A-Z]{3,}\b', text)
    if len(uppercase_words) > 3:
        return False

    # 4. Проверка что текст состоит в основном из букв и обычной пунктуации
    allowed_ratio = len(re.findall(r'[A-Za-zА-Яа-я ,.\-–—?!:;()\n]', text)) / len(text)
    if allowed_ratio < 0.85:
        return False

    return True



def get_next_line(text: str, start_pos: int):
    """возвращает не пустую строку, следующую за назначенной позицией
     и позицию конца строки"""
    # начало строки поиска
    pos = None if start_pos == len(text) else start_pos
    if pos  is None:
        return "", len(text)
    # конец строки поиска
    end = text.find('\n', pos)
    if end == -1 and pos <= len(text):
        end = len(text)
        return text[pos:end], end
    # позиция старта совпадает с переводом строки
    if pos == end and pos < len(text):
        # return text[pos:end+1], end+1
        pos = end + 1
        end = text.find('\n', pos)
        if end == -1 and pos <= len(text):
            end = len(text)
            return text[pos:end], end
    # if end == pos and pos < len(text):
    #     pos = end + 1
    #     end = text.find('\n', pos)
    #     if end == -1 and pos <= len(text):
    #         end = len(text)
    # достигнут конец текста
    if end == pos and pos >= len(text):
        return "", len(text)
    str_line = text[pos:end]
    # str_line = re.sub(
    #     r'^\s*(?:[SK]\.|S\. K\.|v|\. v)\s*(?:\r?\n|$)',
    #     '',
    #     str_line,
    #     flags=re.MULTILINE
    # )
    # str_line = re.sub(
    #     r'(?m)^\s*\d{1,2}\.\s*',
    #     '',
    #     str_line
    # )

    return str_line, end+1



def find_translate_by_rows(text: str, pos: int=0, n_dop: int=2):
    """поиск перевода начиная с позиции после якоря по строкам
    возвращает транслитерацию или "" и её позиции конца и начала"""
    pos_end_of_line = 0
    pos_start_per = pos
    # start_detect = pos_start_per
    pos_start_translate = 0
    end_translate = 0
    result = ""
    # pattern_end = r"^\d+\.\d+\.\s*(?:(?:\d+\.|[A-Za-z]*\.)?(?:e\.|r\.)|\d+(?:[’'])?)"
    num_row = 0
    while pos < len(text):
    # if pos < len(text):
        # строка от её первой позиции и позиция конца строки
        n_l, pos_end_of_line = get_next_line(text, pos)
        # конец транслитерации
        # match_nl = re.compile(pattern_end).search(n_l)
        # if match_nl:
        #     return result, end_translit, pos_start_transliteration
        # прекращение поиска транслитерации после 2 ложных строк
        # if num_row > n_dop-1:
        #     return "", pos_end_of_line, pos_start_per
        # line_trl = []
        # if n_l and is_clean_akkadian_translation(n_l):
            # line_trl = extract_transliteration(n_l)
        # if line_trl:
        #     pos_start_translate = pos
        #     end_translate = pos_end_of_line
        # end_translit = 0
        pos_start_translate = pos
        while n_l and is_clean_akkadian_translation(n_l):
            # pos_start_translate = pos
            # end_translate = pos_end_of_line
            # pos_start_transliteration = pos
            # сборная транслитерация
            # result += "\n".join(n_l) + "\n"
            result += "".join(n_l)
            end_translate = pos_end_of_line
            # if (pos_end_of_line - len(n_l) - 1) > 0 and pos_start_trlit == start_detect:
            #     pos_start_transliteration = pos_end_of_line - len(n_l) - 1
            #     pos_start_trlit = pos_start_transliteration
            # else:
            pos_start_per = pos_end_of_line - len(n_l) - 1
            # строка
            n_l, pos_end_of_line = get_next_line(text, pos_end_of_line)
            # конец транслитерации
            # match_nl = re.compile(pattern_end).search(n_l)
            # if match_nl:
            #     return result, end_translit, pos_start_transliteration
            if pos_end_of_line == -1:
                return result, end_translate, pos_start_per
            # if n_l:
            #     line_trl = extract_transliteration(n_l)
            # else:
            #     line_trl = ""
            # end_translit = pos_end_of_line
        num_row += 1
        pos = pos_end_of_line
        if result:
            return result, end_translate, pos_start_translate


    return "", pos_end_of_line, pos_start_translate



text = """$í-ip-ri-ni
li-sú-ha-am ú a-tù-nu
lu e-mu-uq
$í-ip-ri-n[i]

15

20

Extradite the man there together with §®p-
I$tar, our messenger, and you
must be the executive arm for our messen-
ger.

"""

txt = find_translate_by_rows(text)
print(txt)