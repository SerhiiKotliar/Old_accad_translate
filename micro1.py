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




def clear_from_ocr_for_text(text: str) -> str:
    """Упорядочивает последовательно значения диапазонов
    и оборачивает в круглые скобки"""
    # --- 1. OCR-мусор: " 3A" → "3-A"
    # text = re.sub(r'(\s\d)\s*(\d\w)', r'\1-\2', text)

    token_pattern = re.compile(
        r'\(?\s*\d{1,3}\s*[-–—]\s*\d{1,3}\s*\)?'
        r'|\(?\s*\d{1,3}\s*\)?'
        r'|(?<!\d)[–—-]\s*\d{1,3}'
        r'|\b\d{1,3}\b'    # шаблон отдельного числа
    )

    tokens = []
    for m in token_pattern.finditer(text):
        # пропуск чисел без признаков нумерации
        if not m.group().startswith("(") and not m.group().endswith(")") and not m.group().endswith("'"):
            continue
        token = m.group()
        # если найдено (N) → превратить в (N-N)
        if (token.startswith("(") and token.endswith(")")) or (token.endswith(")"))  or (token.endswith("'")):
            inner = token[1:-1].strip()
            if inner.isdigit():
                token = f"({inner}-{inner})"

        tokens.append({
            "start": m.start(),
            "end": m.end(),
            # "text": m.group()
            "text": token
        })

    # --- 2. Разбираем токены в диапазоны
    parsed = []
    for t in tokens:
        s = t["text"]
        m = re.match(r'\(?\s*(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*\)?'r'|\(?\s*\d{1,3}\s*\)?', s)
        # m = re.match(r'\(?[^\S\n]*(\d{1,3})[^\S\n]*[-–—][^\S\n]*(\d{1,3})[^\S\n]*\)?(?=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', s)
        if m:
            parsed.append({"type": "range", "a": int(m.group(1)), "b": int(m.group(2)), **t})
            continue

        m = re.match(r'[–—-]\s*(\d{1,3})', s)
        if m:
            parsed.append({"type": "broken", "b": int(m.group(1)), **t})
            continue
        # ----------------------------------------------------------
        # собирает отдельные числа
        # parsed.append({"type": "single", "n": int(s), **t})
        parsed.append({"type": "single", "a": int(s), **t})
        # -------------------------------------------------------------
    # --- 3. Исправляем логику (исправляем ПРЕДЫДУЩИЙ диапазон)
    last_range = None
    del_items = []
    for i, item in enumerate(parsed):
        if item["type"] == "range":
            # is_del = False
            if last_range and item["a"] <= last_range["b"]:
                diff = item["a"] - last_range["b"]
                if diff > 1 or diff < -1:
                    del_items.append(i)
                    # is_del = True
                    continue
                last_range["b"] = item["a"] - 1
                if last_range["b"] < last_range["a"]:
                    last_range["b"] = last_range["a"]
            # if is_del:
            #     continue
            last_range = item

        elif item["type"] == "broken" and last_range:
            a = last_range["b"] + 1
            item["type"] = "range"
            item["a"] = a
        # --------------------------------------------------
            last_range = item
        # для обнаружения отдельных чисел
        # elif item["type"] == "single":
        #     if last_range and item["n"] <= last_range["b"]:
        #         item["n"] = last_range["b"] + 1
        elif item["type"] == "single":
            is_del = False
            if last_range and item["a"]:
                str_it = str(item["a"])
                if len(str_it) > 2:
                    str_it = str_it[:2]
                    item["a"] = int(str_it)
                diff = item["a"] - last_range["b"]
                if diff> 1 or diff < -1:
                    del_items.append(i)
                    is_del = True
                elif diff == -1 or diff == 0:
                    item["a"] = last_range["b"] + 1
            # if last_range and item["a"] <= last_range["b"]:
            #     item["a"] = last_range["b"] + 1
            if is_del:
                continue
            if not last_range:
                del_items.append(i)
                # is_del = True
                continue
            item["type"] = "range"
            item["b"] = item["a"]
            last_range = item
    parsed = [item for i, item in enumerate(parsed) if i not in del_items]
        # ------------------------------------------------------
    # --- 4. Точечная замена (справа налево!)
    chars = list(text)

    for item in reversed(parsed):
        if item["type"] == "range":
            repl = f" {item['a']}-{item['b']} "
        # учитывает отдельные числа
        # elif item["type"] == "single":
        #     repl = str(item["n"])
        else:
            continue

        chars[item["start"]:item["end"]] = repl
    result = "".join(chars)
    # # ----------------------------------------------------
    # # если не обёрнуты, оборачивает в скобки
    # # pattern = re.compile(r'\(?(\d+)-(\d+)\)?'r'|\b\d{1,3}\b')
    # pattern = re.compile(r'\(?(\d+)-(\d+)\)?')
    # 
    # def wrap_if_no_parentheses(match: re.Match) -> str:
    #     full = match.group(0)  # всё совпадение
    #     a = match.group(1)
    #     b = match.group(2)
    # 
    #     if full.startswith("(") and full.endswith(")"):
    #         return full  # уже в скобках — оставить как есть
    #     else:
    #         return f"({a}-{b})"  # обернуть
    # 
    # result = pattern.sub(wrap_if_no_parentheses, result)
    return result


def clear_from_ocr_for_text_last(text: str) -> str:
    """Окончательно чистит мусор и форматирует по пробелам диапазоны"""

    pattern = re.compile(
        r'(\d+)\s*-\s*(\d+)(\s+([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]+))?'
    )

    def range_repl(m):
        left = m.group(1)
        right = m.group(2)
        word = m.group(4)

        # если правая часть длиннее
        if len(right) > len(left) and len(left) > 1:
            # отрезаем излишек
            main_right = right[:len(left)]
            # излишек
            extra = right[len(left):]

            # смотрим что идёт после всего совпадения
            rest = text[m.end():]

            #  если справа дробь → удаляем extra
            if re.match(r'\s*\d+/\d+', rest):
                return f"{left}-{main_right}"

            #  если справа слово (захваченное)
            if word:
                extra_conv = (
                    extra.replace('1', 'I')
                         .replace('0', 'O')
                         .replace('5', 'S')
                )
                return f"{left}-{main_right} {extra_conv}{word}"

            #  иначе просто удаляем extra
            return f"{left}-{main_right} "

        return m.group(0)

    text = pattern.sub(range_repl, text)
    text = re.sub(r'\s+', ' ', text).strip()
    # ----------------------------------------------------
    # если не обёрнуты, оборачивает в скобки
    # pattern = re.compile(r'\(?(\d+)-(\d+)\)?'r'|\b\d{1,3}\b')
    pattern = re.compile(r'\(?(\d+)-(\d+)\)?')
    last_range = None
    def wrap_if_no_parentheses(match: re.Match) -> str:
        nonlocal last_range
        full = match.group(0)  # всё совпадение
        a = match.group(1)
        b = match.group(2)
        if last_range:
            diff = int(a) - int(last_range["b"])
            if diff > 1 or diff < -1:
                return full
        last_range = {"b": b, "a": a}

        if full.startswith("(") and full.endswith(")"):
            return full  # уже в скобках — оставить как есть
        else:
            return f"({a}-{b})"  # обернуть

    text = pattern.sub(wrap_if_no_parentheses, text)
    return text


def cleaning_from_ocr_prelim(text: str) -> str:
    text = re.sub(
        # r'^\s*(?:[SK]\.|S\. K\.|S\.K\.|K\.\s*\d|\n|v|\. v)\s*$',
        # r'^\s*(?:[SK]\.|S\. ?K\.|K\.\s*\d+|v|\. v)\s*\r?\n?',
        r'^\s*(?:S\.(?:\s*K\.)?|K\.(?:\s*)?|v|\. v)\s*',
        '',
        text,
        flags=re.MULTILINE
    )
    # text = re.sub(r'^K\.\s*(\d+)', '\g<1>', text, flags=re.MULTILINE)
    text = re.sub(r'^\w\.\s*K\.\s*\w+', '', text, flags=re.MULTILINE)
    subs = [
        (r'([a-z])ı\s+', r'\g<1>i '),
        (r'ı\s+ı', '11'),
        (r'ı\s+', '1'),
        (r'ı', '1'),
        (r'\s5([A-Za-zА-Яа-я])', r' S\g<1>'),
        (r'A1', 'Ai'),
        (r'([A-Za-zА-Яа-я])1\b', r'\g<1>i'),
        (r'([A-Za-zА-Яа-я]),(\d)', r'\g<1> \g<2>'),
        (r'\s[iI]\s?(\d+)', r'1\g<1>'),
        (r'(?<=\d)o', '0'),
        (r'(?<=\d)°', '0'),
        (r'S([-–—])(\d)', r'5\g<1>\g<2>'),
        (r'(\d)([-–—])S', r'\g<1>\g<2>5'),
        (r'‰', ''),
        (r'™', ''),
        (r'([^\W\d_])4(-|[^\W\d_])', r'\g<1>h\g<2>'),
        (r'(\s\d\s*)4([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>h\g<2>'),
        (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\g<1>h'),
        (r"r'", "r"),
        (r'(\s\d)\s(\d[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>-\g<2>'),
        (r'([^\d\s])(\d+[\s\-–—]?\d+)([^\d\s])', r'\g<1> \g<2> \g<3>'),
        (r'(\d+[\s\-–—]?\d+)([^\d\s])', r'\g<1> \g<2>'),
        (r'(\d+)\s*-\s*(\d+)', r'\g<1>-\g<2>'),
        (r'^.\.?\s?y\.\s?\r?\n?', ''),
        (r'(^\d)(\d{1,2})\s(\d{1,2})(^\d)', r'\g<1> \g<2>-\g<3> \g<4>'),
        # (r'(\d\s*[-–—]?\s*)["“”«»„‟](\w)', r'\g<1>11\g<2>'),
        # (r'([^\d])(\d{1,2})\s(\d{1,2})([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>\g<2>-\g<3>\g<4>'),
        # (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])(\d+\s*[-–—]?\s*\d+)([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1> \g<2> \g<3>'),
        # (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])(\d+[\s*-–—\s*]\d+)([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>\s\g<2>\s\g<3>'),
        # (r'.\,.', ''),
        (r'\,\n', ''),
        (r'^\d+\n', ''),
        (r"(\d{1,2})'[-–—]\s*(\d{1,2})",r'\g<1>1-\g<2>'),
        (r"[-–—]'(\d{1,2})", r'-\g<1>'),
        (r'(\w)1(\w)', r'\g<1>i\g<2>')
        # (r'(?<=[^\W_]):(?=[^\W_])', ' '),
        # (r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', ''),
        # (r'§', 'S'),
        # (r'\,', ' '),
        # (r'^.\.y\.\s*', ''),
        # (r'^.\.y\.\n', ''),
    ]
    for pattern, repl in subs:
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)

    PATTERN: Pattern[str] = re.compile(
         r'(\D\s*)(\d{1,2})\s+(\d{1,3})(\s*[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])'
    )

    def transform_last_char(char: str) -> str:
        """
        Transform trailing digit:
            5 -> S
            1 -> I
            0 -> O
            other digit -> removed
            letter -> unchanged
        """
        if char.isdigit():
            if char == "5":
                return "S"
            if char == "1":
                return "I"
            if char == "0":
                return "O"
            return ""  # why: other digits must be removed
        return char

    def conditional_replace(match: Match[str]) -> str:
        len_left = len(match.group(2))
        len_right = len(match.group(3))
        right_out = len_right - len_left
        last_char: str = ""
        if right_out > 0:
            right: int = int(match.group(3)[:-right_out])
            last_char = transform_last_char(match.group(3)[-right_out:])
        else:
            right: int = int(match.group(3))
        left: int = int(match.group(2))
        next_str = match.group(4).strip()

        return f"{match.group(1)} {left}-{right} {last_char}{next_str}"

    def replace_if_left_less(text: str) -> str:
        """
        Replace space with dash only if left number < right number.
        """
        return PATTERN.sub(conditional_replace, text)

    text = replace_if_left_less(text)

    PATTERN1: Pattern[str] = re.compile(
        r'([^\d])(\d)\s(\d)-(\d{2})([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])'
    )

    def conditional_replace1(match: Match[str]) -> str:
        left: int = int(match.group(2) + match.group(3))
        right: int = int(match.group(4))

        if left < right:
            return f"{match.group(1)}{left}-{right}{match.group(5)}"

        return match.group(0)  # why: preserve original if condition fails

    def process_text1(text: str) -> str:
        return PATTERN1.sub(conditional_replace1, text)
    text = process_text1(text)

    pattern2 = re.compile(r'(\d\s*[-–—]?\s*)[\"“”«»„‟]([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])')

    def replace_func2(match):
        # первая группа: убрать пробелы, оставить цифру и тире
        first = re.sub(r'\s+', '', match.group(1))
        return f"{first}11 {match.group(2)}"

    text = pattern2.sub(replace_func2, text)

    return text


def cleaning_from_ocr(text: str, trlit: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)
   # уборка мусора
    if trlit:
        # text = re.sub(
        #     r'^\s*(?:[SK]\.|S\. K\.|S\.K\.|K\.\s*\d|\n|v|\. v)\s*$',
        #     '',
        #     text,
        #     flags=re.MULTILINE
        # )
        subs = [
            # (r'([a-z])ı\s+', r'\1i '),
            # (r'ı\s+ı', '11'),
            # (r'ı\s+', '1'),
            # (r'ı', '1'),
            # (r'5([A-Za-zА-Яа-я])', r'S\1'),
            # (r'A1', 'Ai'),
            # (r'([A-Za-zА-Яа-я])1\b', r'\1i'),
            # (r'([A-Za-zА-Яа-я]),(\d)', r'\1 \2'),
            # (r'\s(\d)\s(\d)\s', r' \1-\2 '),
            # (r'(?<=\d)o', '0'),
            # (r'(?<=\d)°', '0'),
            # (r'S-9', '5-9'),
            # (r'‰', ''),
            # (r'™', ''),
            (r':', ' '),
            (r'<([^<>]+)>', r'\g<1>'),
            (r'^.\d{1,}\n', ''),
            (r'^.\.?\s?y\.\s?\r?\n?', ''),
            (r'(?<=[A-Za-z0-9]):(?=[A-Za-z0-9])', ' '),
            # (r'([^\W\d_])4(-|[^\W\d_])', r'\g<1>h\g<2>'),
            # (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\g<1>h'),
            (r'(.)\,(.)', r'\g<1>\g<2>'),
            (r'\,\n', ''),
            #(r'(?<=[^\W_]):(?=[^\W_])', ' '),
            # (r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', ''),
            # (r'§', 'S'),
            # (r'\,', ' '),
            # (r'^.\.y\.\s*', ''),
            # (r'^.\.y\.\n', ''),
        ]
    else:
        subs = [
            # (r'([a-z])ı\s+', r'\1i '),
            # (r'ı\s+ı', '11'),
            # (r'ı\s+', '1'),
            # (r'ı', '1'),
            # (r'5([A-Za-zА-Яа-я])', r'S\1'),
            # (r'A1', 'Ai'),
            # (r'([A-Za-zА-Яа-я])1\b', r'\1i'),
            # (r'([A-Za-zА-Яа-я]),(\d)', r'\1 \2'),
            # (r'\s(\d)\s(\d)\s', r' \1-\2 '),
            # (r'(?<=\d)o', '0'),
            # (r'(?<=\d)°', '0'),
            # (r'S-9', '5-9'),
            # (r'‰', ''),
            # (r'™', ''),
            (r':', ''),
            (r'§', 'S'),
            (r'\$', '9'),
            # (r'([^\W\d_])4(-|[^\W\d_])', r'\g<1>h\g<2>'),
            # (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\g<1>h'),
            # (r'9([A-ZА-ЯÀ-ÖØ-Þ])([a-zа-яà-öø-ÿ])', r'\1\2'),
            (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])([-\s])9([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>\g<2>\g<3>'),
            # (r'(\d{1,2}[-\s]\d{1,2})\$', r'\g<1>9'),
            (r'(?<=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1(?=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', 'i'),
            (r'(\d)S([A-Z])', r'\g<1>5\g<2>'),
            (r'\s4([a-zа-яà-öø-ÿ])', r' h\g<1>'),
            (r'(\d)\s*/\s*(\d)', r'\g<1>/\g<2>'),
            # (r'([^\d\s])(\d+[\s\-–—]?\d+)([^\d\s])', r'\g<1> \g<2> \g<3>'),
            # (r'(\d+[\s\-–—]?\d+)([^\d\s])', r'\g<1> \g<2>'),
            # (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])(\d+\s*[-–—]?\s*\d+)([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1> \g<2> \g<3>'),
            # (r"r'", "7"),
            # (r'(\s\d)\s*(\d[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>-\g<2>'),
            ]
    for pattern, repl in subs:
        text = re.sub(pattern, repl, text)
    for old, new in CHAR_MAP.items():
        text = text.replace(old, new)

    def remove_9(match):
        A = match.group(1)
        sep = match.group(2)
        B = match.group(3)
        # удаляем 9 только если длина B на 1 больше A
        if len(B) - len(A) == 1:
            return f"{A}{sep}{B}"
        else:
            return match.group(0)  # оставляем как есть
    pattern9 = r'(\d+)([-\s])(\d+)9'
    text = re.sub(pattern9, remove_9, text)

    def remove_9_after_dash(match):
        A = match.group(1)  # число до дефиса/пробела
        sep = match.group(2)  # дефис или пробел
        B = match.group(3)  # число после дефиса/пробела
        letters = match.group(4)  # буквы после числа

        # удаляем 9 только если B длиннее A на 1
        if len(B) - len(A) == 1:
            return f"{A}{sep}{B}{letters}"
        else:
            return match.group(0)  # оставляем как есть

    pattern = r'(\d+)([-\s])(\d+)9([A-ZА-ЯÀ-ÖØ-Þa-zа-яà-öø-ÿ]+)'

    text = re.sub(pattern, remove_9_after_dash, text)

    # text = re.sub(r'^.\.y\.\s*', '', text, flags=re.MULTILINE)
    return text

def process_text(text, trlit: bool = True):
    """Возвращает текст без мусора ОСR"""
    lines = text.splitlines(keepends=True)  # сохраняем \n
    processed_lines = [cleaning_from_ocr(line, trlit) for line in lines]
    return ''.join(processed_lines)


def process_text_last(text: str, lines_dict: Dict[int, str]) -> Tuple[List[str], List[str]]:
    # очистка от мусора текста
    # text = process_text(text, False)
    # форматирует диапазоны пробелами
    text = clear_from_ocr_for_text_last(text)
    print("форматирует диапазоны пробелами")
    print(text)
    # упорядочивает по порядку последовательности диапазонов
    text = clear_from_ocr_for_text(text)
    print("упорядочивает по порядку последовательности диапазонов")
    print(text)

    # range_pattern = re.compile(r'(\d{1,3})(?:\s*[-–—]\s*(\d{1,3}))?')
    # шаблон диапазона
    range_pattern = re.compile(r'\b(\d{1,3})\s*[-–—]\s*(\d{1,3})\b|(?<!\d)[-–—]\s*(\d{1,3})\b')
    # разделение текста по диапазонам
    matches = list(range_pattern.finditer(text))

    dict_results: List[str] = []
    text_results: List[str] = []
    if not isinstance(lines_dict, dict):
        print(repr(lines_dict))
        raise TypeError(f"Ожидался dict, получен {type(lines_dict)}")

    if not matches:
        merged = " ".join(lines_dict.values())
        # удаление из строк словаря нумерации(это, вероятно, уже сделано ранее)
        cleaned = re.sub(r'\d{1,2}[\.:]\s*', '', merged)
        dict_results.append(cleaned)
        text_results.append(text.replace("\n", " ").strip())
        return dict_results, text_results
    # перебор текста перевода по диапазонам
    for i, match in enumerate(matches):
        # начало дапазона
        start = int(match.group(1))
        # конец диапазона
        # end = int(match.group(2)) if match.group(2) else start + 1
        end = int(match.group(2)) if match.group(2) else start
        # для случая типа 1-4 5-6 7-8
        keys = range(start, end+1)
        # # для случая типа 1-4 4-7 7-10
        # keys = range(start, end)
        # пропуск тех строк транслитерации, что отсутствуют в диапазонах перевода
        # или присутствуют в неполном количестве
        if not all(k in lines_dict for k in keys):
            continue

        dict_results.append(" ".join(lines_dict[k] for k in keys))
        # сбор текста из участков с диапазонами,
        # которым соответствуют имеющиеся транслитерации
        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        # fragment = text[text_start:text_end].strip(" ()")
        fragment = text[text_start:text_end].strip()
        text_results.append(fragment)

    return dict_results, text_results




def _extract_number(text: str) -> int:
    """Извлекает первое число из строки."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


def _restore_sequence(anchors: List[Tuple[int, int]], total_length: int) -> List[int]:
    """Восстанавливает номера по якорям."""
    result = [None] * total_length

    if not anchors:
        return list(range(1, total_length + 1))

    # ДО первого якоря
    first_idx, first_num = anchors[0]
    for i in range(first_idx, -1, -1):
        result[i] = first_num - (first_idx - i)

    # МЕЖДУ якорями
    for (i1, n1), (i2, n2) in zip(anchors, anchors[1:]):
        result[i1] = n1
        for i in range(i1 + 1, i2):
            result[i] = n1 + (i - i1)
        result[i2] = n2

    # ПОСЛЕ последнего якоря
    last_idx, last_num = anchors[-1]
    for i in range(last_idx, total_length):
        result[i] = last_num + (i - last_idx)

    return result


def renumber_trust_source(text: str) -> Dict[int, str]:
    """
    Преобразует текст с частичной нумерацией (кратной 5)
    в словарь {номер: строка}.

    Поддерживаются форматы:
    - 10.
    - 10:
    - (10)
    - (abc10xyz)
    - 10)
    - 10'

    Если строка не содержит номера,
    номер восстанавливается по ближайшим якорям.
    """
    if not text.strip():
        return {}

    pattern_line_start = re.compile(r'^\s*(\d+)\s*[.:]')
    inline_patterns = [
        r'\(([^)]*\d+[^)]*)\)',
        r'(\d+)\)',
        r'(\d+)\'',
    ]

    lines = text.splitlines()

    # --- РЕЖИМ 1: текст уже разбит на строки
    if any(pattern_line_start.match(line) for line in lines):
        anchors: List[Tuple[int, int]] = []

        for idx, line in enumerate(lines):
            m = pattern_line_start.match(line)
            if m:
                num = _extract_number(m.group(0))
                if num % 5 == 0:
                    anchors.append((idx, num))

        numbers = _restore_sequence(anchors, len(lines))

        result: Dict[int, str] = {}
        for num, line in zip(numbers, lines):
            content = pattern_line_start.sub('', line, count=1).strip()
            result[num] = content

        return result

    # --- РЕЖИМ 2: сплошной текст
    for pat in inline_patterns:
        compiled = re.compile(pat)
        matches = list(compiled.finditer(text))
        if not matches:
            continue

        segments = []
        anchors: List[Tuple[int, int]] = []

        for i, match in enumerate(matches):
            num = _extract_number(match.group(0))
            start = match.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            segments.append(content)

            if num % 5 == 0:
                anchors.append((i, num))

        numbers = _restore_sequence(anchors, len(segments))

        return {num: seg for num, seg in zip(numbers, segments)}

    return {1: text.strip()}

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


def normalize_for_mt(text: str) -> str:
    # 0. Базовая очистка (translate-таблица уже применяется снаружи)
    a = text
    chars_to_remove = "!?/:.<>™‰˹˺[]⅁ᲟᲠᲢ"
    table = str.maketrans("", "", chars_to_remove)
    # удаление ненужных символов
    a = a.translate(table)
    normalize_gaps(a)
    # 4. Удаляем редакторские маркеры
    a = re.sub(r"^Pl-/\s*", "", a)  # Pl-/
    a = a.replace("\\", "")  # перенос строки
    a = a.replace(",", "")  # маркер переноса строки
    # номера строк
    a = re.sub(r"^\s*\(\s*\d+\s*(?:[-–]\s*\d+)?\s*\)\s*", "", a)
    # каталожные таблички
    a = re.sub(r"\(\s*[A-Z]\.\s*\d+\s*\)","", a)

    # удалить редакторские параграфы
    a = re.sub(r"\b§{1,2}\s*\d+\b", "", a)
    # 1. ASCII → Unicode
    for old, new in CHAR_MAP.items():
        a = a.replace(old, new)

    # 2. Надстрочные детерминативы
    a = normalize_akkadian_determinatives(a)

    # 3. Подстрочные цифры
    a = normalize_subscripts(a)


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
    if total_translation == 0:
        return translit_text.strip() + " " + marker

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


# -------------------------------------------------------------------------------------
text_trlit = """55
su-ma ta-dâ-ga-lct-ma
e-ti-i15
K. 10. si-im-tcim sa-6,i-qi-lâ-su
A. y.
a-hu-a a- tù-nu
i4-da-ma KÙ. BABBAR
sa..â -qi-lâ-su gis-im-14-ni
A-sur li-tù-ul
15. ki-ma sa a-tù-nu
ta-qt-sa-ni-su-ni
li-bi4 lu i-lza-du
ù té-er-ta-ku-nu
li-li-kam-ma
20. ù a-na-ku a-sar
K.
KÙ. BABBARPL-ku-nu I GÎN.TA
û-kâc-lu-ni
S. K.
ld ..4i-id-ma
lu-sa-âs-qi-lâ-ku-nu-ti
25. gis-im-lâ-ni
"""

text_translate = """ı-3İlabrat-bäni, Uşur-sa-İstar ve Amur-ili'ye şöyle söylüyor: 4-5Sizler kardeşlerim (ve)
efendilerimsiziniz, dikkat ediniz! 5-72 1/2 mina tasfiye edilmiş gümüşü ve faizini Hubâbum'a
tarttiriniz. 8-9Eğer siz bana (iyi nazarla) bakarsanız ben de (size karşı) yiğitçe davranırım.ıo
Ona bedeli tarttiriniz. ı ı - ı2 Sizler kardeşlerimsiniz, dikkat ediniz ve ı2- ı 3parayı tarttiriniz
ve hoşnutluğumu kazaniniz. "'Tanrı Asur şâhit olsun ki sizin onu bana hediye etmeniz
(halinde) kalbim gerçekten memnun olacaktır.18-23Haberiniz buraya gelsin ve ben paranı-
zın yerine l'er segel tutmaya dikkat edeyim ve24-25size tarttırsınlar, sevgimi kazaniniz!
"""
# работа в первом блоке
text_trlit = cleaning_from_ocr_prelim(text_trlit)
text_translate = cleaning_from_ocr_prelim(text_translate)
print("После предварительной чистки")
print(text_trlit)
print(text_translate)
text_trlit = process_text(text_trlit)
text_translate = process_text(text_translate)
print("После основной чистки")
print(text_trlit)
print(text_translate)
dict_trlit = renumber_trust_source(text_trlit)
print("После подготовки к добавлению в список")
print(dict_trlit)
print(text_translate)
# работа во втором блоке
list_trl_transl = process_text_last(text_translate, dict_trlit)
print("После создания списка")
for el in list_trl_transl:
    print(el)
# окончательная чистка и обработка
if isinstance(list_trl_transl, tuple):
    accad_str_arr = list_trl_transl[0]
    translate_str_arr = list_trl_transl[1]
num_i = 1
for translate_str, accad_str in zip(translate_str_arr, accad_str_arr):
    # 1. Очистка перевода
    t = translate_str.replace("\n", " ")

    # 2. Очистка аккадского
    a = accad_str.replace("\n", " ")
    a = normalize_for_mt(a)

    # # 3. Токенизация перевода
    t_sentences = sent_tokenize(t)
    # # --------------------------------------------------------------
    # # # 3. Токенизация перевода
    # t_sentences = sent_tokenize(t)
    # t_sentences = [sent for sent in t_sentences if looks_like_real_translation(sent)]
    # # определение языка и перевод на английский, если перевод не английский\n",
    # t_sentences = [translate_to_english(sent) if detect_language(sent) != 'en' else sent for sent in t_sentences]
    # # ---------------------------------------------------------------------------
    # 4. Выравнивание + маркеры
    a = align_and_mark_sentences(a, t_sentences, marker="<sent>")
    # print("После разделения транслитерации на предложения")
    # print(a)
    # print(t_sentences)

    # 5. Склеиваем перевод обратно
    t = " ".join(t_sentences)

    # 6. CSV-экранирование (ОДИН РАЗ!)
    a = a.replace('"', '""')
    t = t.replace('"', '""')
    print(f"\nТранслитерация {num_i}\n {a}")
    print(f"\nПеревод {num_i}\n {t}")
    print("-" * 50)

# for el in list_trl_transl:
#     print(el)