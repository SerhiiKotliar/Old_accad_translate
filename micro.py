import re
from typing import Dict

pattern = r'^[A-Z]{1,3}[a-z]{1,2}\s*(?:\d{1,3}/k|n/k|\d{1,2}\,)\s*\d{1,4}[a-z]?(?::\s*\d+[–\-]\d+)?\n'

text_translate = ""
text_transliterate = ""
flag_vyp = False

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
# Стоп-слова для фильтрации английского, немецкого и турецкого текста - расширенный список
FOREIGN_WORD_RE = re.compile(
    r"\b("
    # Немецкие слова
    r"Jetzt|ist|gerade|ein|Brief|der|das|mit|"
    r"für|auf|aus|bei|nach|über|unter|zwischen|durch|wegen|"

    # Английские слова
    r"desk|bound|conducted|"
    r"beneath|beside|"
    r"except|outside|throughout|toward|"
    r"pour|avec|sans|entre|vers|"
    r"senza|tra|verso"
    r")\b",
    re.I
)

TRANSLIT_LINE_RE = re.compile(r'''
^(?!\s*\d+\s*$)            |   # не начинается с чистого номера
(?=.*(
        -[a-z]            |   # дефисная слоговая морфология
        \d                |   # индексные цифры (Puzur4)
       \b(?:DINGIR|LUGAL|EN|NIN|DUMU|SAL|MUNUS|GURUŠ|LU₂|AMA|AB|AḪ|ŠEŠ|NIN₉|E₂|KI|URU|KUR|ABZU|A|IM|UD|U₄|ITI|MU|GIŠ|DU₃|GAR|GUB|TUKU|ŠU₂|ZI|NAM|ME|ŠU|IGI|DIŠ|MIN|EŠ|LIMMU|IA|KIŠIB|LÚ|AŠ|ŠA|BABBAR|KÙ|NUMUN|SU.|U.BA|TUG|NIGIN|GIN|KÙ.|TA)\b  # формулы / логограммы
        [šḫṭṣ]            |   # диакритика
))
(?!.*[.,;:!?])                # нет пунктуации перевода
(?!.*\b[A-Z]?[a-z]{3,}\b\s+\b[A-Z]?[a-z]{3,}\b)       # нет нормального текста
[A-Za-zúēīāíšḫṭṣŠÍÚḪṮṢ0-9.\[\] \?\!§⅀⅁ℵᲟᲠᲢ–\- ]+
$
''', re.VERBOSE)

# Морфемные разделители (дефис или ℵ)
MORPHEME_SEP_RE = re.compile(r"[-ℵ]")

AKKADIAN_INDICATOR_RE = re.compile(
    r"[ŠšḪḥṢṣṬṭʾʿ⅀⅁ᲟᲠ]|"
    r"[₀₁₂₃₄₅₆₇₈₉]|"
    r"[ᵈᵐᶠᵏ]|(?:\{[dmfkg]\})|"
    r"\b[A-Z]{2,}(?:\.[A-Z]{2,})+\b|"
    r"\b[A-Z]{2,}[-ℵ][a-z]+\b|"
    r"\[.*?\]|\(.*?\)|\{.*?\}|"
    r"\b[A-Z][a-zšḫṭṣ]+[-ℵ][a-zšḫṭṣ]+\b|"
    r"\b[a-zšḫṭṣ]+[-ℵ][a-zšḫṭṣ]+\b|"
    r"\b\d+[rv]\b|"
    r"x\+|x\-|x\?|x=\d+|"
    r"\.\.\.|…|"
    r"\d+['ˈ]|"
    r"–[^ ]|"
    r"\|"
)


NOT_TRANSLIT_RE = re.compile(
    r"\b[A-Z][a-z]{3,} [A-Z][a-z]{3,}\b|"        # Два заглавных слова подряд
    r"\b[a-z]{4,} [a-z]{4,} [a-z]{4,}\b|"        # Три длинных слова подряд
    r"^\d+ [A-Z][a-z]|"                          # Цифра + заглавное слово
    r"[a-z]{5,}[-ℵ][a-z]{4,}(?![šḫṭṣʾʿ])|"       # англ. дефис/ℵ (НО не аккад.)
    r"[a-zA-ZäöüÄÖÜß]{5,}[-ℵ][a-zA-ZäöüÄÖÜß]{4,}|" # нем.
    r"[a-zA-ZçğıİöşüÇĞİÖŞÜ]{5,}[-ℵ][a-zA-ZçğıİÖŞÜ]{4,}|" # тур.
    r", |; |: |\. [A-Z]|"                        # Пунктуация
    r"\b(?:[A-Za-z]+ ){3,}[A-Za-z]+\b"           # 3+ слов подряд
)

# Разделители блоков
SEPARATOR_RE = re.compile(r'^-+$')



def cleaning_from_ocr_prelim(text: str) -> str:
    text = re.sub(
        r'^\s*(?:S\.(?:\s*K\.)?|K\.(?:\s*)?|v|\. v)\s*',
        '',
        text,
        flags=re.MULTILINE
    )
    text = re.sub(r'^\w\.\s*K\.\s*\w+', '', text, flags=re.MULTILINE)
    subs = [
        (r'([a-z])ı\s*', r'\g<1>i'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])ı([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>i\g<2> '),
        (r'\s*ı\s*ı', ' 11'),
        (r'\s*ı\s*(\d)', r' 1\g<1>'),
        (r'o', '0'),
        (r'ı', '1'),
        (r'4ssur', r'Assur'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])[-–—]1([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>-i\g<2>'),
        (r'\s5([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' S\g<1>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])15([-–—])', r' \g<1>lš\g<2>'),
        (r'\s0([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' O\g<1>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])0([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>O\g<2>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])5([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>S\g<2>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>i\g<2>'),
        (r'A1', 'Ai'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1\s', r'\g<1>i '),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]),(\d)', r'\g<1> \g<2>'),
        (r'\s[iI]\s?(\d+)', r'1\g<1>'),
        (r'(?<![A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])l\s*(\d)', r' 1\g<1>'),
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
        (r'(\s\d+)\s(\d+)[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]', r'\g<1>-\g<2> '),
        (r'([^\d,\s])(\d+[\s\-–—]\d+)([^\d,:\n])', r'\g<1> \g<2> \g<3>'),
        (r'(\d+[\s\-–—]\d+)([^\d,:\s*])', r'\g<1> \g<2>'),
        (r'(\d+)\s*-\s*(\d+)', r'\g<1>-\g<2>'),
        (r'^.\.?\s?y\.\s?\r?\n?', ''),
        (r'(^\d,\s)(\d{1,2})\s(\d{1,2})(^\d\n)', r'\g<1> \g<2>-\g<3> \g<4>'),
        (r'\,\n', ''),
        (r'^\n', ''),
        (r"(\d{1,2})'[-–—]\s*(\d{1,2})",r'\g<1>1-\g<2>'),
        (r"[-–—]'(\d{1,2})", r'-\g<1>'),
        (r"\s'(\d)\s*[-–—]", r' 1\g<1>-'),
        (r'(\w)1(\w)', r'\g<1>i\g<2>'),
        (r'K Ù\.', r'KÙ\.'),
        (r'\"\'\"', ''),
        (r'(\d)i(\d)', r'\g<1>1\g<2>'),
        (r'^\d+\r?\n(?=Kt)', ''),
        # (r'^(\d+\.)\r?\n?', r'\g<1>'),
        (r'\s[ÖO](?=[A-ZÀ-ÖØİŞĞÇÜ])', r'0 '),
        (r'(\d{1,2}\s*)\'(\s*\d{1,2})', r'\g<1>-\g<2>'),
    ]
    for pattern, repl in subs:
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)
    # r'(\D\s*)(\d{1,2})\s+(\d{1,3})(\s*[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])'
    PATTERN: re.Pattern[str] = re.compile(
         r'(^\d,\s*)(\d{1,2})\s+(\d{1,3})(\s*[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])'

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

    def conditional_replace(match: re.Match[str]) -> str:
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

    PATTERN1: re.Pattern[str] = re.compile(
        r'([^\d,\s])(\d)\s(\d)-(\d{2})([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])'
    )

    def conditional_replace1(match: re.Match[str]) -> str:
        left: int = int(match.group(2) + match.group(3))
        right: int = int(match.group(4))

        if left < right:
            return f"{match.group(1)} {left}-{right} {match.group(5)}"

        return match.group(0)  # why: preserve original if condition fails

    def process_text1(text: str) -> str:
        return PATTERN1.sub(conditional_replace1, text)
    text = process_text1(text)

    pattern3 = re.compile(r'^(\d+)\n', re.MULTILINE)

    def repl(match: re.Match) -> str:
        num = int(match.group(1))
        if num % 5 == 0 and num <= 40:
            return f"{num}."
        return ""  # иначе удаляем

    text = pattern3.sub(repl, text)

    pattern2 = re.compile(r'(\d\s*[-–—]?\s*)[\"“”«»„‟]([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])')

    def replace_func2(match):
        # первая группа: убрать пробелы, оставить цифру и тире
        first = re.sub(r'\s+', '', match.group(1))
        return f"{first}11 {match.group(2)}"

    text = pattern2.sub(replace_func2, text)

    return text

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
            (r'\:', ' '),
            (r'\!', ''),
            (r'\?', ''),
            (r'\/', ''),
            (r"\'", ''),
            (r'\"', ''),
            (r"\'\'", ''),
            (r'<([^<>]+)>', r'\g<1>'),
            (r'^.\d{1,}\n', ''),
            (r'^.\.?\s?y\.\s?\r?\n?', ''),
            (r'(?<=[A-Za-z0-9]):(?=[A-Za-z0-9])', ' '),
            # (r'([^\W\d_])4(-|[^\W\d_])', r'\g<1>h\g<2>'),
            # (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\g<1>h'),
            (r'(.)\,(.)', r'\g<1>\g<2>'),
            (r'\,\n', ''),
            (r'\\', ''),
            (r'\s*i0\s*', r' 10 '),
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
            (r'\:', ' '),
            (r'\!', ''),
            (r'\?', ''),
            # (r'\/', ''),
            (r"\'", ''),
            (r'\"', ''),
            (r"\'\'", ''),
            # (r'([^\W\d_])4(-|[^\W\d_])', r'\g<1>h\g<2>'),
            # (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\g<1>h'),
            # (r'9([A-ZА-ЯÀ-ÖØ-Þ])([a-zа-яà-öø-ÿ])', r'\1\2'),
            (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])([-\s])9([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>\g<2>\g<3>'),
            # (r'(\d{1,2}[-\s]\d{1,2})\$', r'\g<1>9'),
            (r'(?<=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1(?=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', 'i'),
            (r'(\d)S([A-Z])', r'\g<1>5\g<2>'),
            (r'\s4([a-zа-яà-öø-ÿ])', r' h\g<1>'),
            (r'(\d)\s*/\s*(\d)', r'\g<1>/\g<2>'),
            (r'\s*i0\s*', r' 10 '),
            (r'(\d{1,2}\s*)\'(\s*\d{1,2})', r'\g<1>-\g<2>'),
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

def extract_transliteration(text) -> list:
    """
    Извлекает блоки транслитерации из текста с их проверкой.
    Склеивает строки, оканчивающиеся на - или ℵ с последующей.
    Возвращает список блоков.
    """
    if isinstance(text, list):
        text = "\n".join(text)
    # очистка от мусора OCR
    text = cleaning_from_ocr(text)
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
        is_transliteration = False
        # Проверка 1: Соответствует ли базовому формату транслитерации?
        has_basic_format = (
                TRANSLIT_LINE_RE.match(line_trimmed) and
                MORPHEME_SEP_RE.search(line_trimmed)
                if len(line_trimmed) > 25 else TRANSLIT_LINE_RE.match(line_trimmed)
        )
        if has_basic_format:
            is_transliteration = True
        # if not has_basic_format:
        #     if current:
        #         blocks.append("\n".join(current).strip())
        #         current = []
        #     continue

        # Проверка 2: Содержит ли иностранные слова?
        has_foreign_words = FOREIGN_WORD_RE.search(line_trimmed)

        # Проверка 4: Содержит ли признаки аккадской транслитерации?
        has_akkadian_indicators = AKKADIAN_INDICATOR_RE.search(line_trimmed)
        # if not has_foreign_words and has_basic_format and has_akkadian_indicators:
        #     if ':' in line_trimmed:
        #         line_trimmed = line_trimmed.replace(":", "")
        # Проверка 3: Содержит ли явные признаки НЕ транслитерации?
        is_not_translit = False
        if not is_transliteration:
            is_not_translit = NOT_TRANSLIT_RE.search(line_trimmed)
        # else:
        #     is_not_translit = False

        # # Короткие токены (ša, ina, a-na и т.п.)
        is_tokens = False
        tokens = re.findall(r"[A-Za-zšṣṭḫʾʿ]+", line_trimmed.lower())
        if tokens:
            long_tokens = [t for t in tokens if len(t) >= 4]
            short_tokens = [t for t in tokens if len(t) <= 3]
            llt = len(long_tokens)
            lst = len(short_tokens)
            lt = len(tokens)
            des = lst/lt
            if des > 0.6:
            # if llt > 0:
            #     lline = len(line_trimmed)
            #     if lline/llt > 15:
                is_tokens = True
        else:
            is_tokens = False

        # Логика принятия решения:
        # 1. Должен быть базовый формат
        # 2. Не должен содержать иностранных слов ИЛИ должен иметь аккадские индикаторы
        # 3. Не должен быть явно НЕ транслитерацией
        is_transliteration = (
                has_basic_format and is_tokens and
                (not has_foreign_words or has_akkadian_indicators) and
                not is_not_translit
        )

        # # Особый случай: если есть аккадские индикаторы, принимаем даже с некоторыми иностранными словами
        # if has_akkadian_indicators and has_basic_format and not is_not_translit:
        #     is_transliteration = True

        # num_morf = text.count("ℵ")
        # num_defis = text.count('-')
        # num_div = max(num_morf, num_defis)
        # # мало дефисов в строке
        # if (num_div > 0 and len(text) / num_div - 1 > 12) or num_div == 0:
        #     is_transliteration = False
        num_morf = line.count("ℵ")
        num_defis = line.count('-')
        num_div = num_morf + num_defis
        # мало дефисов в строке
        if (num_div > 0 and len(line) / num_div - 1 > 16) or num_div == 0 and not has_basic_format:
            is_transliteration = False
        # проверка количества цифр в строке
        def more_than_half_digits(line_trimmed):
            digits = sum(ch.isdigit() for ch in line_trimmed)
            return digits > len(line_trimmed) / 2

        if more_than_half_digits(line_trimmed):
            is_transliteration = False

        if is_transliteration:
            current.append(line_trimmed)
        else:
            break
            # if blocks:
            #     return blocks
            # return []
            # if current:
            #     blocks.append("\n".join(current).strip())
            #     current = []

    if current:
        blocks.append("\n".join(current).strip())

    return blocks


def find_translit_by_rows(text: str, pos: int=0, n_dop: int=2):
    """поиск транслитерации начиная с позиции после якоря по строкам
    возвращает транслитерацию или "" и её позиции конца и начала"""
    pos_end_of_line = 0
    pos_start_trlit = pos
    result = ""
    num_row = 0
    while pos < len(text):
        # строка от её первой позиции и позиция конца строки
        n_l, pos_end_of_line = get_next_line(text, pos)
        # прекращение поиска транслитерации после 2 ложных строк
        if num_row > n_dop-1:
            return "", pos_end_of_line, pos_start_trlit
        line_trl = []
        if n_l:
            line_trl = extract_transliteration(n_l)
        end_translit = 0
        # pos_start_trlit = pos
        while line_trl:
            # сборная транслитерация
            result += "\n".join(line_trl) + "\n"
            end_translit = pos_end_of_line
            pos_start_trlit = pos_end_of_line - len(n_l) - 1
            # строка
            n_l, pos_end_of_line = get_next_line(text, pos_end_of_line)
            if pos_end_of_line == -1:
                return result, end_translit, pos_start_trlit
            if n_l:
                line_trl = extract_transliteration(n_l)
            else:
                line_trl = ""
            # pos = pos_end_of_line
        num_row += 1
        pos = pos_end_of_line
        if result:
            return result, end_translit, pos_start_trlit


    return "", pos_end_of_line, pos_start_trlit



def process_text(text, trlit: bool = True):
    """Возвращает текст без мусора ОСR"""
    lines = text.splitlines(keepends=True)  # сохраняем \n
    processed_lines = [cleaning_from_ocr(line, trlit) for line in lines]
    return ''.join(processed_lines)


text ="""17. With the exception of ownership of investments in
joint-stock funds; see further below.
18. See for instance AKT 2, 57: 5-14, a letter to a lady in
Assur from her brother in Anatolia, where he writes: "My
dear sister, my dear lady, if there is any silver there of our
father's house, then satisfy Idaya's son. If you do not wish to
(do that), then let Pilah-Assur sell my house there to pay off my
creditor" (a-ha-ti a-ti be-el-ti a-ti a-ma-kam i-na KU.BABBAR
s"a E a-bi4-ni su-ma i-ba-si DUMU I-da-a-a tà-i-bi4 su-ma la
li-bi4-ki a-[ma]-kam É-ti-a Pi-la-ah-A-sùr li-di-ma tam-kà-ri
lu-sa-bi4). The situation seems to indicate that the father was
dead, and the request to spend money available in his house
may have been somewhat irregular, since noone could know,
presumably, to whom the money belonged. The very rare ref-
erences to debts of the paternal house (see for instance BIN 4,
83: 33-37) are probably in all instances to be placed in the
context of inheritance problems.
19. See Larsen (2002,168: 3-7), where Assur-nada writes
to the lady Abaya: "In accordance with the missive I sent to
you Innaya has here discussed-with the customers in your
name and in the name of your father's house" (a-ma-la
na-as-pè-er-tim sa as-pu ra-ki-ni a-na-kam I-na-a a s"u ma ki it a-na
su-mi É a-bi-ki tam-kà-ri e-ta-uru; kt 94/k 1742: 26-29: "I also
entrusted 40 minas of refined silver that was in the name
of my father's house to Pilah-Istar" (u a-ha-ma 40 ma-na
KU.BABBAR sa-ru-pa-am sa a-na su-mi É a-bi4-a a-na
Pl-/
la-ah-Is\ [tar] ap-qi-id);orkt
a/k 1030: 1-5: "Out of Idin-Suen's
copper Tab-Assur received 13 talents 20 minas of poor copper
on behalf of his father's house and Alili" (i-na URUDU sa
I-di-Su-in 13 GU 20 ma-na URUDU la-mu-nam DU 10-A-sur ki-ma
É a-bi-su ù [A] li li il5 qe).
trader's possibility to function independently in
the commercial system. It was the contractual
foundation for an arrangement where a group of

"""

text = cleaning_from_ocr_prelim(text)
pattern1 = r'\d{2,}:\s*(?:\d+[-–—]\d+[:,)]\s*[\s\S]{0,80}?)?\s*"'

pattern1 = re.compile(pattern1, re.MULTILINE)
match = pattern1.search(text, 0)
# предварительная очистка
# text = cleaning_from_ocr_prelim(text)
# ---------------------------------------------------
text_translate = ""
flag_vyp = False
# позиция начала поиска
pos = match.end()
# транслитерация
text_transliterate, pos_end, pos_start = find_translit_by_rows(text, pos)
# if text_transliterate != "":
    # словарь транслитерации ключ номер строки и значение строка
    # text_transliterate = renumber_trust_source(text_transliterate)
# else:
#     return (text_translate, text_transliterate), flag_vyp, len(text)
# ------------------------------------------------------------
if pos_end < len(text):
    pos_start_translate = pos_end
    pattern2 = re.compile(pattern, re.MULTILINE)
    match = pattern2.search(text, pos_start_translate)
    if not match:
        pos_end_translate = len(text)
    else:
        pos_end_translate = match.start()
    text_translate = text[pos_start_translate:pos_end_translate]
    text_translate = process_text(text_translate)
    # if is_translation(text_translate) and looks_like_real_translation(text_translate) and text_transliterate != "":
    #     flag_vyp = True
#     return (text_translate, text_transliterate), flag_vyp, pos_end_translate
# return ("", text_transliterate), flag_vyp, len(text)
