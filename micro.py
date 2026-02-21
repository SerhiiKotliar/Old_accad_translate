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
        (r'([a-z])ı\s*', r'\g<1>i'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])ı([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>i\g<2> '),
        (r'\s*ı\s*ı', ' 11'),
        (r'\s*ı\s*(\d)', r' 1\g<1>'),
        (r'o', '0'),
        (r'ı', '1'),
        (r'4ssur', r'Assur'),
        (r'(\w)[-–—]1(\w)', r'\g<1>-i\g<2>'),
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
        (r'^\n', ''),
        # (r'^\d+\n', ''),
        (r"(\d{1,2})'[-–—]\s*(\d{1,2})",r'\g<1>1-\g<2>'),
        (r"[-–—]'(\d{1,2})", r'-\g<1>'),
        (r"\s'(\d)\s*[-–—]", r' 1\g<1>-'),
        (r'(\w)1(\w)', r'\g<1>i\g<2>'),
        (r'K Ù\.', r'KÙ\.'),
        (r'\"\'\"', ''),
        (r'^(\d+\.)\r?\n?', r'\g<1>'),
        (r'\s[ÖO](?=[A-ZÀ-ÖØİŞĞÇÜ])', r'0 '),
        (r'(\d{1,2}\s*)\'(\s*\d{1,2})', r'\g<1>-\g<2>'),
        # (r'\s*i0\s*', r' 10 '),
        # (r'(?<=[^\W_]):(?=[^\W_])', ' '),
        # (r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', ''),
        # (r'§', 'S'),
        # (r'\,', ' '),
        # (r'^.\.y\.\s*', ''),
        # (r'^.\.y\.\n', ''),
    ]
    for pattern, repl in subs:
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)

    PATTERN: re.Pattern[str] = re.compile(
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
        r'([^\d])(\d)\s(\d)-(\d{2})([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])'
    )

    def conditional_replace1(match: re.Match[str]) -> str:
        left: int = int(match.group(2) + match.group(3))
        right: int = int(match.group(4))

        if left < right:
            return f"{match.group(1)}{left}-{right}{match.group(5)}"

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


text ="""Appendix 1:
Appendix 1.1
Selected references to guides (rādium)
Kt 87/k 479: 4-10
Ú-lu-ma-il5 ra-dí-ú-um ša GAL sí-ki-tim
a-mì-ša-am i-la-kam ak-lam ma-la ù šé-ni-šu
ša-ki-il5-šu
Ulumail, the guide of the rabi sikkitim will
go there, pay him food a few times
Kt 94/k 628: 4-7
ŠÀ.BA 30 ku-ta-ni a-na Šu-pu-nu-ma-an ni-
dí-in 5 TÚG a-na ra-dí-im Iš-hi-a-ni-tí-im
Thereof, we gave 30 kutānu-textiles to
Šupunuman. 5 to the guide from Išhianit.
Kt 94/k 760: 10-22
2 1/2 GÍN ig-ru-šu a-na ra-dí-im ša Ha-tim
ni-dí-in
His wages were 2 1/2 shekels for the guide
from Hattum (PN).
Kt 94/k 1126: 4-8
mì-šu ša [a]-na-kam ra-dí-ú e-ta-wu-ú um-
ma šu-nu-ma ṣú-ha-ru-kà ú-ṭá-tám a-ba-kam
lá i-mu-ú
Why is it that the guides have been arguing
here, saying: ‘Your servants refuse to ship
the grain.’
AKT 1, 39b: 14-16
IGI Lu-ùh-ra-ah-šu ra-dí-ú ša Ha-ra-áš-tal)
Witness: Luhrahšu, the guide of Haraštal.
AKT 2, 24: 13-16
3 TÚG.HI.A ša 10 GÍN.TA a-na a-wi-il-tim
dí-na-ma ra-dí-e lu tù-lá-bi4-iš.
Give 3 textiles, worth 10 shekels each, to the
lady so she can dress the guide.
BIN 4, 203: 12-14
1/2 GÍN ší-im ki-ri-im ša a-na ra-dí-im.
Half a shekel (of silver was) the price of the
jar that was for the guide.
BIN 6, 122
lu ku-ta-nam [x x] lu ša a-ki-dí-e [(x x)]
ma-ma-an lá il5-t[a-na-qé] ú a-na ra-dí-im ú
DUB.SAR TÚG.HI.A iš-[té-e]t ú AN.NA 1
ma-na lá ta-da-na a-wa-at É-GAL-lim da-
[na].
Nobody is to take any kutānu-textiles, any
[…] or any Akkadian textiles, and you (pl.)
should not give a single piece of textile or a
single mina of tin to the guide or the scribe.
The orders from the palace are strict.
C 16
12 TÚG ku-ta-ni ša Puzur4-A-šur e-zi-ba-
ku-ni TÚG.HI.A a-na En-na-nim DUMU
A-bi4-a pí-qí-id-ma ù šu-ut a-na ra-dí-e li-dí-
nu-šu-ma lu-ub-lu-nim.
The twelve kutānu-textiles that Puzur-Aššur
left behind with you – assign those textiles
to Ennam-Anum, son of Abia, so that he
personally can give them to the guide and
they can bring them here.
CCT 1, 29
[1/]3 ma-na 2 GÍN KÙ.BABBAR ší-im [1
TÚG] ku-ta-nim ša a-ra-dí-im ni-dí-nu.
22 shekels of silver, the price of the kutānu-
textile that we gave to the guide.
CCT 2, 19b
ki-ma ší-lá-tám lá i-šu-ú É.GAL-lúm ra-dí-šu
a-na pá-tí iš-pu-ra-ma a-na Wa-ah-šu-ša-na
a-tù-ar. (Collations by M. Trolle Larsen).
Since he did nothing wrong, the palace wrote
to its guide at the frontier and I will return to
Wahšušana
CCT 5, 3b
té-er-ta-kà a-na Za-al-pá i-li-kà-ni um-ma
a-ta-ma a-ma-kam lá wa-áš-ba-tí ra-dí-ú
a-ma-kam e-mu-ru-kà-ma i-a-um a-na-kam
li-bi e am-ra-aṣ a-na Té-ga-r[a-ma] e-tí-iq-
ma i-na Té-ga-ra-ma lu wa-áš-ba-tí
Your message reached Zalpa, in which you
said: ‘You should not stay there. The guides
will see you there, and I refuse to worry here.
Go to Tegarama and stay in Tegarama!’
Kt c/k 204
2/3 GÍN KÙ.BABBAR a-na 2 ki-re-en a-na
ra-dí-e ša iš-tí-a i-li-ku-<ni> áš-qúl
I paid 2/3 shekels for two jars for the guides
who went with me.
Kt c/k 441
4 qá-tí a-na ra-dí-e ... mì-ma a-nim i-na Bur-
hi-im a-dí-in
Four … to the guides … all this I gave in
Burhum

"""


pattern1 = re.compile(pattern, re.MULTILINE)
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
