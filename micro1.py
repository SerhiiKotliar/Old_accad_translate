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

def cleaning_from_ocr_prelim(text: str) -> str:
    text = re.sub(
        r'^\s*(?:S\.(?:\s*K\.)?|K\.(?:\s*)?|v|\. v)\s*',
        '',
        text,
        flags=re.MULTILINE
    )
    text = re.sub(r'^\w\.\s*K\.\s*\w+', '', text, flags=re.MULTILINE)
    subs = [
        (r'(?<=[A-Za-z])6(?=(?:-[A-Za-z]))', 'b'),
        (r'(\d+)([A-Za-z])', r'\g<1> \g<2>'),
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
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])0([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>o\g<2>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])5([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>s\g<2>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>i\g<2>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])6([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>b\g<2>'),
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
        (r'¥', 'y'),
        (r'ª', 'a'),
        (r'#', 'h'),
        (r'\st\s*0\s', ' to '),
        # (r'^.*?(\d+)\s*[-–—]\s*(\d+):', r'\g<2>:'),
        (r'(\d+)\s*[-–—]\s*(\d+):', r'\g<2>:'),
        (r'([^\W\d_])4(-|[^\W\d_])', r'\g<1>h\g<2>'),
        (r'(\s\d\s*)4([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>h\g<2>'),
        (r'(\w)4([-–—]\w)', r'\g<1>h\g<2>'),
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
        (r'(\d{1,2}\s*[-–—]\s*)IS', r'\g<1>18'),
        (r"\s'(\d)\s*[-–—]", r' 1\g<1>-'),
        (r'(\w)1(\w)', r'\g<1>i\g<2>'),
        (r'K Ù\.', r'KÙ\.'),
        (r'\"\'\"', ''),
        (r'(\d)i(\d)', r'\g<1>1\g<2>'),
        (r'^\d+\r?\n(?=Kt)', ''),
        # (r'^(\d+\.)\r?\n?', r'\g<1>'),
        (r'\s[ÖO](?=[A-ZÀ-ÖØİŞĞÇÜ])', r'0 '),
        (r'(\d{1,2}\s*)\'(\s*\d{1,2})', r'\g<1>-\g<2>'),
        (r'(\d+)\s*[-–—]\s*(\d+)', r' \g<1>-\g<2> '),
        (r'(\d+)\s*`\s*(\d+)', r' \g<1>-\g<2> '),
        (r'\s*(\d)\s*(\d)\s*[-–—]\s*(\d)\s*s', r' \g<1>\g<2>-\g<3>5 '),
        (r'\s*(\d)\'\'\s*(\d+)\s*', r' \g<1>7-\g<2> '),
        (r'\s*(l)\'\'\s*(\d+)\s*', r' 17-\g<2> '),
        (r'\s*\'\'\s*(\d+)\s*', r' 7-\g<1> '),
        (r'(\d)(^\d+)', r'\g<1> \g<2>'),
        (r'^(\d)\s(\d)(\.)', r'\g<1>\g<2>\g<3>'),
        (r'[-–—]\s*(\d)\s*(\d)\s*', r'-\g<1>\g<2> '),
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

    pattern4 = re.compile(r'(?<![,])(\d+)\s*[-–—]\s*(\d+)(?!:)')
    def replace_func4(match):
        left = int(match.group(1))
        right = int(match.group(2))
        if left <= 9 and len(match.group(2)) > len(match.group(1)):
            ext = len(match.group(2)) - len(match.group(1))
            last_right = match.group(2)[:-ext]
            if int(last_right) > left and int(last_right) - left <= 10:
                return f" {left}-{last_right} {ext}"
        # first = re.sub(r'\s+', '', match.group(1))
        return f" {left}-{right} "

    text = pattern4.sub(replace_func4, text)

    return text

# -----------------------------------------------------------------------------------------

text = """19. See Larsen (2002,168: 3-7), where Assur-nada writes
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
financiers came together to invest in a capital fund
that was entrusted to an individual to function
as the basis for "carrying out trade" (makârum)
during a specified number of years, probably
always a period of at least ten years 20
"""


print(cleaning_from_ocr_prelim(text))