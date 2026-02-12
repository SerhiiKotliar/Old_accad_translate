import re
from typing import Dict, List, Tuple, Match, Pattern

def cleaning_from_ocr_prelim(text: str) -> str:
    text = re.sub(
        # r'^\s*(?:[SK]\.|S\. K\.|S\.K\.|K\.\s*\d|\n|v|\. v)\s*$',
        # r'^\s*(?:[SK]\.|S\. ?K\.|K\.\s*\d+|v|\. v)\s*\r?\n?',
        r'^\s*(?:S\.(?:\s*K\.)?|K\.(?:\s*\d+)?|v|\. v)\s*',
        '',
        text,
        flags=re.MULTILINE
    )
    subs = [
        (r'([a-z])ı\s+', r'\g<1>i '),
        (r'ı\s+ı', '11'),
        (r'ı\s+', '1'),
        (r'ı', '1'),
        (r'\s5([A-Za-zА-Яа-я])', r' S\g<1>'),
        (r'A1', 'Ai'),
        (r'([A-Za-zА-Яа-я])1\b', r'\g<1>i'),
        (r'([A-Za-zА-Яа-я]),(\d)', r'\g<1> \g<2>'),
        # (r'\s(\d)\s(\d)\s', r' \g<1>-\g<2> '),
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
        # (r'\,\n', ''),
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

    # match = PATTERN.search(text)
    # if match:
    #     # Теперь можно отладить вручную
    #     text = conditional_replace(match)

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


text = """1-3Uşur-si-Îstar'a Gallâbum şöyle söylüyor:3-6Senin talimatın gereğince kumaşları seç-
tik ve sana âit 30 kumaş1,7 8İkûppia'nın 18 kumaşini, tüccarın 12 kumaşını, 9- "içinden 5
Abarna kumaşini, Lulu'nun 23 kumaşini, kaşşârum'un 2 kumaşini: ı 2-15Yekûnen 91 kumaş
Iiî
(ile ilgili olarak) kaum Ab~c'nın evi(nde) (hesap) yaptik. ı6 ı85ü-Kûbum'un huzurunda, -
bâni'nin huzurunda, Ili-wédäku'nun huzurunda 19-20işitiyoruz ki, Ileûppia sıhhattedir.
21Kalbin korkmasin (endişelenme). 22-23Burada Idi-Assur'un kumaşlarını seçmedik. 244 sı-
pa, 251 (tane) yük eşeği (de berâberdir)."""



text = cleaning_from_ocr_prelim(text)
print(text)