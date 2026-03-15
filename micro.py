import re
from typing import Dict, List, Tuple

def _extract_number(text: str) -> int:
    """Извлекает первое число из строки."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


def _restore_sequence(
    anchors: List[Tuple[int, int]],
    total_length: int
) -> List[int]:
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



pattern_line_start = re.compile(
    r"""
    ^\s*              # начало строки + пробелы
    \(?                # необязательная открывающая скобка
    \s*               
    (?P<number>\d{1,3})  # число (1-3 цифры)
    \s*               
    (?:               # незахватывающая группа для окончания
        [`'’‘ʼʹˈ]   # любой апостроф/кавычка, 0 или 1
        \s*          # пробел
        \.           # точка
        |            # или
        [`'’‘ʼʹˈ]   # любой апостроф/кавычка, 0 или 1
        |             # или
        \.            # точка
        |             # или
        :             # двоеточие
        |             # или
        \)?           # необязательная закрывающая скобка
    )
    """,
    re.VERBOSE
)

# pattern_line_start_r = re.compile(
#     r"""
#     ^\s*              # начало строки + пробелы
#     \(?                # необязательная открывающая скобка
#     \s*
#     (?P<number>\d{1,3})  # число (1-3 цифры)
#     \s*
#     (?:               # незахватывающая группа для окончания
#         [`'’‘ʼʹˈ]?   # любой апостроф/кавычка, 0 или 1
#         \.?           # необязательная точка
#         |             # или
#         \.            # точка
#         |             # или
#         :             # двоеточие
#         |             # или
#         \)           # необязательная закрывающая скобка
#     )
#     """,
#     re.VERBOSE
# )

pattern_line_start_r = re.compile(
    r"""
    ^\s*                  # начало строки + пробелы
    \(?                    # необязательная открывающая скобка
    \s*                    
    (?P<number>\d{1,3})    # число (1-3 цифры)
    \s*                    
    (?:                    # незахватывающая группа для окончания
        \.                 # точка
        |                   # или
        [`'’‘ʼʹˈ]         # любой апостроф/кавычка, 0 или 1
        |                   # или
        [`'’‘ʼʹˈ]         # любой апостроф/кавычка, 0 или 1
        \s*                 # пробелы после кавычки
        \.                 # точка
        |                   # или
        :                   # двоеточие
        |                   # или
        \)                  # закрывающая скобка
    )
    """,
    re.VERBOSE
)


def renumber_trust_source(text: str) -> dict:
    if not text.strip():
        return {}

    lines = text.splitlines()
    pattern = pattern_line_start_r  # твой универсальный шаблон

    found_numbers: List[int] = []

    # ищем все номера в начале строк
    for line in lines:
        m = pattern.match(line)
        if m:
            found_numbers.append(int(m.group("number")))

    result: dict = {}

    if found_numbers:
        # --- Все номера кратные 5
        if all(n % 5 == 0 for n in found_numbers):
            anchors: list[tuple[int, int]] = []
            for idx, line in enumerate(lines):
                m = pattern.match(line)
                if m:
                    num = int(m.group("number"))
                    anchors.append((idx, num))
            numbers = _restore_sequence(anchors, len(lines))  # твоя существующая логика
            for num, line in zip(numbers, lines):
                content = pattern.sub('', line, count=1).strip()
                result[num] = content

        # --- Есть обычные номера (не кратные 5)
        else:
            result = {}
            # собираем все позиции с номерами
            anchors = [(idx, int(pattern.match(line).group("number")))
                       for idx, line in enumerate(lines) if pattern.match(line)]

            if not anchors:
                # вообще нет номеров → обычная последовательность
                for i, line in enumerate(lines, start=1):
                    result[i] = line.strip()
            else:
                # восстанавливаем нумерацию для всех строк
                for idx, line in enumerate(lines):
                    # ищем ближайший «якорь» впереди
                    next_anchor = next(((a_idx, num) for a_idx, num in anchors if a_idx >= idx), None)
                    if next_anchor:
                        a_idx, num = next_anchor
                        key = num - (a_idx - idx)  # уменьшаем от ближайшего номера
                    else:
                        # если впереди нет якоря — продолжаем с последнего найденного
                        last_anchor_num = anchors[-1][1]
                        key = last_anchor_num + (idx - anchors[-1][0])

                    # удаляем номер из строки, если он есть
                    m = pattern.match(line)
                    content = pattern.sub('', line, count=1).strip() if m else line.strip()
                    result[key] = content
    # --- Номеров нет совсем
    else:
        for i, line in enumerate(lines, start=1):
            result[i] = line.strip()

    # проверка на один элемент
    if len(result) == 1:
        only_key = next(iter(result))
        if only_key != 1:
            result = {1: result[only_key]}

    return result

def split_numbered_text_with_intro(text) -> dict:
    """Выводит словарь из текста с последовательно нумерованными участками.
       Шаблонные номера удаляются из строк."""

    matches = list(pattern_line_start.finditer(text))
    result = {}

    if matches:
        first_num = int(matches[0].group("number"))

        # текст перед первой нумерацией
        if matches[0].start() > 0:
            result[first_num - 1] = text[:matches[0].start()].strip()

        # основной цикл по найденной нумерации
        for i, m in enumerate(matches):
            key = int(m.group("number"))
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            # удаляем номер из строки
            content_clean = pattern_line_start.sub('', content, count=1).strip()
            result[key] = content_clean
    else:
        return {1: text.strip()}
    if len(result) == 1:
        # получаем единственный ключ
        only_key = next(iter(result))
        if only_key != 1:
            result = {1: result[only_key]}

    return result

# ---------------------------------------------------------------

text = """is-tù ha-mug-tim
6. sa Su-a-nim
ù Du-du
Tr.8. a-na 30 ha -am -sa-t[i]m
R. i-sa-qû-lu-û
10. su-ma i-na u4 mi su nu
ma-al-d-tim
12. ld is-qw-lu-1 i-na ITi./KAM
1 2 GiN.TA a-na
14. 1 ma -na-em
û-sû -bu KÙ.BABBAR
16. i-qd-qd-ad
[sd]l-mi-su-nu (°érasure)
Tr.18. ù ki ni su nu ra-ki-is
CG. IGI Im-dI-DINGIR
20. IGI Du-ra-a
"""

print(renumber_trust_source(text))