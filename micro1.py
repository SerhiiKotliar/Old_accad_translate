import re


def clear_from_ocr_for_text_last(text: str) -> str:
    """
    Исправляет диапазоны число-дефис-число:
    - отделяет лишние цифры справа (только если правая часть длиннее левой)
    - если справа дробь → удаляем отделённую лишнюю цифру
    - если справа буквенное слово → OCR 1->I, 0->O, 5->S, присоединяем к слову
    - если справа не буквенное слово и не дробь → удаляем цифру
    """

    # 1. Находим все диапазоны и сразу обрабатываем их с контекстом
    pattern = r'(\d+)\s*[-–—]\s*(\d+)'

    result = []
    i = 0
    n = len(text)

    while i < n:
        match = re.match(pattern, text[i:])
        if match:
            left = match.group(1)
            right = match.group(2)
            match_len = len(match.group(0))

            # Добавляем текст до диапазона
            result.append(text[i:i + match.start()])

            # Только если правая часть длиннее левой
            if len(right) > len(left):
                main_right = right[:len(left)]
                extra = right[len(left):]  # лишние цифры

                # Добавляем исправленный диапазон
                result.append(f"{left}-{main_right}")

                # Смотрим, что идёт после диапазона
                pos = i + match_len

                # Запоминаем позицию для дальнейшей обработки
                after_pos = pos

                # Проверяем, есть ли пробел и наша лишняя цифра
                if (pos < n and text[pos] == ' ' and
                        pos + 1 < n and text[pos + 1] == extra[0]):

                    # Пропускаем пробел и лишнюю цифру
                    pos += 2

                    # Пропускаем возможные дополнительные пробелы
                    while pos < n and text[pos] == ' ':
                        pos += 1

                    # Анализируем следующий токен
                    if pos < n:
                        # СЛУЧАЙ 1: дробь (цифры/цифры)
                        fraction_match = re.match(r'(\d+/\d+)', text[pos:])
                        if fraction_match:
                            # Удаляем лишнюю цифру, добавляем пробел и дробь
                            result.append(' ')
                            result.append(fraction_match.group(1))
                            pos += len(fraction_match.group(1))
                            i = pos
                            continue

                        # СЛУЧАЙ 2: буквенное слово
                        word_match = re.match(r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]+)', text[pos:])
                        if word_match:
                            word = word_match.group(1)
                            # Преобразуем цифру в букву
                            if extra[0] == '1':
                                result.append(' I')
                            elif extra[0] == '0':
                                result.append(' O')
                            elif extra[0] == '5':
                                result.append(' S')
                            else:
                                # Если цифра не 1,0,5 - удаляем её (просто пробел)
                                result.append(' ')
                            result.append(word)
                            pos += len(word)
                            i = pos
                            continue

                        # СЛУЧАЙ 3: число
                        number_match = re.match(r'(\d+)', text[pos:])
                        if number_match:
                            # Удаляем лишнюю цифру, добавляем пробел и число
                            result.append(' ')
                            result.append(number_match.group(1))
                            pos += len(number_match.group(1))
                            i = pos
                            continue

                    # Если ничего не подошло - удаляем цифру (просто пробел)
                    result.append(' ')
                    i = pos
                    continue
                else:
                    # Нет нашей лишней цифры - просто добавляем пробел
                    result.append(' ')
                    i = pos
                    continue
            else:
                # Правая часть не длиннее левой - оставляем как есть
                result.append(f"{left}-{right}")
                i += match_len
                continue
        else:
            # Не нашли диапазон - добавляем текущий символ
            if i < n:
                result.append(text[i])
            i += 1

    text = ''.join(result)

    # 2. Финальная очистка
    # Убираем множественные пробелы
    text = re.sub(r' +', ' ', text)

    # Убираем пробелы перед знаками препинания
    text = re.sub(r'\s+([,.!?;:])', r'\1', text)

    return text


# Тестирование
if __name__ == "__main__":
    tests = [
        ("17-191 1/2 talent", "17-19 1/2 talent"),
        ("27-280 nlarin", "27-28 Onlarin"),
        ("or 5-9 B", "or 5-9 B"),
        ("bana geri getirdiler. 17-191 1/2 talent bakira karşilik",
         "bana geri getirdiler. 17-19 1/2 talent bakira karşilik"),
        ("gelsin de 27-280 nlarin bedeli",
         "gelsin de 27-28 Onlarin bedeli"),
    ]

    for input_text, expected in tests:
        result = clear_from_ocr_for_text_last(input_text)
        print(f"Вход:  '{input_text}'")
        print(f"Выход: '{result}'")
        print(f"Ожид:  '{expected}'")
        print(f"OK:    {result == expected}\n")