import re

def clear_from_ocr_for_text_last(text: str) -> str:
    # шаблон диапазона
    pattern = re.compile(
        r'(\d+)\s*-\s*(\d+)(\s+([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]+))?'
    )

    def range_repl(m):
        left = m.group(1)
        right = m.group(2)
        word = m.group(4)

        # если правая часть(конец диапазона)длиннее
        if len(right) > len(left):

            main_right = right[:len(left)]
            extra = right[len(left):]

            # смотрим что идёт после всего совпадения
            rest = text[m.end():]

            # 1️⃣ если справа дробь → удаляем extra
            if re.match(r'\s*\d+/\d+', rest):
                return f"{left}-{main_right}"

            # 2️⃣ если справа слово (захваченное)
            if word:
                extra_conv = (
                    extra.replace('1', 'I')
                         .replace('5', 'S')
                         .replace('0', 'O')
                )
                return f"{left}-{main_right} {extra_conv}{word}"

            # 3️⃣ иначе просто отделяем extra
            return f"{left}-{main_right} {extra}"

        return m.group(0)

    text = pattern.sub(range_repl, text)
    text = re.sub(r'\s+', ' ', text).strip()

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