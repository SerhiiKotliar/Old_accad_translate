
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
        return text[pos:end], end
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
        return "", end
    str_line = text[pos:end+1]
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

    return str_line, end
txt1 = "Stroka"
txt2 = "\n"
txt3 = "Tretia stroka\n"
txt4 = "Chetvertaia stroka"
print(get_next_line(txt1, 0))
pos_start_trlit = get_next_line(txt1, 0)[1] - len(txt1)
print(pos_start_trlit)
print(repr(get_next_line(txt2, 0)))
print(get_next_line(txt3, 0))
print(get_next_line(txt4, 0))