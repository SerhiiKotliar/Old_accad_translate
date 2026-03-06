



# проверка количества цифр в строке
def more_than_half_digits(line_trimmed):
    digits = sum(ch.isdigit() for ch in line_trimmed)
    print(digits)
    print(len(line_trimmed))
    return digits > len(line_trimmed) / 3

if more_than_half_digits("Gtn 40, 17; 42, 21."):
    is_transliteration = False
    print(is_transliteration)