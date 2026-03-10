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
    print(translit_tokens)
    translation_lengths = [len(sent.split()) for sent in translation_sentences]
    print(translation_lengths)
    total_translit = len(translit_tokens)
    print(total_translit)
    total_translation = sum(translation_lengths)
    print(total_translation)
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


trlit = ''
translate = "I love you"
print(align_and_mark_sentences(trlit, translate))