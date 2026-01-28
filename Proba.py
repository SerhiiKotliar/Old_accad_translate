import pandas as pd
import re

# Разделители блоков
SEPARATOR_RE = re.compile(r'^-+$')

# Разрешенные символы для транслитерации
# TRANSLIT_LINE_RE = re.compile(
#     r"^[A-Za-zŠšḫḪṣṢṭṬʾʿ0-9\-ℵ \[\]\.\!⅀⅁ᲟᲠ–]+$"
# )
TRANSLIT_LINE_RE = re.compile(r'''
^(?!\s*\d)                |    # не начинается с чистого номера
(?=.*(
        -[a-z]            |   # дефисная слоговая морфология
        \d                |   # индексные цифры (Puzur4)
        \b(?:DUMU|KIŠIB|LÚ|IGI|EN|AŠ|ŠA)\b |  # формулы / логограммы
        [šḫṭṣ]            |   # диакритика
))
(?!.*[.,;:!?])                # нет пунктуации перевода
(?!.*\b[A-Z]?[a-z]{3,}\b\s+\b[A-Z]?[a-z]{3,}\b)       # нет нормального текста
[A-Za-zúēīāíšḫṭṣŠÍÚḪṮṢ0-9.\[\] \?\!§⅀⅁ᲟᲠᲢ–\- ]+
$
''', re.VERBOSE)

# Морфемные разделители (дефис или ℵ)
MORPHEME_SEP_RE = re.compile(r"[-ℵ]")

# Стоп-слова для фильтрации английского, немецкого и турецкого текста - расширенный список
FOREIGN_WORD_RE = re.compile(
    r"\b("
    # Немецкие слова
    r"Jetzt|ist|gerade|ein|Brief|des|und|der|die|das|von|mit|"
    r"für|auf|aus|bei|nach|über|unter|zwischen|durch|wegen|"

    # Английские слова
    r"desk|bound|commercial|manager|who|conducted|"
    r"this|must|have|been|invented|institution|"
    r"if|when|going|to|and|palace|textiles|old|assyrian|procedures|"
    r"the|of|for|with|from|by|on|as|or|but|not|so|then|also|"
    r"that|which|what|where|why|how|"
    r"he|she|it|we|they|"
    r"was|were|be|being|been|"
    r"will|would|can|could|should|may|might|must|"
    r"about|above|after|against|among|around|before|behind|below|beneath|beside|between|beyond|"
    r"during|except|inside|outside|since|through|throughout|toward|under|until|upon|within|without|"

    # Турецкие слова
    r"ile|bir|şu|ben|sen|biz|siz|onlar|"
    r"ama|fakat|ancak|çünkü|eğer|"
    r"evet|hayır|lütfen|teşekkür|ediyorum|ederim|"
    r"gibi|kadar|göre|sonra|önce|arasında|altında|üstünde|içinde|dışında|"
    r"ile|sadece|hem|de|mi|mı|mü|"
    r"var|yok|olmak|yapmak|gitmek|gelmek|almak|vermek|"
    r"büyük|küçük|yeni|eski|güzel|iyi|kötü|"
    r"bugün|dün|yarın|şimdi|sonra|"
    r"nerede|ne|kim|nasıl|niçin|niye|ne zaman|"
    r"kitap|defter|kalem|masa|sandalye|ev|okul|iş|"
    r"türkçe|türk|türkiye|ankara|istanbul|izmir|"
    r"merhaba|selam|hoşgeldiniz|güle güle|allah|allahım|"
    r"efendim|bey|hanım|bay|bayan|"
    r"lütfen|rica|ediyorum|mümkün|mü|"
    r"anlamak|bilmek|düşünmek|söylemek|konuşmak|"
    r"üzgünüm|özür|dilerim|affedersiniz|"
    r"tabii|elbette|belki|muhtemelen|kesinlikle|"
    r"sağ|sol|ön|arka|yukarı|aşağı|"
    r"hızlı|yavaş|uzun|kısa|geniş|dar|"
    r"aç|tok|susuz|susamış|yorgun|dinç|"
    r"zengin|fakir|mutlu|mutsuz|hasta|sağlıklı|"
    r"anne|baba|kardeş|çocuk|aile|arkadaş|"
    r"yemek|içmek|uyumak|çalışmak|oynamak|"
    r"okumak|yazmak|dinlemek|bakmak|görmek|"
    r"almak|satmak|ödemek|kazanmak|kaybetmek|"
    r"gitmek|gelmek|dönmek|kalmak|ayrılmak|"
    r"başlamak|bitmek|devam|etmek|değişmek|"
    r"istemek|sevmek|nefret|etmek|beğenmek|"
    r"anlamak|anlaşılmak|anlaşmak|"
    r"yardım|istemek|yardım|etmek|"
    r"beklemek|aramak|bulmak|kaybetmek|"
    r"düşmek|kalkmak|oturmak|ayakta|durmak|"
    r"koşmak|yürümek|uçmak|yüzmek|"
    r"gülmek|ağlamak|bağırmak|fısıldamak|"
    r"öpmek|sarılmak|tutmak|bırakmak|"
    r"açmak|kapamak|bağlamak|çözmek|"
    r"yıkamak|temizlemek|kirletmek|"
    r"pişirmek|kızartmak|haşlamak|"
    r"giymek|çıkarmak|değiştirmek|"
    r"uyanmak|uyumak|rüya|görmek|"
    r"doğmak|ölmek|yaşamak|yaşam|"
    r"zaman|mekan|yer|dünya|evren|"
    r"güneş|yıldız|gezegen|"
    r"hava|toprak|ateş|"
    r"renk|şekil|boyut|ağırlık|"
    r"ses|müzik|gürültü|sessizlik|"
    r"ışık|karanlık|sıcak|soğuk|"
    r"tatlı|ekşi|tuzlu|acı|"
    r"yumuşak|sert|pürüzsüz|pürüzlü|"
    r"taze|bayat|temiz|kirli|"
    r"canlı|cansız|bitki|hayvan|"
    r"ağaç|çiçek|yaprak|meyve|"
    r"kedi|kopek|kuş|balık|"
    r"şehir|köy|kasaba|ülke|"
    r"cadde|sokak|meydan|park|"
    r"bina|ev|apartman|villa|"
    r"oda|mutfak|banyo|tuvalet|"
    r"kapı|pencere|duvar|tavan|"
    r"masa|sandalye|koltuk|yatak|"
    r"dolap|raf|çekmece|"
    r"buzdolabı|fırın|ocak|"
    r"televizyon|radyo|bilgisayar|telefon|"
    r"para|bank|kredi|borç|"
    r"iş|meslek|maaş|izin|"
    r"okul|üniversite|öğrenci|öğretmen|"
    r"ders|sınav|not|ödev|"
    r"spor|futbol|basketbol|voleybol|"
    r"müzik|resim|tiyatro|sinema|"
    r"kitap|gazete|dergi|internet|"
    r"tatil|seyahat|otel|plaj|"
    r"hava|durumu|yağmur|kar|güneş|"
    r"sağlık|hasta|doktor|hastane|"
    r"yasa|mahkeme|polis|suç|"
    r"din|inanç|tanrı|ibadet|"
    r"siyaset|parti|seçim|hükümet|"
    r"ekonomi|ticaret|sanayi|tarım|"
    r"kültür|sanat|edebiyat|bilim|"
    r"tarih|coğrafya|matematik|fizik|"
    r"dil|kelime|cümle|gramer|"
    r"numara|adres|telefon|numara|"
    r"ad|soyad|yaş|doğum|tarihi|"
    r"milliyet|vatandaşlık|pasaport|"
    r"aile|durumu|medeni|hal|"
    r"eğitim|durumu|mezuniyet|"
    r"iş|tecrübesi|referans|"
    r"hobi|ilgi|alanı|beceri|"
    r"özellik|avantaj|dezavantaj|"
    r"problem|çözüm|sonuç|etki|"
    r"sebep|neden|amaç|hedef|"
    r"plan|program|proje|"
    r"rapor|belge|dosya|"
    r"toplantı|konferans|seminer|"
    r"yazışma|iletişim|görüşme|"
    r"sözleşme|anlaşma|protokol|"
    r"satış|pazarlama|reklam|"
    r"üretim|kalite|kontrol|"
    r"nakliye|lojistik|depolama|"
    r"finans|muhasebe|denetim|"
    r"insan|kaynakları|personel|"
    r"teknoloji|sistem|yazılım|"
    r"güvenlik|koruma|tedbir|"
    r"çevre|doğa|kirlilik|"
    r"enerji|elektrik|doğalgaz|"
    r"ulaşım|trafik|yol|köprü|"
    r"iletişim|medya|haber|"
    r"eğlence|oyun|festival|"
    r"alışveriş|market|mağaza|"
    r"restoran|cafe|bar|"
    r"otel|konaklama|rezervasyon|"
    r"bank|atm|kredi|kartı|"
    r"posta|kargo|kurye|"
    r"sigorta|sağlık|sigortası|"
    r"vergi|harç|ceza|"
    r"kanun|yönetmelik|tüzük|"
    r"hak|özgürlük|sorumluluk|"
    r"değer|ilk|erdem|"
    r"sevgi|saygı|hoşgörü|"
    r"dostluk|arkadaşlık|aşk|"
    r"mutluluk|hüzün|öfke|"
    r"korku|endişe|panik|"
    r"umut|hayal|gerçek|"
    r"başarı|başarısızlık|tecrübe|"
    r"zaman|mekan|an|geçmiş|gelecek|"
    r"hayat|ölüm|doğum|yaşam|"
    r"ruh|beden|akıl|kalp|"
    r"düşünce|duygu|davranış|"
    r"alışkanlık|gelenek|görenek|"
    r"festival|bayram|kutlama|"
    r"yemek|içecek|tatlı|"
    r"giyim|kuşam|moda|"
    r"mimari|tasarım|estetik|"
    r"mühendislik|teknik|teknoloji|"
    r"tarım|hayvancılık|balıkçılık|"
    r"madencilik|enerji|sanayi|"
    r"turizm|seyahat|konaklama|"
    r"eğitim|öğretim|araştırma|"
    r"sağlık|tıp|hastane|"
    r"spor|egzersiz|antrenman|"
    r"sanat|müzik|resim|heykel|"
    r"edebiyat|şiir|roman|hikaye|"
    r"sinema|tiyatro|konser|"
    r"medya|gazete|televizyon|"
    r"internet|sosyal|medya|"
    r"bilgisayar|telefon|tablet|"
    r"yazılım|program|uygulama|"
    r"veri|bilgi|bilgi|sistemi|"
    r"güvenlik|şifre|erişim|"
    r"ağ|internet|bağlantı|"
    r"donanım|yazılım|sistem|"
    r"sunucu|istemci|veritabanı|"
    r"web|site|domain|hosting|"
    r"e-ticaret|online|alışveriş|"
    r"dijital|pazarlama|reklam|"
    r"sosyal|ağ|platform|"
    r"blog|forum|yorum|"
    r"fotoğraf|video|ses|"
    r"grafik|animasyon|efekt|"
    r"oyun|konsol|simülasyon|"
    r"yapay|zeka|makine|öğrenme|"
    r"robot|otomasyon|sensör|"
    r"sürücü|kontrol|sistemi|"
    r"enerji|tasarrufu|verimlilik|"
    r"çevre|dostu|sürdürülebilir|"
    r"geri|dönüşüm|atık|"
    r"iklim|değişikliği|küresel|ısınma|"
    r"doğal|afet|deprem|sel|"
    r"sağlık|hijyen|temizlik|"
    r"beslenme|diyet|spor|"
    r"psikoloji|terapi|danışmanlık|"
    r"hukuk|avukat|mahkeme|"
    r"ekonomi|finans|yatırım|"
    r"emlak|konut|ofis|"
    r"taşıt|araba|motor|"
    r"ulaşım|toplu|taşıma|"
    r"inşaat|mimari|mühendislik|"
    r"dekorasyon|mobilya|aksesuar|"
    r"bahçe|peyzaj|bitki|"
    r"ev|hayvanı|bakım|"
    r"çocuk|bakımı|eğitim|"
    r"yaşlı|bakım|hizmet|"
    r"engelli|erişilebilirlik|"
    r"kadın|erkek|çocuk|"
    r"genç|yaşlı|orta|yaş|"
    r"bekar|evli|boşanmış|"
    r"çocuk|sahibi|çocuksuz|"
    r"öğrenci|çalışan|emekli|"
    r"meslek|maaşlı|"
    r"part|time|full|time|"
    r"uzaktan|çalışma|esnek|saat|"
    r"kariyer|gelişim|eğitim|"
    r"yetenek|beceri|deneyim|"
    r"cv|özgeçmiş|referans|"
    r"mülakat|görüşme|test|"
    r"işe|alım|oriantasyon|"
    r"performans|değerlendirme|"
    r"terfi|zam|ikramiye|"
    r"izin|tatil|rapor|"
    r"işten|çıkarma|istifa|"
    r"sendika|toplu|sözleşme|"
    r"grev|lokavt|uzlaşma|"
    r"vergi|sigorta|prim|"
    r"emeklilik|fon|yardım|"
    r"sağlık|sigortası|özel|"
    r"hayat|sigortası|kaza|"
    r"konut|sigortası|araba|"
    r"seyahat|sigortası|bagaj|"
    r"yasal|sorumluluk|sigortası|"

    # Другие общие иностранные слова
    r"dass|sich|nicht|nur|auch|aber|oder|"
    r"por|para|con|sin|sobre|entre|hacia|"
    r"pour|avec|sans|entre|vers|"
    r"per|con|senza|tra|verso"
    r")\b",
    re.I
)

# Явные признаки аккадской транслитерации
AKKADIAN_INDICATOR_RE = re.compile(
    r"[ŠšḪḥṢṣṬṭʾʿ⅀⅁ᲟᲠ]|"  # Аккадские специальные символы
    r"\[.*?\]|"  # Квадратные скобки
    r"\(.*?\)|"  # Круглые скобки
    r"\{.*?\}|"  # Фигурные скобки
    r"\b[A-Z][a-zšḫṭṣ]+-[a-zšḫṭṣ]+\b|"  # Слова с дефисом, начинающиеся с заглавной
    r"\b[a-zšḫṭṣ]+-[a-zšḫṭṣ]+\b|"  # Слова с дефисом из строчных
    r"\b\d+[rv]\b|"  # Номера строк: 14r, 15v и т.д.
    r"x\+|x\-|x\?|x=\d+|"  # Фрагменты табличек
    r"\.\.\.|…|"  # Многоточия
    r"\d+['ˈ]|"  # Числа с апострофом
    r"–[^ ]"  # Длинное тире не после пробела
)

# Признаки, что это НЕ транслитерация (пропускать такие строки)
NOT_TRANSLIT_RE = re.compile(
    r"\b[A-Z][a-z]{3,} [A-Z][a-z]{3,}\b|"  # Два заглавных слова подряд (имя собственное)
    r"\b[a-z]{4,} [a-z]{4,} [a-z]{4,}\b|"  # Три длинных слова подряд (предложение)
    r"^\d+ [A-Z][a-z]|"  # Начинается с цифры и заглавной буквы
    r"[a-z]{5,}-[a-z]{4,}[^šḫṭṣʾʿ]|"  # Длинные английские слова с дефисом
    r"[a-zA-ZäöüÄÖÜß]{5,}-[a-zA-ZäöüÄÖÜß]{4,}|" # Длинные немецкие слова с дефисом
    r"[a-zA-ZçğıİöşüÇĞİÖŞÜ]{5,}-[a-zA-ZçğıİöşüÇĞİÖŞÜ]{4,}|" # Длинные турецкие слова с дефисом
    r", |; |: |\. [A-Z]|"  # Знаки пунктуации с пробелом
    r"\b(?:[A-Za-z]+ ){3,}[A-Za-z]+\b"  # Более 3 слов подряд
)


def extract_transliteration(text) -> list:
    """
    Извлекает блоки транслитерации из текста.
    Склеивает строки, оканчивающиеся на - или ℵ с последующей.
    Возвращает список блоков.
    """
    if isinstance(text, list):
        text = "\n".join(text)

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

        # Проверка 1: Соответствует ли базовому формату транслитерации?
        has_basic_format = (
                TRANSLIT_LINE_RE.match(line_trimmed) and
                MORPHEME_SEP_RE.search(line_trimmed)
        )

        # if not has_basic_format:
        #     if current:
        #         blocks.append("\n".join(current).strip())
        #         current = []
        #     continue

        # Проверка 2: Содержит ли иностранные слова?
        has_foreign_words = FOREIGN_WORD_RE.search(line_trimmed)
        # if has_foreign_words:
        #     print("Найдено слово:", match.group())

        # Проверка 3: Содержит ли явные признаки НЕ транслитерации?
        is_not_translit = NOT_TRANSLIT_RE.search(line_trimmed)

        # Проверка 4: Содержит ли признаки аккадской транслитерации?
        has_akkadian_indicators = AKKADIAN_INDICATOR_RE.search(line_trimmed)

        # Логика принятия решения:
        # 1. Должен быть базовый формат
        # 2. Не должен содержать иностранных слов ИЛИ должен иметь аккадские индикаторы
        # 3. Не должен быть явно НЕ транслитерацией
        is_transliteration = (
                has_basic_format and
                (not has_foreign_words or has_akkadian_indicators) and
                not is_not_translit
        )

        # Особый случай: если есть аккадские индикаторы, принимаем даже с некоторыми иностранными словами
        if has_akkadian_indicators and has_basic_format and not is_not_translit:
            is_transliteration = True

        if is_transliteration:
            current.append(line_trimmed)
        else:
            if current:
                blocks.append("\n".join(current).strip())
                current = []

    if current:
        blocks.append("\n".join(current).strip())

    return blocks



# def extract_transliteration_only(text, start_pos: int) -> list:
#     """
#     Извлекает блоки транслитерации из текста.
#     Склеивает строки, оканчивающиеся на - или ℵ с последующей.
#     Возвращает список блоков.
#     """
#     start_line_index = 0
#     result = []
#     # if isinstance(text, list):
#     #     text = "\n".join(text)
#
#     raw_lines = text.splitlines()
#     # 1. Подстрока от позиции до конца строки
#     result.append(raw_lines[start_line_index][start_pos:])
#
#     # lines = []
#     buffer = ""
#
#     for line in raw_lines:
#         line = line.rstrip()
#         if not line:
#             if buffer:
#                 result.append(buffer)
#                 buffer = ""
#             continue
#
#         if not buffer:
#             buffer = line
#         else:
#             buffer += " " + line
#
#         if not line.endswith(("-", "ℵ")):
#             result.append(buffer)
#             buffer = ""
#
#     if buffer:
#         # lines.append(buffer)
#         result.append(buffer)
#
#     # Формируем блоки транслитерации
#     blocks = []
#     current = []
#
#     for line in result:
#         # Пропускаем разделители
#         if SEPARATOR_RE.match(line):
#             continue
#
#         line_trimmed = line.strip()
#
#         # Пропускаем пустые строки
#         if not line_trimmed:
#             continue
#
#         # Проверка 1: Соответствует ли базовому формату транслитерации?
#         has_basic_format = (
#                 TRANSLIT_LINE_RE.match(line_trimmed) and
#                 MORPHEME_SEP_RE.search(line_trimmed)
#         )
#
#         if not has_basic_format:
#             if current:
#                 blocks.append("\n".join(current).strip())
#                 current = []
#             continue
#
#         # Проверка 2: Содержит ли иностранные слова?
#         has_foreign_words = FOREIGN_WORD_RE.search(line_trimmed)
#         if has_foreign_words:
#             print("Найдено слово:", has_foreign_words.group())
#         # Проверка 3: Содержит ли явные признаки НЕ транслитерации?
#         is_not_translit = NOT_TRANSLIT_RE.search(line_trimmed)
#
#         # Проверка 4: Содержит ли признаки аккадской транслитерации?
#         has_akkadian_indicators = AKKADIAN_INDICATOR_RE.search(line_trimmed)
#
#         # Логика принятия решения:
#         # 1. Должен быть базовый формат
#         # 2. Не должен содержать иностранных слов ИЛИ должен иметь аккадские индикаторы
#         # 3. Не должен быть явно НЕ транслитерацией
#         is_transliteration = (
#                 has_basic_format and
#                 (not has_foreign_words and has_akkadian_indicators) and
#                 not is_not_translit
#         )
#
#         # # Особый случай: если есть аккадские индикаторы, принимаем даже с некоторыми иностранными словами
#         # if has_akkadian_indicators and has_basic_format and not is_not_translit:
#         #     is_transliteration = True
#
#         if is_transliteration:
#             current.append(line_trimmed)
#         # else:
#         #     if current:
#         #         blocks.append("\n".join(current).strip())
#         #         current = []
#
#     if current:
#         blocks.append("\n".join(current).strip())
#
#     return blocks
def extract_letter_space_digit_colon_space(text: str, start_search_pos: int, pattern: str):
    if start_search_pos != 0:
        start_search_pos += 1
    pattern = re.compile(pattern)

    match = pattern.search(text, start_search_pos)
    if not match:
        return None, None, len(text)
    print(f"Найден поисковый якорь: {match.group()}")
    # ---------------------------------------------------
    pos = match.end()
    # pos = pos - 3
    pos_middle = text.find("\n", pos)
    pos = text.find("\n", pos_middle+1)
    result = ""
    # pos = match.end() + 1
    # поиск начала транслитерации
    # следующая после якоря позиция строки
    num_row = 0
    while pos < len(text):
        n_l, next_first_pos = get_next_line_trl(text, pos)
        if next_first_pos >= len(text):
            return result, True, pos
        if num_row > 2:
            return None, None, match.end()
        num_row += 1
        line_trl = []
        if n_l:
            line_trl = extract_transliteration(n_l)
        while line_trl:
            # сборная транслитерация
            result += (" ".join(line_trl))
            # последняя позиция в строке \n
            # pos = pos + len(line_trl) + 1
            # первая позиция в следующей строке
            pos = text.find("\n", next_first_pos + 1)
            if pos >= len(text):
                return result, True, next_first_pos
            # строка
            n_l, next_first_pos = get_next_line_trl(text, pos)
            if next_first_pos >= len(text):
                return result, True, pos
            if n_l:
                line_trl = extract_transliteration(n_l)
            else:
                line_trl = ""
        if result:
            return result, True, next_first_pos
        pos = next_first_pos
    return result, None, next_first_pos


def get_next_line_trl(text: str, start_pos: int):
    # начало строки поиска
    pos = None if start_pos == len(text) else start_pos
    if pos  is None:
        return None, len(text)
    # конец строки поиска
    end = text.find('\n', pos)
    if end == -1 and pos < len(text):
        end = len(text)
    if end == -1:
        return None, pos
    if end == pos:
        pos = end + 1
        end = text.find('\n', pos)

    str_line = text[pos:end]
    str_line = re.sub(
        r'^\s*(?:[SK]\.|S\. K\.|v|\. v)\s*(?:\r?\n|$)',
        '',
        str_line,
        flags=re.MULTILINE
    )
    str_line = re.sub(
        r'(?m)^\s*\d{1,2}\.\s*',
        '',
        str_line
    )

    return str_line, end

# # Завантаженние и обработка данных
# # thiscompteca = "D:/Projects/Python/Конкурсы/Old_accad_translate/"
# thiscompteca = "G:/Visual Studio 2010/Projects/Python/Old_accad_translate/"
# csv_file_path = thiscompteca + '/data/publications.csv'
# df_trnl = pd.read_csv(csv_file_path)
#
# df_trnl = df_trnl.drop_duplicates()
# idx = df_trnl[df_trnl['has_akkadian']].index
# df_trnl.loc[idx, df_trnl.columns[2]] = (
#     df_trnl.loc[idx, df_trnl.columns[2]]
#     .str.replace("\\n", "\n", regex=False)
# )
#
# # Извлекаем все блоки транслитерации
# all_blocks = []
# for i in idx:
#     text = df_trnl.at[i, df_trnl.columns[2]]
#     blocks = extract_transliteration(text)
#     if blocks:
#         all_blocks.extend(blocks)
#
# # Формируем финальный текст с разделителями
# if all_blocks:
#     separator = "\n" + "-" * 40 + "\n"
#     result_text = separator.join(all_blocks)
# text = "(1) i-na 13 GIN a-mu-tim (2) sa E-là-li Na-àb-Sı'ı-en6 (3) 3 GIN a-mu-tàm (4) qà-tàm sa E-la-li (5 sa a-şé-er Ku-ra (6) 1146-bi-la-ni (7) 2 1/3 ma-na. {TA} (8) a KU.BABBAR ki-ma Ku-ra (9) Su A-nu-um a Là-qi-pi-im (K. 10) i-di-si (Ay. 11) IGI A-sur-i-nıi-ti (12) DUMU A-mur-DINGIR (13) IGI A-sur-na-da (14) DUMU A-sur-i-di (15) a-na a-wa-tim a-ni-a-tim (16) kà-ru-um Kà-ni-is (17) i-di-ni-a-ti-ma (18) ICI pà-at-ri-im (19) sa A-sur si-bu-ti-ni (K. 20) ni-di-in" # (1-2) Elâ-ili ve Nab-Su'en'in 13 segel amûtum'u içinden, (3-10) Elâ-ili'nin Kura vasıta- sıyla bana gönderdiği 3 segel serbest amûtum'u 2 1/3 mina gümüşe karşılık Kura'nın vekili Sû-Anum, Lâ-gipum'a onu satti. (11-12) Amur-ili'nin oğlu Assur-imittti'nin huzûrunda, (13-14) Assur-idi'nin oğlu Assur-nâdâ'nın huzûrunda. (15-17) Bu meseleler hakkinda Kani kârum'u bizim için hükmünü verdi ve (18-20) biz tanri Assur'un hançeri önünde şâhitlerimizi gösterdik."
# text ="""(1) i-na 13 GIN a-mu-tim (2) sa E-là-li Na-àb-Sı'ı-en6 (3) 3 GIN a-mu-tàm (4) qà-tàm sa
# E-la-li (5 sa a-şé-er Ku-ra (6) 1146-bi-la-ni (7) 2 1/3 ma-na. {TA} (8) a KU.BABBAR ki-ma
# Ku-ra (9) Su A-nu-um a Là-qi-pi-im (K. 10) i-di-si (Ay. 11) IGI A-sur-i-nıi-ti (12) DUMU A-
# mur-DINGIR (13) IGI A-sur-na-da (14) DUMU A-sur-i-di (15) a-na a-wa-tim a-ni-a-tim (16)
# kà-ru-um Kà-ni-is (17) i-di-ni-a-ti-ma (18) ICI pà-at-ri-im (19) sa A-sur si-bu-ti-ni (K. 20)
# ni-di-in
# (1-2) Elâ-ili ve Nab-Su'en'in 13 segel amûtum'u içinden, (3-10) Elâ-ili'nin Kura vasıta-
# sıyla bana gönderdiği 3 segel serbest amûtum'u 2 1/3 mina gümüşe karşılık Kura'nın vekili
# Sû-Anum, Lâ-gipum'a onu satti. (11-12) Amur-ili'nin oğlu Assur-imittti'nin huzûrunda, (13-14)
# Assur-idi'nin oğlu Assur-nâdâ'nın huzûrunda. (15-17) Bu meseleler hakkinda Kani kârum'u
# bizim için hükmünü verdi ve (18-20) biz tanri Assur'un hançeri önünde şâhitlerimizi gösterdik.
# """

text ="""consulted for the full list of their occurrences.
(31) KUB 32.9 rev. 8–11, CTH 761.2 (MS), Starke 1985: 90

pa [ha-la-a-l]i-iš a-aš-du ma-a[l-ha-aš-ša-aš-ši-iš EN-aš] ma-a-aš-
ša-na-an-za pár-ra-a[n a]d-du-wa-la-ti EM[E-ti ad-]du-wa-la-ti
i-iš-š[a-ra-ti] ta-pa-ru-wa-aš-ša-an-za-t[i d]a-a-ta-ri-ya-am-n[a-
aš-ša-a]n-za-ti hi-ru-ta-aš-ša-a[n-za-ti] ma-a-⅀ya⅁-aš-ša-an-za-ti
⅀EME-ti⅁

‘Let then the ritual patron be purified in front of the gods from
the evil tongue, from the evil hand, from the tongue of taparu-s,
incantations, curses, from the slander of the adults’.
(32) KUB 9.6 iii 12–14, CTH 759.2 (LNS), Starke 1985: 115

ku-iš-tar ma-al-ha-aš-ša-aš-ša-an-za-an EN-ya a-ad-du-wa-la
a-an-ni-ti a-an DINGIR.MEŠ-in-zi a-ah-ha na-a-ta-at-ta ta-ta-
ar-ha-an-du

‘Whoever causes evil to the patron of the ritual (pl. tantum), let
the gods break him like reeds’.
38 By contrast, the form á-pa-sa-ti in CEKKE § 6b does not necessarily need to be
interpreted as ‘their’, as Hawkins (2000, I: 145) has suggested. The translation “They
bought the city Kamana from the Kanapuweans together with its X”, where ‘its’ has
‘city’ as its antecedent, is likewise acceptable. Therefore this example cannot be used
as a probative argument in the present section.
yakubovich_f3_15-74.indd 47
yakubovich_f3_15-74.indd 47
10/9/2009 9:25:26 AM
10/9/2009 9:25:26 AM
--------------------------------------------------
Текст всієї статті:
 48
chapter one
(33) KUB 35.78 i 7–8, CTH 767.1 (NS), Starke 1985: 134

[x?-x? ma-al-]ha-aš-ša-aš-ša-an-za-an-za EN-an-z[a īkkun]attas 1!
UDU sipanti

‘[?] sacrifices one anointed? sheep on behalf of the patrons of the
rituals’.
Melchert’s analysis can perhaps be carried one step further. The same
variety of Luvian that disambiguated possessive adjectives in oblique
"""
pattern = r'[A-Z][a-z]{3,} \d{4}[a-z]?: \d+(?:[–\-]\d+)?'

text, tr_lit, pos_end = extract_letter_space_digit_colon_space(text, 0, pattern)
print(text)
print(tr_lit)
print(pos_end)
# block = extract_transliteration_only(text)
# if block:
#     print(f"Найден блок транслитерации: {block}")  # , len(all_blocks))

# blocks = extract_transliteration(text)
# if blocks:
#     print("Найден блок транслитерации: ") #, len(all_blocks))

    # Отладочный вывод: покажем проблемные строки которые были отфильтрованы
#     print("\nПримеры отфильтрованных строк (для проверки):")
#     test_strings = [
#         "desk-bound commercial manager who conducted",
#         "This naruqqu-institution must have been invented",
#         "14 Jetzt ist gerade ein Brief des Iriba 15 an Litib-libbaSu 16 und ein Brief"
#     ]
#
#     for test_str in test_strings:
#         print(f"\nПроверка строки: '{test_str}'")
#         has_basic = TRANSLIT_LINE_RE.match(test_str) and MORPHEME_SEP_RE.search(test_str)
#         has_foreign = bool(FOREIGN_WORD_RE.search(test_str))
#         not_translit = bool(NOT_TRANSLIT_RE.search(test_str))
#         akkadian = bool(AKKADIAN_INDICATOR_RE.search(test_str))
#
#         print(f"  Базовый формат: {has_basic}")
#         print(f"  Иностранные слова: {has_foreign}")
#         print(f"  NOT транслитерация: {not_translit}")
#         print(f"  Аккадские индикаторы: {akkadian}")
#         print(
#             f"  Результат: {'ПРИНЯТО' if (has_basic and (not has_foreign or akkadian) and not not_translit) else 'ОТФИЛЬТРОВАНО'}")
#
#     print("\n" + "=" * 60 + "\n")
#
#     # Выводим первые 3 блока для проверки
#     for i, block in enumerate(all_blocks[:6], 1):
#         print(f"Блок транслитерации {i}:")
#         print(block[:300] + "..." if len(block) > 300 else block)
#         print("-" * 50)
#
#     # # Сохраняем в файл
#     # with open(thiscompteca + '/transliteration_blocks.txt', 'w', encoding='utf-8') as f:
#     #     f.write(result_text)
# else:
#     print("Блоки транслитерации не найдены")