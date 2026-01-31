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


def get_next_line_trl(text: str, start_pos: int):
    """возвращает строку транслитерации, следующую за назначенной позицией
     очищенную от мусора и позицию конца строки"""
    # начало строки поиска
    pos = None if start_pos == len(text) else start_pos
    if pos  is None:
        return "", len(text)
    # конец строки поиска
    end = text.find('\n', pos)
    if end == -1:
        end = len(text)
        return text[pos:end], end
    # позиция старта совпадает с переводом строки
    if end == pos and pos < len(text):
        pos = end + 1
        end = text.find('\n', pos)
        if end == -1 and pos <= len(text):
            end = len(text)
    # достигнут конец текста
    if end == pos and pos >= len(text):
        return "", end
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

    return str_line, end

def get_next_line(text: str, start_pos: int):
    """возвращает строку следующую за назначенной позицией"""
    # начало строки поиска
    pos = None if start_pos == len(text) else start_pos
    if pos is None:
        return "", len(text)
    # конец строки поиска
    end = text.find('\n', pos)
    if end == -1:
        end = len(text)
        return text[pos:end], end
    # позиция старта совпадает с переводом строки
    if end == pos and pos < len(text):
        pos = end + 1
        end = text.find('\n', pos)
        if end == -1 and pos <= len(text):
            end = len(text)
    # достигнут конец текста
    if end == pos and pos >= len(text):
        return "", end
    str_line = text[pos:end]

    return str_line, end

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

        # Проверка 4: Содержит ли признаки аккадской транслитерации?
        has_akkadian_indicators = AKKADIAN_INDICATOR_RE.search(line_trimmed)
        if not has_foreign_words and has_basic_format and has_akkadian_indicators:
            if ':' in line_trimmed:
                line_trimmed = line_trimmed.replace(":", "")
        # Проверка 3: Содержит ли явные признаки НЕ транслитерации?
        is_not_translit = NOT_TRANSLIT_RE.search(line_trimmed)

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

        num_morf = text.count("ℵ")
        num_defis = text.count('-')
        num_div = max(num_morf, num_defis)
        # мало дефисов в строке
        if (num_div > 0 and len(text) / num_div - 1 > 12) or num_div == 0:
            is_transliteration = False

        if is_transliteration:
            current.append(line_trimmed)
        else:
            if current:
                blocks.append("\n".join(current).strip())
                current = []

    if current:
        blocks.append("\n".join(current).strip())

    return blocks

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
text = """ANKARA KÜLTEPE TABLETLERİ II
65
Éti tù-up-ta-na-ra-ad
10. İGI A-ld-bi-im is-tù
7 ha-am-sa-tim UR UD U
ii-sé-bi-lâ-kum-ma
ki-ma ha-ra-na-tum
K.
şa-âb-ta-ni URUDU pi-ha-ru
A.y. 15. is-tù mi-sa-al 4a-ra-nim
û-ta-e-ru-nim
a-na 1 1/2 GO UR UD U 1 ma-na
KÙ. BABBAR ku-nu-ki-a A-ld-bu-um
na-âs-a-kum A-sur
20. ù i-lu-ku-nu li-tù-ld
a-na li-bi-led
ici lù-ma-nim a-na KU.BABBAR
su-a-ti 2 GU URUDU lu d -giıl
a-hu-a a-tù-nu a-na
v
25. ma-1ci T ÙG. HI. A a-na É hà-ri-im
v
ta-di-a-ni té-er-tâh-nu
li-li-kam-ma si-im-su-nu KU.BABBAR
K.
lu-sé-bi-lci-ku-nu-ti
a-na (J-şur-t-itar qi-bi-ma
30. i-nu-mi KÙ.BABBAR
S. K.
ta-du-nu-su-ni si-bé-e
sa ki-ma URUDU sa 4a-bu-lâ-ak-u-ni
sa-bu-û su-ku-su-ma
ù KÙ.BABBAR di-su-um
ı -4Mannum-k -Assur ve Uşur-si-Istar'a (özellikle) Mannum-Iyi-Assur'a Puzur-Adad şöyle
söylüyor:S-9Bendeki 1 1/2 talent bakir için adresime yazma! Evime devamli yazıyorsun ve
ailemi korkutup duruyorsun! ıo-12A1(i)-abum'un huzurunda 7 4amutum önce bakırı sana
gönderdim ve13- ı6 sanki yollar benim için tutulmuş (kapatfimiş) gibi, uşaklar bakın yolun
yarisindan bana geri getirdiler.17-191 1/2 talent bakıra karşılık 1 mina gümüşü benim müh-
rümle Äl(i)-abum sana taşimaktadir.19-23 anri Asur ve sizin tanrılarınız şâhit olsunlar, se-
nin kalbini kırmamak için o paraya karşılık 2 talent bakiri tartmış oldum.24-27Kardeşlik
ediniz, senin karum dâiresine benim için depo ettiğin kumaşlar hakkındaki haberiniz bana
gelsin de 27-28onların bedeli olan paraya size göndereyim.29-34Usur-si-ßtar'a söyle: Parayi ona
vereceğiniz zaman, benim ona borçlu olduğum bakiri ödediğime dair şâhitleri göster ve pa-
rayı ona ver.
9: satırda geçen parädum'un G ve D kalıplarında "korkutmak" mânâsına; Dtn formun-
da ise bunun devamlılık bildiren fonksiyonu olduğu görüşünü benimsiyoruz: AHw s. 827.;
K. Hecker, GKT 77d ve 78d'de bu fil kökü için "titremek, ürpermek" karşiliğini vermiştir."""

def cleaning_from_ocr(text: str) -> str:
    # уборка мусора
    subs = [
        (r'ı\s+ı', '11'),
        (r'ı\s+', '1'),
        (r'ı', '1'),
        (r'5([A-Za-zА-Яа-я])', r'S\1'),
        (r'A1', 'Ai'),
        (r'([A-Za-zА-Яа-я])1\b', r'\1i'),
        (r'([A-Za-zА-Яа-я]),(\d)', r'\1 \2'),
        (r'\s(\d)\s(\d)\s', r' \1-\2 '),
        (r'(?<=\d)o', '0'),
        (r'S-9', '5-9'),
        # (r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', ''),
    ]
    for pattern, repl in subs:
        text = re.sub(pattern, repl, text)

    text = re.sub(
        r'^\s*(?:[SK]\.|S\. K\.|A\.y\.\s*|v|\. v)\s*$',
        '',
        text,
        flags=re.MULTILINE
    )
    text = re.sub(r'^.\.y\.\s*', '', text, flags=re.MULTILINE)
    return text

def process_text(text, cleaning_from_ocr):
    lines = text.splitlines(keepends=True)  # сохраняем \n
    processed_lines = [cleaning_from_ocr(line) for line in lines]
    return ''.join(processed_lines)

def is_tablet(text: str):
    pos_tablet = text.find("tablet")
    pos_start_tr_after_tablet = re.search(r'^.\.y\.\s*', text)
    if pos_start_tr_after_tablet is not None:
        pos_start_tr_after_tablet = pos_start_tr_after_tablet.start()
        if pos_start_tr_after_tablet > pos_tablet:
            return True, pos_tablet + 1, pos_start_tr_after_tablet - 1
    return False, len(text), 0

def str_to_first_diapazon(text: str):
    pos_first_diapazon = re.search(r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', text)
    if pos_first_diapazon is not None:
        pos_first_diapazon = pos_first_diapazon.start()
    if pos_first_diapazon:
        return text[:pos_first_diapazon] if pos_first_diapazon else ""
    return ""

result = process_text(text, cleaning_from_ocr)
print(result)
res_is_tablet = is_tablet(result)
if res_is_tablet[0]:
    print(f"Перевод - Транслитерация {res_is_tablet[1]} - {res_is_tablet[2]}")
else:
    print("Транслитерация - Перевод")
print(str_to_first_diapazon(text))

# print(tr_lit)
# print(pos_end)
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