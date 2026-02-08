import sys
import pandas as pd
# import numpy as np
import re
import nltk
from langdetect import detect
from langdetect import DetectorFactory
from deep_translator import GoogleTranslator
from typing import Dict, List, Tuple

from tensorflow.python.debug.lib.check_numerics_callback import enable_check_numerics

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

Unfin_Data: dict = {"number":"", "trlit":"", "perevod":""}

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


# Разделители блоков
SEPARATOR_RE = re.compile(r'^-+$')

# Разрешенные символы для транслитерации
# TRANSLIT_LINE_RE = re.compile(
#     r"^[A-Za-zŠšḫḪṣṢṭṬʾʿ0-9\-ℵ \[\]\.\!⅀⅁ᲟᲠ–]+$"
# )
# -----------------------------------------
# ^(?!\s*\d)                |    # не начинается с чистого номера
# ---------------------------------------------
TRANSLIT_LINE_RE = re.compile(r'''
^(?!\s*\d+\s*$)            |   # не начинается с чистого номера
(?=.*(
        -[a-z]            |   # дефисная слоговая морфология
        \d                |   # индексные цифры (Puzur4)
       \b(?:DINGIR|LUGAL|EN|NIN|DUMU|SAL|MUNUS|GURUŠ|LU₂|AMA|AB|AḪ|ŠEŠ|NIN₉|E₂|KI|URU|KUR|ABZU|A|IM|UD|U₄|ITI|MU|GIŠ|DU₃|GAR|GUB|TUKU|ŠU₂|ZI|NAM|ME|ŠU|IGI|DIŠ|MIN|EŠ|LIMMU|IA|KIŠIB|LÚ|AŠ|ŠA|BABBAR|KÙ|NUMUN)\b  # формулы / логограммы
        [šḫṭṣ]            |   # диакритика
))
(?!.*[.,;:!?])                # нет пунктуации перевода
(?!.*\b[A-Z]?[a-z]{3,}\b\s+\b[A-Z]?[a-z]{3,}\b)       # нет нормального текста
[A-Za-zúēīāíšḫṭṣŠÍÚḪṮṢ0-9.\[\] \?\!§⅀⅁ℵᲟᲠᲢ–\- ]+
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
    r"[ŠšḪḥṢṣṬṭʾʿ⅀⅁ᲟᲠ]|"
    r"[₀₁₂₃₄₅₆₇₈₉]|"
    r"[ᵈᵐᶠᵏ]|(?:\{[dmfkg]\})|"
    r"\b[A-Z]{2,}(?:\.[A-Z]{2,})+\b|"
    r"\b[A-Z]{2,}[-ℵ][a-z]+\b|"
    r"\[.*?\]|\(.*?\)|\{.*?\}|"
    r"\b[A-Z][a-zšḫṭṣ]+[-ℵ][a-zšḫṭṣ]+\b|"
    r"\b[a-zšḫṭṣ]+[-ℵ][a-zšḫṭṣ]+\b|"
    r"\b\d+[rv]\b|"
    r"x\+|x\-|x\?|x=\d+|"
    r"\.\.\.|…|"
    r"\d+['ˈ]|"
    r"–[^ ]|"
    r"\|"
)
# Признаки, что это НЕ транслитерация (пропускать такие строки)
NOT_TRANSLIT_RE = re.compile(
    r"\b[A-Z][a-z]{3,} [A-Z][a-z]{3,}\b|"        # Два заглавных слова подряд
    r"\b[a-z]{4,} [a-z]{4,} [a-z]{4,}\b|"        # Три длинных слова подряд
    r"^\d+ [A-Z][a-z]|"                          # Цифра + заглавное слово
    r"[a-z]{5,}[-ℵ][a-z]{4,}(?![šḫṭṣʾʿ])|"       # англ. дефис/ℵ (НО не аккад.)
    r"[a-zA-ZäöüÄÖÜß]{5,}[-ℵ][a-zA-ZäöüÄÖÜß]{4,}|" # нем.
    r"[a-zA-ZçğıİöşüÇĞİÖŞÜ]{5,}[-ℵ][a-zA-ZçğıİÖŞÜ]{4,}|" # тур.
    r", |; |: |\. [A-Z]|"                        # Пунктуация
    r"\b(?:[A-Za-z]+ ){3,}[A-Za-z]+\b"           # 3+ слов подряд
)

WORD_RE = re.compile(
    r"[A-Za-zÀ-ÖØ-öø-ÿĞğŞşİıÇçÜüÖöÄäßÉéÈèÊêÂâÎîÔôÛûšṣṭḫʾʿ]{2,}"
)

MORPHEME_CHAIN_RE = re.compile(
    r"\b(?:[A-Za-zšṣṭḫʾʿ]{1,3}-){2,}[A-Za-zšṣṭḫʾʿ]{1,3}\b"
)

AKKADIAN_FUNCTION_WORDS = {
    "ša", "ina", "ana", "itti", "eli", "kīma", "kima",
    "ištu", "ištu", "ultu", "adi", "u", "šaṭru"
}


def extract_transliteration(text) -> list:
    """
    Извлекает блоки транслитерации из текста с их проверкой.
    Склеивает строки, оканчивающиеся на - или ℵ с последующей.
    Возвращает список блоков.
    """
    if isinstance(text, list):
        text = "\n".join(text)
    # очистка от мусора OCR
    text = cleaning_from_ocr(text)
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
        is_transliteration = False
        # Проверка 1: Соответствует ли базовому формату транслитерации?
        has_basic_format = (
                TRANSLIT_LINE_RE.match(line_trimmed) and
                MORPHEME_SEP_RE.search(line_trimmed)
        )
        if has_basic_format:
            is_transliteration = True
        # if not has_basic_format:
        #     if current:
        #         blocks.append("\n".join(current).strip())
        #         current = []
        #     continue

        # Проверка 2: Содержит ли иностранные слова?
        has_foreign_words = FOREIGN_WORD_RE.search(line_trimmed)

        # Проверка 4: Содержит ли признаки аккадской транслитерации?
        has_akkadian_indicators = AKKADIAN_INDICATOR_RE.search(line_trimmed)
        # if not has_foreign_words and has_basic_format and has_akkadian_indicators:
        #     if ':' in line_trimmed:
        #         line_trimmed = line_trimmed.replace(":", "")
        # Проверка 3: Содержит ли явные признаки НЕ транслитерации?
        is_not_translit = False
        if not is_transliteration:
            is_not_translit = NOT_TRANSLIT_RE.search(line_trimmed)
        # else:
        #     is_not_translit = False

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

        # num_morf = text.count("ℵ")
        # num_defis = text.count('-')
        # num_div = max(num_morf, num_defis)
        # # мало дефисов в строке
        # if (num_div > 0 and len(text) / num_div - 1 > 12) or num_div == 0:
        #     is_transliteration = False
        num_morf = line.count("ℵ")
        num_defis = line.count('-')
        num_div = num_morf + num_defis
        # мало дефисов в строке
        if (num_div > 0 and len(line) / num_div - 1 > 16) or num_div == 0:
            is_transliteration = False

        if is_transliteration:
            current.append(line_trimmed)
        else:
            break
            # if blocks:
            #     return blocks
            # return []
            # if current:
            #     blocks.append("\n".join(current).strip())
            #     current = []

    if current:
        blocks.append("\n".join(current).strip())

    return blocks

def find_translit_by_rows(text: str, pos: int):
    """поиск транслитерации начиная с позиции после якоря по строкам
    возвращает транслитерацию или None и позицию конца"""
    pos_end_of_line = 0
    result = ""
    num_row = 0
    while pos < len(text):
        # строка от её первой позиции и позиция конца строки
        n_l, pos_end_of_line = get_next_line(text, pos)
        # прекращение поиска транслитерации после 2 ложных строк
        if num_row > 1:
            return None, pos
        line_trl = []
        if n_l:
            line_trl = extract_transliteration(n_l)
        end_translit = 0
        while line_trl:
            # сборная транслитерация
            result += ("\n".join(line_trl))
            end_translit = pos_end_of_line
            # строка
            n_l, pos_end_of_line = get_next_line(text, pos_end_of_line)
            if pos_end_of_line == -1:
                return result, end_translit
            if n_l:
                line_trl = extract_transliteration(n_l)
            else:
                line_trl = ""
        num_row += 1
        if result:
            return result, end_translit
        pos = pos_end_of_line
    return None, pos_end_of_line


def is_translation(text: str, one_word: bool=False) -> bool:
    """подтверждает что строка есть перевод"""
    if not text or len(text) < 10:
        return False

    text = text.strip()

    # Морфемные цепочки → почти наверняка транслитерация
    if MORPHEME_CHAIN_RE.search(text):
        return False

    # Слова длиной ≥ 2
    words = WORD_RE.findall(text)
    if len(words)< 2 and not one_word:
        return False

    # # Короткие токены (ša, ina, a-na и т.п.)
    tokens = re.findall(r"[A-Za-zšṣṭḫʾʿ]+", text.lower())
    if tokens:
        short_tokens = [t for t in tokens if len(t) <= 3]
        if len(short_tokens) / len(tokens) > 0.6:
            return False

    # Частотные служебные слова аккадского
    if sum(1 for t in tokens if t in AKKADIAN_FUNCTION_WORDS) >= 2:
        return False

    return True



def get_next_line(text: str, start_pos: int):
    """возвращает не пустую строку, следующую за назначенной позицией
     и позицию конца строки"""
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
    while pos == end and pos < len(text):
        pos = end + 1
        end = text.find('\n', pos)
        if end == -1 and pos <= len(text):
            end = len(text)
    # if end == pos and pos < len(text):
    #     pos = end + 1
    #     end = text.find('\n', pos)
    #     if end == -1 and pos <= len(text):
    #         end = len(text)
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

def count_words(text):
    return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿА-Яа-яЁё]+", text))

def detect_translate(text: str, start_pos: int):
    """подтверждает что  очищенная строка есть перевод
    выводит флаг и строку"""
    is_translate = False
    one_word = False
    # начало строки поиска
    pos = None if start_pos == len(text) else start_pos
    if pos  is None:
        return is_translate, ""
    # конец строки поиска
    end = text.find('\n', pos)
    if end == -1 and pos < len(text):
        end = len(text)
    # if end == -1:
    #     return is_translate, text[pos:end]
    str_line = text[pos:end]
    # уборка мусора
    str_line = cleaning_from_ocr(str_line)
    # subs = [
    #     (r'ı\s+ı', '11'),
    #     (r'ı\s+', '1'),
    #     (r'ı', '1'),
    #     (r'5([A-Za-zА-Яа-я])', r'S\1'),
    #     (r'A1', 'Ai'),
    #     (r'([A-Za-zА-Яа-я])1\b', r'\1i'),
    #     (r'([A-Za-zА-Яа-я]),(\d)', r'\1 \2'),
    #     (r'\s(\d)\s(\d)\s', r' \1-\2 '),
    #     (r'(?<=\d)o', '0'),
    #     # (r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', ''),
    # ]
    #
    # for pattern, repl in subs:
    #     str_line = re.sub(pattern, repl, str_line)
    # str_line = re.sub(r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', '', str_line)
    # шаблон диапазона страниц
    pattern = r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b'

    str_line, count = re.subn(pattern, '', str_line)
    if count_words(str_line) == 1:
        one_word = True
    if count > 2 or is_translation(str_line, one_word):
        is_translate = True
    # print("Количество замен:", count)

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

    return is_translate, str_line

def cleaning_from_ocr_prelim(text: str) -> str:
    subs = [
        (r'([a-z])ı\s+', r'\1i '),
        (r'ı\s+ı', '11'),
        (r'ı\s+', '1'),
        (r'ı', '1'),
        (r'5([A-Za-zА-Яа-я])', r'S\1'),
        (r'A1', 'Ai'),
        (r'([A-Za-zА-Яа-я])1\b', r'\1i'),
        (r'([A-Za-zА-Яа-я]),(\d)', r'\1 \2'),
        (r'\s(\d)\s(\d)\s', r' \1-\2 '),
        (r'(?<=\d)o', '0'),
        (r'(?<=\d)°', '0'),
        (r'S-9', '5-9'),
        (r'‰', ''),
        (r'™', ''),
        (r'([^\W\d_])4(-|[^\W\d_])', r'\1h\2'),
        (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\1h'),
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
        text = re.sub(pattern, repl, text)
    return text


def cleaning_from_ocr(text: str, trlit: bool = True) -> str:
    # уборка мусора

    if trlit:
        text = re.sub(
            r'^\s*(?:[SK]\.|S\. K\.|S\.K\.|K\.\s*\d|\n|v|\. v)\s*$',
            '',
            text,
            flags=re.MULTILINE
        )
        subs = [
            # (r'([a-z])ı\s+', r'\1i '),
            # (r'ı\s+ı', '11'),
            # (r'ı\s+', '1'),
            # (r'ı', '1'),
            # (r'5([A-Za-zА-Яа-я])', r'S\1'),
            # (r'A1', 'Ai'),
            # (r'([A-Za-zА-Яа-я])1\b', r'\1i'),
            # (r'([A-Za-zА-Яа-я]),(\d)', r'\1 \2'),
            # (r'\s(\d)\s(\d)\s', r' \1-\2 '),
            # (r'(?<=\d)o', '0'),
            # (r'(?<=\d)°', '0'),
            # (r'S-9', '5-9'),
            # (r'‰', ''),
            # (r'™', ''),
            (r':', ' '),
            (r'<([^<>]+)>', r'\g<1>'),
            (r'^.\d{1,}\n', ''),
            (r'^.\.?\s?y\.\s?\r?\n?', ''),
            (r'(?<=[A-Za-z0-9]):(?=[A-Za-z0-9])', ' '),
            (r'([^\W\d_])4(-|[^\W\d_])', r'\1h\2'),
            (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\1h'),
            (r'.\,.', ''),
            (r'\,\n', ''),
            #(r'(?<=[^\W_]):(?=[^\W_])', ' '),
            # (r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', ''),
            # (r'§', 'S'),
            # (r'\,', ' '),
            # (r'^.\.y\.\s*', ''),
            # (r'^.\.y\.\n', ''),
        ]
    else:
        subs = [
            # (r'([a-z])ı\s+', r'\1i '),
            # (r'ı\s+ı', '11'),
            # (r'ı\s+', '1'),
            # (r'ı', '1'),
            # (r'5([A-Za-zА-Яа-я])', r'S\1'),
            # (r'A1', 'Ai'),
            # (r'([A-Za-zА-Яа-я])1\b', r'\1i'),
            # (r'([A-Za-zА-Яа-я]),(\d)', r'\1 \2'),
            # (r'\s(\d)\s(\d)\s', r' \1-\2 '),
            # (r'(?<=\d)o', '0'),
            # (r'(?<=\d)°', '0'),
            # (r'S-9', '5-9'),
            # (r'‰', ''),
            # (r'™', ''),
            (r':', ''),
            (r'§', 'S'),
            (r'\$', '9'),
            (r'([^\W\d_])4(-|[^\W\d_])', r'\1h\2'),
            (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\1h'),
            # (r'9([A-ZА-ЯÀ-ÖØ-Þ])([a-zа-яà-öø-ÿ])', r'\1\2'),
            (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])([-\s])9([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\1\2\3'),
            (r'(\d{1,2}[-\s]\d{1,2})\$', r'\g<1>9'),
            (r'(?<=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1(?=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', 'i'),
            (r'(\d)S([A-Z])', r'\g<1>5\2'),
            (r'\s4([a-zа-яà-öø-ÿ])', r' h\1'),
            ]
    for pattern, repl in subs:
        text = re.sub(pattern, repl, text)
    for old, new in CHAR_MAP.items():
        text = text.replace(old, new)

    def remove_9(match):
        A = match.group(1)
        sep = match.group(2)
        B = match.group(3)
        # удаляем 9 только если длина B на 1 больше A
        if len(B) - len(A) == 1:
            return f"{A}{sep}{B}"
        else:
            return match.group(0)  # оставляем как есть
    pattern9 = r'(\d+)([-\s])(\d+)9'
    text = re.sub(pattern9, remove_9, text)

    def remove_9_after_dash(match):
        A = match.group(1)  # число до дефиса/пробела
        sep = match.group(2)  # дефис или пробел
        B = match.group(3)  # число после дефиса/пробела
        letters = match.group(4)  # буквы после числа

        # удаляем 9 только если B длиннее A на 1
        if len(B) - len(A) == 1:
            return f"{A}{sep}{B}{letters}"
        else:
            return match.group(0)  # оставляем как есть

    pattern = r'(\d+)([-\s])(\d+)9([A-ZА-ЯÀ-ÖØ-Þa-zа-яà-öø-ÿ]+)'

    text = re.sub(pattern, remove_9_after_dash, text)

    # text = re.sub(r'^.\.y\.\s*', '', text, flags=re.MULTILINE)
    return text

def process_text(text, trlit: bool = True):
    """Возвращает текст без мусора ОСR"""
    lines = text.splitlines(keepends=True)  # сохраняем \n
    processed_lines = [cleaning_from_ocr(line, trlit) for line in lines]
    return ''.join(processed_lines)

def is_tablet(text: str):
    """Ищет позицию предшествующую транслитерации с таблички
    и возвращает флаг находки и позиции начала перевода и начала транслитерации"""
    pos_tablet = re.search(r'\s*tablet\.\n', text, flags=re.MULTILINE)
    if pos_tablet is not None:
        # позиция начала перевода
        pos_tablet = pos_tablet.end()
        # позиция начала транслитерации
        pos_start_tr_after_tablet = re.search(r'^.\.?\s*y\.\n', text, flags=re.MULTILINE)
        if pos_start_tr_after_tablet is not None:
            # позиция конца перевода
            pos_end_translate_tablet = pos_start_tr_after_tablet.start() - 1
            text_translate = text[pos_tablet:pos_end_translate_tablet].strip()
            # очистка от мусора текста
            text_translate = process_text(text_translate, False)
            if looks_like_real_translation(text_translate):
                # позиция начала транслитерации после слова tablet
                pos_start_tr_after_tablet = pos_start_tr_after_tablet.end()
                text_transliterate = text[pos_start_tr_after_tablet:]
                # text_transliterate = normalize_for_mt(text_transliterate)
                # очистка от мусора
                result = process_text(text_transliterate)
                if pos_start_tr_after_tablet > pos_tablet and extract_transliteration(result):
                    # словарь транслитерации ключ номер строки и значение строка
                    result1 = renumber_trust_source(result)
                    # начало и конец перевода и начало транслитерации
                    # return True, pos_tablet + 1, pos_end_translate_tablet, pos_start_tr_after_tablet
                    # флаг, перевод, словарь транслитерации, позиция конца транслитерации
                    return True, (text_translate, result1), len(text)
    return False, ("", None), len(text)

def translate_after_translite(text: str, start_pos: int = 0):
    """Ищет позицию первого диапазона предложений в переводе
    после транслитерации и возвращает транслитерацию, если найдёт диапазон"""
    # pos_first_diapazon = re.search(r'\d{1,3}\s*[-–—-]\s*\d{1,3}', text[start_pos:], flags=re.MULTILINE)
    # 'r(\d{1,2})\s*[-–—]\s*(\d{1,2})'
    # pos_first_diapazon = re.search(r'\d{1,3}(?:\s*[-–—]\s*|\s+)\d{1,3}', text[start_pos:], flags=re.MULTILINE)
    pos_first_diapazon = re.search(r'(\d{1,3})\s*[-–—]\s*(\d{1,3})', text[start_pos:], flags=re.MULTILINE)
    # pos_first_transliteration = pos_first_translite(text[start_pos:])
    # if not pos_first_transliteration:
    #     text_trlit = text[:pos_first_diapazon]
    # text_trlit = text[pos_first_transliteration:pos_first_diapazon]
    if pos_first_diapazon is not None:
        pos_first_diapazon = pos_first_diapazon.start()
    pos_first_transliteration = pos_first_translite(text[start_pos:])
    if not pos_first_transliteration:
        text_trlit = text[:pos_first_diapazon - 1]
    else:
        text_trlit = text[pos_first_transliteration:pos_first_diapazon - 1]
    if extract_transliteration(text_trlit) and pos_first_diapazon:
        return text_trlit, pos_first_diapazon - 1
    return text_trlit, len(text)

def pos_first_translite(text: str, start_pos: int = 0):
    """Ищет позицию начала транслитерации
    и возвращает её"""
    pos_first_trl = re.search(r'^.\.?\s*y\.\n', text, flags=re.MULTILINE)
    return pos_first_trl.start() if pos_first_trl is not None else None


def _normalize_newlines(text: str) -> str:
    """
    why: корректная склейка строк по правилам:
    - '-\\n' или 'ℵ\\n' → склеить без пробела
    - иначе '\\n' → заменить на пробел
    """
    result = []
    i = 0

    while i < len(text):
        if text[i] == "\n":
            prev = result[-1] if result else ""
            if prev in {"-", "ℵ"}:
                pass  # ничего не добавляем
            else:
                result.append(" ")
            i += 1
        else:
            result.append(text[i])
            i += 1

    return "".join(result).strip()


def renumber_trust_source(text: str) -> Dict[int, str]:
    """преобразует транслитерацию с номерами строк типа ЧИСЛО. или ЧИСЛО:
    в словарь {номер_строки: текст} с восстановлением пропущенной нумерации"""

    lines = text.splitlines()
    n = len(lines)

    dic_trlits: Dict[int, str] = {}
    anchors = []  # (index, source_number)

    for i, line in enumerate(lines):
        m = re.match(r'\s*(\d+)\s*[.:]\s*(.*)', line)
        if m:
            num = int(m.group(1))
            if num % 5 == 0:
                anchors.append((i, num))

    if not anchors:
        dic_trlits[1] = _normalize_newlines(text)
        return dic_trlits

    result_numbers = [None] * n

    # --- до первого якоря
    first_idx, first_num = anchors[0]
    for i in range(first_idx, -1, -1):
        result_numbers[i] = first_num - (first_idx - i)

    # --- между якорями
    for (i1, n1), (i2, n2) in zip(anchors, anchors[1:]):
        for i in range(i1, i2):
            result_numbers[i] = n1 + (i - i1)
        result_numbers[i2] = n2

    # --- после последнего якоря
    last_idx, last_num = anchors[-1]
    for i in range(last_idx, n):
        result_numbers[i] = last_num + (i - last_idx)

    # --- сборка результата
    for num, line in zip(result_numbers, lines):
        content = re.sub(r'^\d{1,3}[.:]?\s*', '', line)
        content = _normalize_newlines(content)
        dic_trlits[num] = content

    return dic_trlits


def process_text_last(text: str, lines_dict: Dict[int, str]) -> Tuple[List[str], List[str]]:
    range_pattern = re.compile(r'(\d{1,3})(?:\s*[-–—]\s*(\d{1,3}))?')

    matches = list(range_pattern.finditer(text))

    dict_results: List[str] = []
    text_results: List[str] = []

    if not matches:
        merged = " ".join(lines_dict.values())
        cleaned = re.sub(r'\d{1,2}[\.:]\s*', '', merged)
        dict_results.append(cleaned)
        text_results.append(text.replace("\n", " ").strip())
        return dict_results, text_results

    for i, match in enumerate(matches):
        start = int(match.group(1))
        end = int(match.group(2)) if match.group(2) else start + 1

        keys = range(start, end)

        if not all(k in lines_dict for k in keys):
            continue  # why: пропускаем некорректный диапазон

        dict_results.append(" ".join(lines_dict[k] for k in keys))

        text_start = match.end()
        text_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        fragment = text[text_start:text_end].strip(" ()")
        text_results.append(fragment)

    return dict_results, text_results

def find_single_quote(text: str, start_pos: int, first: bool=True):
    # 3. Поиск одинарной открывающей кавычки
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("ʼ", "'")
        .replace("ʾ", "'")
    )
    if first:
        quote_pos_prob = text.find(" '", start_pos)
        quote_pos_prob = quote_pos_prob + 1 if quote_pos_prob != -1 else -1
        quote_pos_abz = text.find("\n'", start_pos)
        quote_pos_abz = quote_pos_abz + 1 if quote_pos_abz != -1 else -1
        if quote_pos_prob > 0 and quote_pos_abz > 0:
            quote_pos = min(quote_pos_prob, quote_pos_abz)
        elif quote_pos_abz < 0 or quote_pos_prob < 0:
            quote_pos = max(quote_pos_prob, quote_pos_abz)
        else:
            quote_pos = -1
        # quote_pos = quote_pos_abz
    else:
        quote_pos = text.find("'", start_pos)
    if quote_pos == -1:
        return quote_pos
    # дистанция от транслитерации
    # кавычки после текста
    diff = quote_pos - start_pos
    if first:
        if diff >= 100:
            return -1
    #else:
    return quote_pos

def find_double_quote(text: str, start_pos: int, first: bool=True):
    text = (
        text.replace("“", '"')
        .replace("”", '"')
        .replace("„", '"')
        .replace("‟", '"')
        .replace("«", '"')
        .replace("»", '"')
    )
    if first:
        quote_pos_prob = text.find(" \"", start_pos)
        quote_pos_prob = quote_pos_prob + 1 if quote_pos_prob != -1 else -1
        quote_pos_abz = text.find("\n\"", start_pos)
        quote_pos_abz = quote_pos_abz + 1 if quote_pos_abz != -1 else -1
        if quote_pos_prob > 0 and quote_pos_abz > 0:
            quote_pos = min(quote_pos_prob, quote_pos_abz)
        elif quote_pos_abz < 0 or quote_pos_prob < 0:
            quote_pos = max(quote_pos_prob, quote_pos_abz)
        else:
            quote_pos = -1
    else:
        quote_pos = text.find("\"", start_pos)
    if quote_pos == -1:
        return quote_pos
    # дистанция от транслитерации или от перевода
    # кавычки после текста
    # diff = quote_pos - start_pos
    # if diff >= 100:
    #     return -1
    # else:
    return quote_pos


def parse_numbered_fragments(text: str) -> Dict[int, str]:
    """выбирает из текста пронумерованные в круглых скобках
    строки и создаёт словарь номер строки и строка
    номера очищаются от всех символов кроме цифр"""

    pattern = re.compile(r"\(([^)]*)\)")
    matches = list(pattern.finditer(text))

    result: Dict[int, str] = {}

    for i, match in enumerate(matches):
        raw_key = match.group(1)
        digits_only = re.sub(r"\D", "", raw_key)

        if not digits_only:
            continue

        key = int(digits_only)

        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

        fragment = text[start:end].strip()
        fragment = _normalize_newlines(fragment)

        result[key] = fragment

    return result


def extract_ankara_next(text: str, start_pos: int, pattern: str):
    number_pred = ""
    trlit_pred = ""
    perevod_pred = ""
    pos_start_trlit = 0
    if start_pos < 0 or start_pos >= len(text):
        return None, None, start_pos
    end_pos = len(text)
    # text = text.replace('TABLETLERİ u', 'TABLETLERİ II')
    pattern = re.compile(pattern)
    match = pattern.search(text, start_pos)
    if not match:
        return None, None, len(text)
    print(f"Найден поисковый якорь Ankara: {match.group()}")
    text = text[match.end():]
    if any(Unfin_Data.values()):
        number_pred = Unfin_Data['number']
        trlit_pred = Unfin_Data['trlit']
        perevod_pred = Unfin_Data['perevod']
    # предварительная очистка
    text = cleaning_from_ocr_prelim(text)
    if trlit_pred != "":
        pos_start_trlit = pos_first_translite(text, 0)
    else:
        patterns_number_before = [r'No. \d{1,}?\n', r'Tablet\n']
        patterns_number_after = [r'No. \d{1,}?\n', r'Zarf\n']
        for i, pattern_number in enumerate(patterns_number_before):
            pattern_number_before = re.compile(pattern_number)
            # pattern_number_after = re.compile(patterns_number_after[i])
            match_number = pattern_number_before.search(text, start_pos)
            if match_number:
                match_number_before = match_number
                text = text[match_number_before.end():]
                Unfin_Data['number'] = match_number_before.group()
                break
        pattern_trlit = re.compile(r"\(([^)]*)\)")
        match_trlit = pattern_trlit.search(text, start_pos)
    if match_trlit:
        pos_start_trlit = match_trlit.start()
    text = text[pos_start_trlit:]
    # match_number = pattern_number.search(text, start_pos)
    # транслитерация и первая позиция диапазона в переводе
    text_trlit, pos_start_perevod = translate_after_translite(text)
    if pos_start_perevod == len(text):
        Unfin_Data['trlit'] = text_trlit
    if not text_trlit or not extract_transliteration(text_trlit) or not pos_start_perevod:
        return ("", ""), False, pos_start_perevod
    if trlit_pred and extract_transliteration(trlit_pred):
            text_trlit = trlit_pred + text_trlit
    # последняя позиция перевода
    pos_end_perevod = re.search(r'(^St\.\s\d{1,2}:)', text, flags=re.MULTILINE)
    if pos_end_perevod:
        pos_end_extract = pos_end_perevod.start() - 1
    else:
        pos_end_extract = len(text)
    # перевод
    result = text[pos_start_perevod:pos_end_extract]
    # очистка мусора
    result = process_text(result, False)
    if not detect_translate(result, pos_start_perevod):
        return ("", ""), False, end_pos
    if not pos_end_perevod:
        Unfin_Data['perevod'] = result
    # очистка от мусора
    text_trlit = process_text(text_trlit)
    # словарь с ключами номерами и строками транслитерации
    # result1 = renumber_trust_source(text_trlit)
    result1 = parse_numbered_fragments(text_trlit)
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (result, result1), True, pos_end_extract

def extract_after_ankara_next(text_dict_tr: tuple, pos_s: int):
    text_translate = text_dict_tr[0]
    list_trl_transl = process_text_last(text_translate, text_dict_tr[1])
    # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, True, pos_s


def split_accad_and_translate(csv_lines, marker="<sent>"):
    rows = []
    global_id = 0

    for line in csv_lines:
        accad_str, translate_str = parse_csv_line(line)

        accad_sentences = [s.strip() for s in accad_str.split(marker) if s.strip()]
        translate_sentences = naive_sent_tokenize(translate_str)

        min_len = min(len(accad_sentences), len(translate_sentences))
        accad_sentences = accad_sentences[:min_len]
        translate_sentences = translate_sentences[:min_len]

        for accad, trans in zip(accad_sentences, translate_sentences):
            rows.append({
                "id": global_id,
                "accad_str": accad,
                "translate": trans
            })
            global_id += 1

    return pd.DataFrame(rows, columns=["id", "accad_str", "translate"])
# -----------------------------------------------------------------------


# # # Завантаженние и обработка данных
# # Завантаження даних з CSV-файлу
# thiscompteca = "D:/Projects/Python/Конкурсы/Old_accad_translate"
# # thiscompteca = "G:/Visual Studio 2010/Projects/Python/Old_accad_translate/"
# csv_file_path = thiscompteca+'/data/publications.csv'
# df_trnl = pd.read_csv(csv_file_path)
# # ----------------------------------------
# df_trnl = df_trnl.drop_duplicates()
# #
# # # Извлекаем все блоки транслитерации
# # idx = df_trnl[df_trnl['has_akkadian']].head(40).index
# idx = df_trnl[df_trnl['has_akkadian']].index
# df_trnl.loc[idx, df_trnl.columns[2]] = (
#     df_trnl.loc[idx, df_trnl.columns[2]]
#     .str.replace("\\n", "\n", regex=False)
# )
# num = 0
# num_i = 0
# all_rows = []
# # idx = idx[:4000]
# # for val in values:
# # texts = ''
# # with open("output4.txt", "a", encoding="utf-8", errors="replace") as f:
# for i in idx:
#     print(f"{num_i + 1} текст начинаем искать")
#     print(f"{num + 1} пару блоков начинаем искать.\n")
#     print(f"Index = {i}\n")
#     # if i == 74880:
#     # if i == 5140:
#     # #     не печатает переводы
#     # # if i == 25:
#     # # if i == 130319:
#     #     print("PROVERKA")
#
#     list_row = process_text_and_build_csv_rows(df_trnl.at[i, df_trnl.columns[2]])
#     # list_row = process_text_and_build_csv_rows(val)
#     for row in list_row:
#         if row not in all_rows:
#             all_rows.append(row)
#             print(f"{num + 1} пара блоков найдена.\n")
#             # print(row)
#             num += 1
#     print(f"{num_i + 1} текст прошли")
#     num_i += 1
#
# new_df = split_accad_and_translate(all_rows)
# # new_df.to_csv('translate_from_publication.csv', index=False, quoting=csv.QUOTE_ALL)
# print("Примеры строк:")
# print(new_df.head(10))
# print(f"Кількість статей з перекладом: {len(idx)}\n")
# print(f"Кількість зроблених перекладів: {new_df.shape[0]}\n")
# # print(type(new_df))
# # print(new_df.shape)
# # print(new_df.head(5))
#
# # print(f"Кількість статей з перекладом: {len(values)}\n")
# # print(num)
# sys.exit()
text = """18
ANKARA KÜLTEPE TABLETLERi
No. 2
Elâ ve Assur-mâlik'in borçlu oldugu bir borç senedidir. Bir formülle borç, bütün âile-
nin üzerine de te§mil edilmi§ bulunuyor. Fâiz % 30'dur.
(1) 1 ma-na KI BABBAR (2) i-sé-er E-là-a (3) ù A-gur-ma-lik (4) MAN A-gur i-gu ig-tù
(5) ha-mug-tim sa MAN -A-gar (6) a-na 7 ha-am-ga-ti[m] (7) i-sa-qù-lu gu-ma (8) la ig-qù-lu
11/2 GÎN.T[A] (9) i-na ITU.KAM si-ib-tâm (K.10) ù-sù-bu ITU.KAM (Ay. 11) Ti-i-na-tim (12)
li-mu-um (13) A-gur-i-mi-ti (14) ma-là-hu-um KÙ BABBAR (15) i-qà-qà-ad (16) gal-mi-gu-nu
it (17) ki-ni-gu-nu ra-ki-is (18) IGI I-ku pi-a (19) IGI Zu-zu-ba-ar.
(1-4) Elâ ve AFiur-mâlik'in üzerinde Puzur Aggur'un 1 mina gümü§ü vardlr. (4-7) Puzur-
Aggur'un hamugtum'undan itibâren 7 haftaya kadar tartacaklar. (7-10) Eger tartmazlarsa
ayda hirer buçuk eqel gümü§ fâiz ilâve edecekler. (10-11) Tinâtum ayi, (12-14) gemici Aggur-
imitti'nin limum'u. (15-17) Gümü§ sag ve dâim olanlarin ba§ina baglanmi tir. (18) Iktippia'-
inn huzûrunda, (19) Zuzubar' ln huzûrunda.
St. 2: Elâ ismi Ela+ia'dan mürekkep bir ok§ama ismidir. Bk. a§agida No: 47,15.
11: Tinâtum, Teinâtum=lncirlerin olgunla§tlgl ay. tittu, ti'(it)tu=incir (AHw, s. 1363 a).
15-17: "i-qaqqad galmigunû ù kinigunû rakis" tâbirinin izah §ekli için bk. Landsber-
ger, ZA 35, 30.
19: ilk defa görülen Zuzubar adi yerli bir isim olmalldlr.
No. 3
Bu tabletin diger borç senetlerinden farkli tarafi, borcun zamaninda tediye edilmemesi
hâlinde yapilacak muâmelinin daha açik bir ifâdeyle söylenmi§ olmasidir. Fâiz miktari %
30'dur.
Tablet
(1) 231/2 ma-na 3 GIN (2) KÙ BABBAR sa-ru-pet-am (3) i-sé-er 1-ku-nim (4) A-gur-i-mi-
ti i-gu (5) ig-tù ha-mug-tim (6) 's'a dMAR.TU-ba-ni (7) it 1-ku-nim (8) a-na 40 ha-am-ia-tim
(9) i-ga-qal gu-ma (K.10) i-na u4-mi-gu (11) ma-al-a-tim (Ay. 12) là ig-qù-ul (13) 1 1/2 GIN.TA
(14)i-na ITU.1.KAM (15) a-na 1 ma-na-im (16) si-ib-tàm ù-sa-àb (17) ITU.1.KAM A-là-na-tim
(18) li-mu-um Sù-kà-li-a (19) IGI A-gur-SIPA (20) IGI A-sù-na-a (21) A-gur-ma-lik
(1-4) Ikûnum'un üzerinde Aggur-imitt1'nin 23 1/2 mina 3 gegel tasfiye edilmi§ gümü§ü
vardir. (5-7) Amurru-bâni ve Ikûnum'un hamugtum'undan itibâren (8-9) 40 haftaya ka-
"""
# text = """ANKARA KÜLTEPE TABLETLERİ II
# K.R. Veenhof ise AOATT s. 391-2'de CCT 4 10a, 18'de geçen pa-ru-ud yazılışındaki fiili
# "karıştırmak" mânâsında almiştir.
# 13-14: satırlarında geçen 4arränam Şabätum'un AHw s. 327a lb'de verilen "yola çak-
# mak, yola koyulmak, yolu takip etmek" mânâsından ziyâde bizim, AKT 178, 19-20. satır-
# lardaki 4arränam ka'ulum "yolu tutmak, kapamak" anlamina geldiğini benimsediğimizi
# belirtmek isteriz . Ancak, adıgeçen yerde de belirtildiği gibi, 4arränum'un gerek ka'ulum,
# gerek şabätum v.s. gibi fiillerle beraber geniş ve zengin bir mânâ çevresi meydana getirdiği
# de şüphesizdir.
# 19-20: satırlarda geçen Aiiur u ilukunû liitulct ibâresi hakkinda en son bkz. K.R. Veenhof-
# V. Donbaz, Anatolica XII s. 137,7,18; AHw s. 766b 1 d; CAD N II s. 122b c.
# 21-22: satırlardaki libbum leménum "kızmak, hiddetlenmek; gücenmek" ile ilgili ola-
# rak bkz. AHw s. 542b 5 b; CAD L s. 117a b.
# 33: satırdaki iu-ku-iu-ma'nın, surun-ium-ma şeklinde tahlil edilmesi gereklidir ve bu
# fiil formu 31. satırdaki iibé ile ilgilidir.
# No. 34, Gallâbum'un mektubu
# Kt nik 586;165-586-64; 5,5 X 4, 4 x 1,7 cm.; kahverenkli tablet.
# ,
# Gdllâbum'un Uşur-ii-htar'a, onun talimatı gereğince kumaşları seçtiklerini belirterek
# başladığı mektubudur. Daha sonra kumaşlarin âit olduğu kimselerden Mamma şehrinde
# kailum memuriyeti ile iştigal eden Abâ'nın evinde veyâ iş yerinde hesap yaptıklarından bah-
# setmekte ve üç şâhidin adını kaydetmektedir. Ayrıca İkûppia'nın sağlik haberini iletmekte
# ve kendilerinin İdi-Agur'un kumaşlarını seçmediklerini, bunun için endişelenmemesini ifâde
# etmektedir.
# Ö. y.
# a-na (J-şâ-ur-t-htar
# qi-bi4-ma um-ma
# Ga-lâ-bu-ma
# ma-id té-i-ir-ti-kà
# 5. TZIG ni-be-er-ma
# 30 TUG ku-a-û-tum
# 20 h 2 Tu G ia I-ku-pi-a
# 12 TtI G ia tâm-kit-ri-im
# U.BA 5 A-bar-ni-û
# 10. 23 TUG ia Lu-lu
# 2 TUG ia kà-sa-ri
# K.
# S U. NİGiN 91 Tt1G
# A.y.
# i-na Ma-ma
# """



# ----------------------------------------------------------------
pattern = r'ANKARA KÜLTEPE TABLETLERi\n'
(perevod, transliteration), flag, end_of_perevod = extract_ankara_next(text, 0, pattern)
# print(transliteration)
# print("\n")
# print(perevod)
trl_end, flag, end_perevod = extract_after_ankara_next((perevod, transliteration), end_of_perevod)

print(trl_end)
print("\n")
print(flag)
print("\n")
print(end_perevod)