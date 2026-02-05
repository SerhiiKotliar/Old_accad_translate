#%%
import sys
# from unittest import case

import pandas as pd
# import numpy as np
import re
import nltk
from langdetect import detect
from langdetect import DetectorFactory
from deep_translator import GoogleTranslator

# from Proba import text_translate

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize
# import tensorflow as tf
# from tensorflow import keras
# from keras import layers
# import matplotlib.pyplot as plt

# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import RandomForestClassifier
# from keras.models import Sequential
# from tensorflow.keras.layers import Dense, Flatten, Conv1D
# from keras.utils import to_categorical
# from sklearn.metrics import accuracy_score
# import shutil
#%%
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
        \b(?:DUMU|KIŠIB|LÚ|IGI|EN|AŠ|ŠA|BABBAR|KÙ)\b |  # формулы / логограммы
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
    r"Jetzt|ist|gerade|ein|Brief|der|das|mit|"
    r"für|auf|aus|bei|nach|über|unter|zwischen|durch|wegen|"

    # Английские слова
    r"desk|bound|conducted|"
    r"beneath|beside|"
    r"except|outside|throughout|toward|"

    # Турецкие слова
    r"bir|şu|ben|biz|siz|onlar|"
    r"fakat|ancak|çünkü|eğer|"
    r"evet|hayır|lütfen|teşekkür|ediyorum|ederim|"
    r"gibi|kadar|göre|sonra|önce|arasında|altında|üstünde|içinde|dışında|"
    r"sadece|hem|mü|"
    r"var|yok|olmak|yapmak|gitmek|gelmek|almak|vermek|"
    r"büyük|küçük|yeni|eski|güzel|iyi|kötü|"
    r"bugün|dün|yarın|şimdi|sonra|"
    r"nerede|kim|nasıl|niçin|niye|ne zaman|"
    r"kitap|defter|kalem|masa|sandalye|ev|okul|iş|"
    r"türkçe|türk|türkiye|ankara|izmir|"
    r"merhaba|selam|hoşgeldiniz|güle güle|allah|allahım|"
    r"efendim|bey|hanım|bay|bayan|"
    r"lütfen|rica|ediyorum|mümkün|mü|"
    r"anlamak|bilmek|düşünmek|söylemek|konuşmak|"
    r"üzgünüm|özür|dilerim|affedersiniz|"
    r"tabii|elbette|belki|muhtemelen|kesinlikle|"
    r"sağ|ön|arka|yukarı|aşağı|"
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
    r"kredi|borç|"
    r"iş|meslek|maaş|izin|"
    r"okul|üniversite|öğrenci|öğretmen|"
    r"ders|sınav|ödev|"
    r"spor|futbol|basketbol|voleybol|"
    r"müzik|resim|tiyatro|sinema|"
    r"kitap|gazete|dergi|internet|"
    r"tatil|seyahat|otel|plaj|"
    r"hava|durumu|yağmur|kar|güneş|"
    r"sağlık|hasta|doktor|hastane|"
    r"yasa|mahkeme|polis|suç|"
    r"inanç|tanrı|ibadet|"
    r"siyaset|parti|seçim|hükümet|"
    r"ekonomi|ticaret|sanayi|tarım|"
    r"kültür|sanat|edebiyat|bilim|"
    r"tarih|coğrafya|matematik|fizik|"
    r"dil|kelime|cümle|gramer|"
    r"numara|adres|telefon|numara|"
    r"soyad|yaş|doğum|tarihi|"
    r"milliyet|vatandaşlık|pasaport|"
    r"aile|durumu|medeni|"
    r"eğitim|durumu|mezuniyet|"
    r"iş|tecrübesi|referans|"
    r"hobi|ilgi|alanı|beceri|"
    r"özellik|avantaj|dezavantaj|"
    r"çözüm|sonuç|etki|"
    r"sebep|neden|amaç|hedef|"
    r"program|proje|"
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
    r"eğlence|oyun|"
    r"alışveriş|mağaza|"
    r"restoran|cafe|"
    r"otel|konaklama|rezervasyon|"
    r"atm|kredi|"
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
    r"zaman|mekan|geçmiş|gelecek|"
    r"hayat|ölüm|doğum|yaşam|"
    r"ruh|beden|akıl|kalp|"
    r"düşünce|duygu|davranış|"
    r"alışkanlık|gelenek|görenek|"
    r"kutlama|"
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
    r"edebiyat|şiir|hikaye|"
    r"sinema|tiyatro|konser|"
    r"medya|gazete|televizyon|"
    r"internet|sosyal|medya|"
    r"bilgisayar|telefon|"
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
    r"dass|nicht|auch|aber|"
    r"por|sin|sobre|entre|hacia|"
    r"pour|avec|sans|entre|vers|"
    r"senza|tra|verso"
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
            return []
            # if current:
            #     blocks.append("\n".join(current).strip())
            #     current = []

    if current:
        blocks.append("\n".join(current).strip())

    return blocks

def find_translit_by_rows(text: str, pos: int):
    """поиск с начала транслитерации по строкам
    следующая после якоря позиция строки
    возвращает транслитерацию или None и позицию конца"""
    next_first_pos = 0
    result = ""
    num_row = 0
    while pos < len(text):
        # строка и её первая позиция
        n_l, next_first_pos = get_next_line_trl(text, pos)
        if num_row > 1:
            return None, pos
        line_trl = []
        if n_l:
            line_trl = extract_transliteration(n_l)
        end_translit = 0
        while line_trl:
            # сборная транслитерация
            result += ("\n".join(line_trl))
            end_translit = next_first_pos - 1
            # строка
            n_l, next_first_pos = get_next_line_trl(text, next_first_pos)
            if next_first_pos == -1:
                return result, end_translit
            if n_l:
                line_trl = extract_transliteration(n_l)
            else:
                line_trl = ""
        num_row += 1
        if result:
            return result, end_translit
        pos = next_first_pos
    return None, next_first_pos - 1


def extract_transliteration_only(text) -> str:
    """
    Извлекает блоки транслитерации из текста.
    Склеивает строки, оканчивающиеся на - или ℵ с последующей.
    Возвращает список блоков.
    """
    # if isinstance(text, list):
    #     text = "\n".join(text)

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
    # blocks = []
    blocks = ""
    # current = []
    current = ""

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

        if not has_basic_format:
            # if current:
            #     # blocks.append("\n".join(current).strip())
            #     blocks = " ".join(current).strip()
            #     # current = []
            #     current = ""
            continue

        # Проверка 2: Содержит ли иностранные слова?
        has_foreign_words = FOREIGN_WORD_RE.search(line_trimmed)
        # if has_foreign_words:
        #     print("Найдено слово:", has_foreign_words.group())
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
                (not has_foreign_words and has_akkadian_indicators) and
                not is_not_translit
        )

        # Особый случай: если есть аккадские индикаторы, принимаем даже с некоторыми иностранными словами
        if has_akkadian_indicators and has_basic_format and not is_not_translit:
            is_transliteration = True

        if is_transliteration:
            # current.append(line_trimmed)
            # current = current.join(line_trimmed)
            current += " " + line_trimmed
        # else:
        #     if current:
        #         blocks.append("\n".join(current).strip())
        #         current = []

    if current:
        # blocks.append("\n".join(current).strip())
        blocks += current

    return blocks


def is_translation(text: str, one_word: bool=False) -> bool:
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
    # if tokens:
    #     short_tokens = [t for t in tokens if len(t) <= 3]
    #     if len(short_tokens) / len(tokens) > 0.6:
    #         return False

    # Частотные служебные слова аккадского
    if sum(1 for t in tokens if t in AKKADIAN_FUNCTION_WORDS) >= 2:
        return False

    return True



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

def count_words(text):
    return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿА-Яа-яЁё]+", text))

def detect_translate(text: str, start_pos: int):
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
        return is_translate, text[pos:end]
    str_line = text[pos:end]
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
        # (r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', ''),
    ]

    for pattern, repl in subs:
        str_line = re.sub(pattern, repl, str_line)
    # str_line = re.sub(r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', '', str_line)
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

def cleaning_from_ocr(text: str) -> str:
    # уборка мусора
    for old, new in CHAR_MAP.items():
        text = text.replace(old, new)
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
        (r'S-9', '5-9'),
        (r'§', 'S'),
        (r':', ' '),
        (r'.', ' '),
        (r'<([^<>]+)>', '\1'),
        (r'^.\d{1,}\n', ''),
        (r'^.\.y\.\s*', ''),
        (r'^.\.y\.\n', ''),
        (r'(?<=[A-Za-z0-9]):(?=[A-Za-z0-9])', ' '),
        #(r'(?<=[^\W_]):(?=[^\W_])', ' '),
        # (r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b', ''),
    ]
    for pattern, repl in subs:
        text = re.sub(pattern, repl, text)

    text = re.sub(
        r'^\s*(?:[SK]\.|S\. K\.|K\.\s*\d|.\.?\s*y\.\s*|v|\. v)\s*$',
        '',
        text,
        flags=re.MULTILINE
    )
    # text = re.sub(r'^.\.y\.\s*', '', text, flags=re.MULTILINE)
    return text

def process_text(text, cleaning_from_ocr):
    """Возвращает текст без мусора ОСR"""
    lines = text.splitlines(keepends=True)  # сохраняем \n
    processed_lines = [cleaning_from_ocr(line) for line in lines]
    return ''.join(processed_lines)

def is_tablet(text: str):
    """Ищет позицию предшествующую транслитерации с таблички
    и возвращает флаг находки и позиции начала перевода и начала транслитерации"""
    pos_tablet = re.search(r'\s*tablet\.\n', text, flags=re.MULTILINE)
    if pos_tablet is not None:
        # позиция начала перевода
        pos_tablet = pos_tablet.end()
        pos_start_tr_after_tablet = re.search(r'^.\.?\s*y\.\n', text, flags=re.MULTILINE)
        if pos_start_tr_after_tablet is not None:
            # позиция конца перевода
            pos_end_translate_tablet = pos_start_tr_after_tablet.start() - 1
            text_translate = text[pos_tablet:pos_end_translate_tablet].strip()
            # очистка от мусора текста
            text_translate = process_text(text_translate, cleaning_from_ocr)
            if looks_like_real_translation(text_translate):
                # позиция начала транслитерации после слова tablet
                pos_start_tr_after_tablet = pos_start_tr_after_tablet.end()
                text_transliterate = text[pos_start_tr_after_tablet:]
                # очистка от мусора
                result = process_text(text_transliterate, cleaning_from_ocr)
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
    pos_first_diapazon = re.search(r'\d{1,3}\s*[-–—-]\s*\d{1,3}', text[start_pos:], flags=re.MULTILINE)
    if pos_first_diapazon is not None:
        pos_first_diapazon = pos_first_diapazon.start()
        if extract_transliteration(text[:pos_first_diapazon]):
            return text[:pos_first_diapazon] if pos_first_diapazon else "", pos_first_diapazon
    return "", len(text)

def translate_after_translite_after_table(text: str, start_pos: int = 0):
    """Ищет позицию начала перевода, позицию начала транслитерации
    и возвращает транслитерацию"""
    pos_first_trl = re.search(r'^.\.?\s*y\.\n', text, flags=re.MULTILINE)
    return pos_first_trl.start()


def renumber_trust_source(text: str) -> dict:
    lines = text.splitlines()
    n = len(lines)
    dic_trlits = {}
    anchors = []  # (index, source_number)

    for i, line in enumerate(lines):
        m = re.match(r'\s*(\d+)\s*[.:]\s*(.*)', line)
        if m:
            num = int(m.group(1))
            if num % 5 == 0:
                anchors.append((i, num))

    if not anchors:
        # raise ValueError("Нет ни одного источникового номера")
        dic_trlits["1"] = text
        return dic_trlits

    result_numbers = [None] * n

    # --- сегмент ДО первого якоря (назад)
    first_idx, first_num = anchors[0]
    for i in range(first_idx, -1, -1):
        result_numbers[i] = first_num - (first_idx - i)

    # --- сегменты МЕЖДУ якорями
    for (i1, n1), (i2, n2) in zip(anchors, anchors[1:]):
        result_numbers[i1] = n1
        for i in range(i1 + 1, i2):
            result_numbers[i] = n1 + (i - i1)
        result_numbers[i2] = n2  # источник всегда побеждает

    # --- сегмент ПОСЛЕ последнего якоря
    last_idx, last_num = anchors[-1]
    for i in range(last_idx, n):
        result_numbers[i] = last_num + (i - last_idx)

    # --- сборка результата
    out = []
    for num, line in zip(result_numbers, lines):
        content = re.sub(r'^\s*\d+\s*[.:]?\s*', '', line)
        out.append(f"{num}. {content}")
        dic_trlits[num] = content

    # return "\n".join(out)
    return dic_trlits


# range_pattern = re.compile(r'(\d{1,2})\s*-\s*(\d{1,2})')

def process_text_last(text, lines_dict):

    range_pattern = re.compile(r'(\d{1,2})\s*-\s*(\d{1,2})')

    matches = list(range_pattern.finditer(text))
    if matches:
        dict_results = []
        text_results = []
        for i, match in enumerate(matches):
            start_num, end_num = map(int, match.groups())

            # границы текстового блока
            text_start = match.end()
            text_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)

            # строгая проверка диапазона
            if not all(k in lines_dict for k in range(start_num, end_num + 1)):
                continue

            # собираем строку из словаря
            dict_results.append(
                " ".join(lines_dict[k] for k in range(start_num, end_num + 1))
            )

            # текст без диапазона
            fragment = text[text_start:text_end].strip()
            text_results.append(fragment)
        # return dict_results, text_results
    else:
        dict_results = []
        text_results = []
        text_results_str = text.replace("\n", " ").strip()
        # собираем строку из словаря
        dict_results.append(
            " ".join(lines_dict.values())
        )
        result_string = " ".join(dict_results)

        pattern = r'\d{1,2}\.\s'
        results = re.sub(pattern, '', result_string)
        text_results.append(text_results_str)
        dict_results.append(results)

    return dict_results, text_results



#%%
def extract_quoted_substring(text: str, start_pos: int, pattern: str):
    """
    Ищет в строке text, начиная С позиции start_pos,
    подстроку вида: ' "текст"'.
    Возвращает:
        (substring, is_longer_than_30, closing_quote_pos)
    """
    # 1. Основной шаблон
    pattern = re.compile(pattern)

    match = pattern.search(text, start_pos)
    if not match:
        return None, None, len(text)
    start_pos = match.end() - 2
    translate = False
    # # поиск открывающей кавычки начинается С start_pos
    open_pos = find_double_quote(text, start_pos)
    if open_pos == -1:
        return None, None, len(text)
    # позиция начала текста после открывающей кавычки "
    quote_start = open_pos + 1
    # ищем закрывающую кавычку "
    quote_end = find_double_quote(text, quote_start, False)

    if quote_end == -1:
        return None, None, len(text)

    # подстрока между кавычками
    substring = text[quote_start : quote_end]

    if extract_transliteration(substring):
            return None, None, quote_end

    if len(substring) > 30:
        translate = True
    result = []
    result.append(substring)
    return result, translate, quote_end

#%%
def extract_parenthesized_substring(text: str, start_pos: int):
    """
    С позиции start_pos ищет '('.
    Возвращает:
        (substring, flag, close_pos)
    """
    # 1. найти открывающую скобку
    open_pos = text.find("(", start_pos)
    if open_pos == -1:
        return None, None, start_pos

    # 2. проверить расстояние
    if open_pos - start_pos <= 3:
        close_pos_tz = text.find(";", open_pos + 1)
        close_pos_s = text.find(")", open_pos + 1)
        if close_pos_tz != -1 and close_pos_s != -1:
            close_pos = min(close_pos_tz, close_pos_s)
        else:
            if close_pos_tz == -1:
                close_pos = close_pos_s
            if close_pos_s == -1:
                close_pos = close_pos_tz
        if close_pos == -1:
            return None, None, start_pos
        # # подстрока между скобками
        substring = text[open_pos + 1 : close_pos]

        blocks = extract_transliteration(substring)
        if not blocks:
            return None, None, close_pos
        # 4. условия
        is_long = len(substring) > 30

        flag = is_long
        result = []
        result.append(substring)
        return result, flag, close_pos
    return None, None, start_pos + 4

def find_single_quote(text: str, start_pos: int, first: bool=True):
    # 3. Поиск одинарной открывающей кавычки
    text = (
        text.replace("’", "'")
        .replace("‘", "'")
        .replace("ʼ", "'")
        .replace("ʾ", "'")
    )
    if first:
        # quote_pos_prob = text.find(" '", start_pos)
        # quote_pos_prob = quote_pos_prob + 1 if quote_pos_prob != -1 else -1
        quote_pos_abz = text.find("\n'", start_pos)
        quote_pos_abz = quote_pos_abz + 1 if quote_pos_abz != -1 else -1
        # if quote_pos_prob > 0 and quote_pos_abz > 0:
        #     quote_pos = min(quote_pos_prob, quote_pos_abz)
        # elif quote_pos_abz < 0 or quote_pos_prob < 0:
        #     quote_pos = max(quote_pos_prob, quote_pos_abz)
        # else:
        #     quote_pos = -1
        quote_pos = quote_pos_abz
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

#%%
def extract_letter_space_digit_colon_space(text: str, start_search_pos: int, pattern: str):
    pattern = re.compile(pattern, re.MULTILINE)
    next_first_pos = 0
    match = pattern.search(text, start_search_pos)
    if not match:
        return None, None, len(text)
    print(f"Найден поисковый якорь: {match.group()}")
    # ---------------------------------------------------
    pos = match.end()
    # pos = pos - 3
    pos_middle = text.find("\n", pos)
    if pos_middle == -1:
        return None, None, pos
    if pos_middle - pos > 2:
        return None, None, pos
    # начало строки поиска
    pos = text.find("\n", pos_middle + 1)
    if pos == 0:
        return None, None, len(text)
    if pos == -1:
        return None, None, pos_middle
    if pos - pos_middle > 2:
        return None, None, pos_middle
    # поиск начала транслитерации по строкам
    # следующая после якоря позиция строки
    result = ""
    result, pos_end = find_translit_by_rows(text, pos)
    if result:
        return result, True, pos_end
    else:
        return result, False, pos_end
    #
    # result = ""
    # num_row = 0
    # while pos < len(text):
    #     # строка и её первая позиция
    #     n_l, next_first_pos = get_next_line_trl(text, pos)
    #     if num_row > 1:
    #         return None, None, match.end()
    #     line_trl = []
    #     if n_l:
    #         line_trl = extract_transliteration(n_l)
    #     end_translit = 0
    #     while line_trl:
    #         # сборная транслитерация
    #         result += ("\n".join(line_trl))
    #         end_translit = next_first_pos - 1
    #         n_l, next_first_pos = get_next_line_trl(text, next_first_pos)
    #         if next_first_pos == -1:
    #             return result, True, end_translit
    #         if n_l:
    #             line_trl = extract_transliteration(n_l)
    #         else:
    #             line_trl = ""
    #     num_row += 1
    #     if result:
    #         return result, True, end_translit
    #     pos = next_first_pos
    # return result, None, next_first_pos - 1
    # -----------------------------------------------

#%%
def extract_single_quotes(text: str, start_pos: int):
    if start_pos < 0 or start_pos >= len(text):
        return None, None, start_pos
    # 1. Поиск открывающей одинарной кавычки
    qu_pos = find_single_quote(text, start_pos, True)
    if qu_pos == -1:
        return None, None, start_pos

    # 1. Поиск закрывающей одинарной кавычки
    # quote_pos = text.find("'", qu_pos+1)
    quote_pos = find_single_quote(text, qu_pos+1, False)
    if quote_pos == -1:
        return None, None, start_pos

    # 2. Проверка длины подстроки
    if quote_pos - start_pos > 1000:
        return None, None, start_pos

    # 3. Извлечение подстроки
    translate_txt = text[qu_pos+1:quote_pos]

    # 4. Возврат результата
    # print(f"Выбран перевод: {translate_txt}")
    return translate_txt, True, quote_pos

def extract_ankara(text: str, start_pos: int, pattern: str):
    if start_pos < 0 or start_pos >= len(text):
        return None, None, start_pos
    text = text.replace('TABLETLERİ u', 'TABLETLERİ II')
    pattern = re.compile(pattern)
    match = pattern.search(text, start_pos)
    if not match:
        return None, None, len(text)
    print(f"Найден поисковый якорь Ankara: {match.group()}")
    text = text[match.end():]
    res_is_tablet = is_tablet(text)
    result = []
    if res_is_tablet[0]:
        # Перевод - Транслитерация
        # очищенный от мусора текст и словарь транслитерации,
        # флаг выполнения, конец транслитерации
        return (res_is_tablet[1][0], res_is_tablet[1][1]), True, res_is_tablet[2]
    else:
        # Транслитерация - Перевод
        # очистка от мусора
        result = process_text(text, cleaning_from_ocr)
        # вывод транслитерации после якоря
        text_trlit = translate_after_translite(result)[0]
        if text_trlit and extract_transliteration(text_trlit):
            # первая позиция диапазона в переводе
            pos_start_perevod = translate_after_translite(result)[1]
            # последняя позиция перевода
            pos_end_perevod = re.search(r'^\d+:', result, flags=re.MULTILINE)
            if pos_end_perevod:
                pos_end_extract = pos_end_perevod.start()
            else:
                pos_end_extract = len(text)
            result = text[pos_start_perevod:pos_end_extract]
            # словарь с ключами номерами и строками транслитерации
            result1 = renumber_trust_source(text_trlit)
            # очищенный от мусора текст и словарь транслитерации,
            # флаг выполнения, позиция конца перевода
            return (result, result1), True, pos_end_extract
        return ("", ""), False, len(text)

def extract_after_ankara(text_dict_tr: tuple, pos_s: int):
    text_translate = text_dict_tr[0]
    list_trl_transl = process_text_last(text_translate, text_dict_tr[1])
    # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, True, pos_s



#%%
def normalize_akkadian_determinatives(text: str) -> str:
    for sup, norm in DETERMINATIVE_MAP.items():
        text = text.replace(sup, norm)
    return text

#%%
def normalize_subscripts(text: str) -> str:
    return text.translate(SUBSCRIPT_DIGITS)

#%%
def normalize_gaps(text: str) -> str:
    # порядок замен важен!
    replacements = [
        (r"\[\s*…\s*…\s*\]", "<big_gap>"),  # [… …]
        (r"\[x\]", "<gap>"),               # [x]
        (r"…", "<big_gap>"),               # …
    ]

    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text)

    return text


def normalize_for_mt(text: str) -> str:
    # 0. Базовая очистка (translate-таблица уже применяется снаружи)
    a = text
    chars_to_remove = "!?/:.<>˹˺[]⅁ᲟᲠᲢ"
    table = str.maketrans("", "", chars_to_remove)
    # удаление ненужных символов
    a = a.translate(table)
    normalize_gaps(a)
    # 4. Удаляем редакторские маркеры
    a = re.sub(r"^Pl-/\s*", "", a)  # Pl-/
    a = a.replace("\\", "")  # перенос строки
    a = a.replace(",", "")  # маркер переноса строки
    # номера строк
    a = re.sub(r"^\s*\(\s*\d+\s*(?:[-–]\s*\d+)?\s*\)\s*", "", a)
    # каталожные таблички
    a = re.sub(r"\(\s*[A-Z]\.\s*\d+\s*\)","", a)

    # удалить редакторские параграфы
    a = re.sub(r"\b§{1,2}\s*\d+\b", "", a)
    # 1. ASCII → Unicode
    for old, new in CHAR_MAP.items():
        a = a.replace(old, new)

    # 2. Надстрочные детерминативы
    a = normalize_akkadian_determinatives(a)

    # 3. Подстрочные цифры
    a = normalize_subscripts(a)


    # 5. Квадратные скобки: восстановление vs лакуны
    def handle_brackets(match):
        content = match.group(1).strip()

        # лакуны
        if content == "x":
            return "<gap>"
        if re.fullmatch(r"[.…\s]+", content):
            return "<big_gap>"

        # восстановленный текст → включаем в слово
        return "-" + content

    a = re.sub(r"\[([^\]]+)\]", handle_brackets, a)

    # 6. Финальная нормализация пробелов
    a = re.sub(r"\s+", " ", a).strip()

    return a

#%%
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
    translation_lengths = [len(sent.split()) for sent in translation_sentences]
    total_translit = len(translit_tokens)
    total_translation = sum(translation_lengths)
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

# ----------------------------------------------------------------------------------
def looks_like_real_translation(text, min_len=10):
    """Проверка: текст реально перевод, а не транслитерация/номер/каталог"""
    # if not text or not isinstance(text, str):
    #     return False
    text = text.strip()
    if len(text) < min_len:
        return False
    if "." not in text:
        return False
    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    # if digit_ratio > 0.15:
    if digit_ratio > 0.3:
        return False
    return True
# Чтобы langdetect всегда возвращал один и тот же результат для одного текста
DetectorFactory.seed = 0

def detect_language(text):
    """
    Определяет язык текста.
    Возвращает код языка, например: 'en', 'fr', 'de', 'ru'
    """
    try:
        lang = detect(text)
        return lang
    except Exception as e:
        print(f"Не удалось определить язык: {e}")
        return None

def translate_to_english(text):
    """
    Переводит текст на английский, если язык не английский.
    """
    lang = detect_language(text)

    if not lang:
        # Если язык не определён, возвращаем оригинальный текст
        return text

    if lang != 'en':
        try:
            translated_text = GoogleTranslator(source=lang, target='en').translate(text)
            return translated_text
        except Exception as e:
            print(f"Ошибка перевода: {e}")
            return text
    else:
        # Если текст уже на английском
        return text

 # texts = [
 #        \"Bonjour, comment ça va?\",           # французский
 #        \"Привет, как дела?\",                 # русский
 #        \"This text is already English.\",     # английский
 #        \"Hola, ¿cómo estás?\"                 # испанский
 #    ]
 #
 #    for t in texts:
 #        print(\"Оригинал:\", t)
 #        translated = translate_to_english(t)
 #        print(\"Перевод :\", translated)
 #        print(\"-\" * 50)"
# ----------------------------------------------------------------------------------


# def process_text_last(text: str, lines_dict: dict):
#     dict_results = []
#     text_results = []
#     range_pattern = re.compile(r'(\d{1,2})\s*-\s*(\d{1,2})')
#     matches = list(range_pattern.finditer(text))
#
#     for i, match in enumerate(matches):
#         start_num, end_num = map(int, match.groups())
#
#         # границы текстового блока
#         text_start = match.end()
#         text_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
#
#         # строгая проверка диапазона
#         if not all(k in lines_dict for k in range(start_num, end_num + 1)):
#             continue
#
#         # собираем строку из словаря
#         dict_results.append(
#             " ".join(lines_dict[k] for k in range(start_num, end_num + 1))
#         )
#
#         # текст без диапазона
#         fragment = text[text_start:text_end].strip()
#         text_results.append(fragment)
#
#     return dict_results, text_results


def process_text_and_build_csv_rows(text: str):
    """
    Обрабатывает текст ячейкеи и возвращает список строк CSV
    (без заголовка)
    """
    # списки шаблонов поиска для разных вариантов пар первого и второго блоков
    # patterns1 = [r'/k \d{2,}:', r'[A-Za-z]{3,5} \d,', r'[A-Za-z]{3,5} \(\d{4},']
    # patterns1 = [r'\d{2,}:\s(?:\d{1,3}-\d{1,3})?[:),]']
    # patterns1 = [r'\d{2,}:\s']
    # patterns1 = [r'\d{2,}:\s(?:(?:\d{1,3}-\d{1,3})?[:),])?(?:.{1,30})? "']
    # patterns1 = [r'\d{2,}:\s(?:(?:\d{1,3}-\d{1,3})?[:),])?(?:.{1,80})?\s*"']
    # patterns1 = [r'\d{2,}:\s(?:\d{1,3}-\d{1,3}[:),])?.*?\s*"']
    # patterns1 = ['r\d{2,}:\s(?:\d{1,3}-\d{1,3}[,:)]\s*)?[^"]*"']
    # patterns1 = [r'\d{2,}:\s(?:\d{1,3}-\d{1,3}[,:)]\s*)?[\s\S]*?"']
    # patterns1 = [r'\d{2,}:\s(?:\d{1,3}-\d{1,3}[:),]\s*)?[\s\S]*?\s"']
    pattern1 = r'\d{2,}:\s+(?:\d+-\d+[:,)]\s*[^"]{0,80}?\s)?"'
    pattern2 = r'[A-Z][a-z]{3,} \d{4}[a-z]?: \d+(?:[–\-]\d+)?'
    pattern3 = r'ANKARA KÜLTEPE TABLETLERİ II\n'
    pattern4 = r'^ANKARA KÜLTEPE TABLETLERİ\n$'
    # список списков шаблонов поиска первого блока
    all_patterns = [pattern1, pattern2, pattern3]
    len_arr = len(all_patterns)
    # len_arr = 1
    # список функций поиска первого блока соответствует списку списков шаблонов
    extract_function_1 = [extract_quoted_substring, extract_letter_space_digit_colon_space, extract_ankara]
    # extract_function_1 = [extract_ankara]
    # список функций поиска второго блока соответствует списку функций поиска первого блока
    extract_function_2 = [extract_parenthesized_substring, extract_single_quotes, extract_after_ankara]
    # extract_function_2 = [extract_after_ankara]
    str_txt = [""] * len_arr
    str_txt_1 = [""] * len_arr
    # str_txt = ['', '']
    # str_txt_1 = ['', '']

    i = 0
    csv_rows = []
    start_pos = 0

    while i < len_arr:
        pattern = all_patterns[i]
        print(f"Работаем с {i + 1} группой шаблонов")
        # for pattern in all_patterns:
        work = True
        while work:
            # поиск по двойным кавычкам потом по буквам пробелам цифрам
            str_txt[i % len_arr], flag, next_pos = extract_function_1[i % len_arr](text, start_pos, pattern)

            if flag:
                print("Найден 1 блок")
                if isinstance(str_txt[i % len_arr], tuple):
                    text_tuple = str_txt[i % len_arr]
                    str_txt_1[i % len_arr], flag2, close_pos = extract_function_2[i % len_arr](text_tuple, next_pos)
                else:
                    str_txt_1[i % len_arr], flag2, close_pos = extract_function_2[i % len_arr](text, next_pos)
                # поиск по круглым скобкам потом по одинарным кавычкам
                # str_txt_1[i % len_arr], flag2, close_pos = extract_function_2[i % len_arr](text, next_pos)
                if flag2:
                    print("Найден 2 блок")
                    translate_str_arr = []
                    accad_str_arr = []
                    # double_txt, double_flag, double_next_pos = extract_function_1[i % len_arr](text, next_pos, pattern)
                    # if double_flag and double_next_pos < (close_pos - len(str_txt_1[i % len_arr])):
                    #     print(f"Найден уточняющий текст {double_txt}")
                    #     str_txt[i % len_arr] = double_txt
                    #     next_pos = double_next_pos
                    match i:
                        case 0:
                            translate_str_arr = str_txt[i % len_arr]
                            accad_str_arr = str_txt_1[i % len_arr]
                        case 1:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 2:
                            if isinstance(text, tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                    num_i = 1
                    for translate_str, accad_str in zip(translate_str_arr, accad_str_arr):
                        # accad_str, translate_str = process_text_last(translate_str_1, accad_str_1)
                        # 1. Очистка перевода
                        # t = translate_str.replace("\n", " ")
                        if isinstance(translate_str, list):
                            t = " ".join(map(str, translate_str)).replace("\n", " ")
                        else:
                            t = translate_str.replace("\n", " ")

                        # 2. Очистка аккадского
                        a = accad_str.replace("\n", " ")
                        a = normalize_for_mt(a)

                        # # 3. Токенизация перевода
                        # t_sentences = sent_tokenize(t)
                        # --------------------------------------------------------------
                        # 3. Токенизация перевода
                        t_sentences = sent_tokenize(t)
                        t_sentences = [sent for sent in t_sentences if looks_like_real_translation(sent)]
                        # определение языка и перевод на английский, если перевод не английский\n",
                        t_sentences = [translate_to_english(sent) if detect_language(sent) != 'en' else sent for sent in t_sentences]
                        # ---------------------------------------------------------------------------
                        # 4. Выравнивание + маркеры
                        a = align_and_mark_sentences(a, t_sentences, marker="<sent>")

                        # 5. Склеиваем перевод обратно
                        t = " ".join(t_sentences)

                        # 6. CSV-экранирование (ОДИН РАЗ!)
                        a = a.replace('"', '""')
                        t = t.replace('"', '""')
                        print(f"\nТранслитерация{i + 1}-{num_i}\n {a}")
                        print(f"\nПеревод{i + 1}-{num_i}\n {t}")
                        print("-" * 50)
                        csv_rows.append(f'"{a}","{t}"\n')
                        num_i += 1
                    # найден 2 блок, ищем следующие первые
                    start_pos = close_pos + 1
                    print("Ищем следующий 1 блок")
                else:
                    print("Не найден 2 блок")
                    # не найден 2 блок,
                    if close_pos < len(text):
                        # ищем следующие первые
                        # start_pos = close_pos + 1
                        # print("Меняем шаблон")
                        print("Ищем следующий 1 блок")
                        # меняем шаблон
                        # work = False
                        # start_pos = 0
                        start_pos = close_pos + 1
                    else:
                        print("Прошли текст, меняем шаблон")
                        # прошли текст, меняем шаблон
                        work = False
                        start_pos = 0
            else:
                print("Не найден 1 блок")
                # не найден первый блок
                if next_pos < len(text):
                    print("Продолжаем по тексту поиск 1 блока")
                    # продолжаем идти по тексту
                    start_pos = next_pos + 1
                else:
                    print("Прошли текст, меняем шаблон")
                    # прошли текст, меняем шаблон
                    work = False
                    start_pos = 0
        # меняем шаблон
        print(f"Переходим на {i+2} группу шаблонов")
        # меняем очерёдность поиска блоков
        i += 1
    return csv_rows

#%%
# ----------------------------
# Функция разбивки перевода на предложения
# ----------------------------
def naive_sent_tokenize(text):
    """
    Разделяет текст на предложения по точкам, восклицательным и вопросительным знакам.
    Работает для английского перевода.
    """
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]
#%%
import csv
from io import StringIO

def parse_csv_line(line: str):
    reader = csv.reader(StringIO(line))
    accad_str, translate_str = next(reader)
    return accad_str, translate_str
#%%
# ----------------------------
# Выравнивание и разбивка транслитерации по <sent>
# ----------------------------
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

#%%
def print_file_head(path, n=5, encoding="utf-8"):
    with open(path, "r", encoding=encoding) as f:
        for i, line in enumerate(f):
            if i >= n:
                break
            print(f"{i}: {line.rstrip()}")

#%%
# Завантаження даних з CSV-файлу
# thiscompteca = "D:/Projects/Python/Конкурсы/Old_accad_translate"
thiscompteca = "G:/Visual Studio 2010/Projects/Python/Old_accad_translate/"
csv_file_path = thiscompteca+'/data/publications.csv'
df_trnl = pd.read_csv(csv_file_path)
# ----------------------------------------
df_trnl = df_trnl.drop_duplicates()

# df_trnl.to_csv("publications_new.csv", index=False)
# # -------------------------------------------
# # print(df_trnl[df_trnl['has_akkadian']].head(10))  # Перші 5 строк даних
# # print(df_trnl.shape)  # Dataset Shape
# # print(df_trnl.info())  # Dataset Information
# # print(df_trnl.describe())   # Statistics
# # print(df_trnl.isnull().sum())  # Missing Values
# print('\n')

# idx = df_trnl[df_trnl['has_akkadian']].head(40).index
idx = df_trnl[df_trnl['has_akkadian']].index
df_trnl.loc[idx, df_trnl.columns[2]] = (
    df_trnl.loc[idx, df_trnl.columns[2]]
    .str.replace("\\n", "\n", regex=False)
)
# --------------------------------------------------------------------
# text = "Starke 1985: 68"
# pattern = re.compile(re.escape(text), re.IGNORECASE)
#
# matches = []
#
# with open(csv_file_path, encoding='utf-8', errors='ignore') as f:
#     for i, line in enumerate(f):
#         if pattern.search(line):
#             matches.append(i)
#
# print(matches[:10])   # номера строк файла

# # -----------------------------------------------------------------------
# start = 225242        # первая строка с интересующим текстом
# count = 1512
# values = []
#
# with open(csv_file_path, encoding='utf-8', errors='ignore') as f:
#     for i, line in enumerate(f):
#         if i < start:
#             continue
#         if i >= start + count:
#             break
#         values.append(line.strip())  # тут строка целиком, потом можно взять столбец через split(';') или regex
#
# # print(i)
# # ----------------------------------------------------------------------
num = 0
num_i = 0
all_rows = []
# for val in values:
# texts = ''
# with open("output4.txt", "a", encoding="utf-8", errors="replace") as f:
for i in idx:
    print(f"{num_i + 1} текст начинаем искать")
    print(f"{num + 1} пару блоков начинаем искать.\n")
    print(f"Index = {i}\n")
    # if i == 5141:
    if i == 201325:
    # if i == 25:
    # if i == 130319:
        print("PROVERKA")
    # if i > 28:
    # print("Текст всієї статті:\n", df_trnl.at[i, df_trnl.columns[2]])
    # texts = '\n'.join(f"Index = {i}\nТекст всієї статті:\n{df_trnl.at[i, df_trnl.columns[2]]}")
    # with open("output.txt", "w", encoding="utf-8", errors="replace") as f:
    #     f.write(f"Index = {i}\nТекст всієї статті:\n{df_trnl.at[i, df_trnl.columns[2]]}")
    # if num_i > 12000:
    #     f.write(f"\n\nIndex = {i}\nТекст всієї статті:\n{df_trnl.at[i, df_trnl.columns[2]]}")
    #     f.write("\n")
    # else:
    #     f1.write(f"Index = {i}\nТекст всієї статті:\n{df_trnl.at[i, df_trnl.columns[2]]}")
    #     f.write("\n")
    #     print("Текст всієї статті всі символи:\n", repr(df_trnl.at[i, df_trnl.columns[2]]))

            # print("Назва файлу:", df_trnl.at[i, df_trnl.columns[0]])
            # print("Сторінка з текстом, що містить переклад:", df_trnl.at[i, df_trnl.columns[1]])
    # print("Текст всієї статті:\n", df_trnl.at[i, df_trnl.columns[2]])
            # print("-" * 50)
    list_row = process_text_and_build_csv_rows(df_trnl.at[i, df_trnl.columns[2]])
    # list_row = process_text_and_build_csv_rows(val)
    for row in list_row:
        if row not in all_rows:
            all_rows.append(row)
            print(f"{num + 1} пара блоков найдена.\n")
            # print(row)
            num += 1
    print(f"{num_i + 1} текст прошли")
    num_i += 1


    # for i in idx[:10]:  # первые 10 для проверки
    #     text = df_trnl.iat[i, 2]
    #     rows = process_text_and_build_csv_rows(text)
    #     print(f"Строка {i}: найдено {len(rows)} фрагментов")



new_df = split_accad_and_translate(all_rows)
# new_df.to_csv('translate_from_publication.csv', index=False, quoting=csv.QUOTE_ALL)
print("Примеры строк:")
print(new_df)
print(f"Кількість статей з перекладом: {len(idx)}\n")
# print(f"Кількість статей з перекладом: {len(values)}\n")
# print(num)
sys.exit()
print('\n')

#%%
# Завантаження даних з CSV-файлу
# thiscompteca = "C:/Users/arecs/Мій диск (2armnot@gmail.com)/Питон/Конкурси/Old_Assyrian/"
csv_file_path = thiscompteca+'/data/published_texts.csv'
df_txt = pd.read_csv(csv_file_path)
num_row = 0
for num_row in range(df_txt.shape[0]):
    if num_row > 3:
        break
    for num_col in range(df_txt.shape[1]):
        print(df_txt.iat[num_row, num_col])
    print('-' * 50)

# print(df_txt.head())  # Перші 5 строк даних
# print(df_txt.shape)  # Dataset Shape
# print(df_txt.info())  # Dataset Information
# print(df_txt.describe())   # Statistics
# print(df_txt.isnull().sum())  # Missing Values
#%%
# Завантаження даних з CSV-файлу
# thiscompteca = "C:/Users/arecs/Мій диск (2armnot@gmail.com)/Питон/Конкурси/Old_Assyrian/"
csv_file_path = thiscompteca+'/data/bibliography.csv'
df_txt = pd.read_csv(csv_file_path)


# print(df_txt.head())  # Перші 5 строк даних
# print(df_txt.shape)  # Dataset Shape
# print(df_txt.info())  # Dataset Information
# print(df_txt.describe())   # Statistics
# print(df_txt.isnull().sum())  # Missing Values

num_row = 0
for num_row in range(df_txt.shape[0]):
    # if num_row > 10:
    #     break
    for num_col in range(df_txt.shape[1]):
        if df_txt.iat[num_row, 2] == 'Mogens Trolle Larsen':
            print(df_txt.iat[num_row, num_col])
            if num_col == df_txt.shape[1] - 1:
                print('-' * 50)
#%%
# Завантаження даних з CSV-файлу
# thiscompteca = "C:/Users/arecs/Мій диск (2armnot@gmail.com)/Питон/Конкурси/Old_Assyrian/"
csv_file_path = thiscompteca+'/data/train.csv'
df_txt = pd.read_csv(csv_file_path)
num_row = 0
for num_row in range(df_txt.shape[0]):
    if num_row > 5:
        break
    for num_col in range(df_txt.shape[1]):
        print(df_txt.iat[num_row, num_col])
    print('-' * 50)

# print(df_txt.head())  # Перші 5 строк даних
# print(df_txt.shape)  # Dataset Shape
# print(df_txt.info())  # Dataset Information
# print(df_txt.describe())   # Statistics
# print(df_txt.isnull().sum())  # Missing Values
#%%
