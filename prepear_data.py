#%%
import sys
from unittest import case

import pandas as pd
# import numpy as np
import re
import nltk
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

        num_defis = text.count('-')
        if (num_defis > 0 and len(text) / num_defis - 1 > 12) or num_defis == 0:
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
    # начало строки поиска
    pos = None if start_pos == len(text) else start_pos
    if pos  is None:
        return None, len(text)
    # конец строки поиска
    end = text.find('\n', pos)
    if end == -1 and len(text) < pos <= len(text):
        end = len(text)
    # if end == -1:
        return None, end
    if end == pos and pos < len(text):
        pos = end + 1
        end = text.find('\n', pos)
        if end == -1 and pos <= len(text):
            end = len(text)
    if end == pos and pos >= len(text):
        return None, end
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

def get_next_line(text: str, start_pos: int) -> str:
    # начало строки поиска
    pos = None if start_pos == len(text) else start_pos
    if pos  is None:
        return None
    # конец строки поиска
    end = text.find('\n', pos)
    if end == -1 and pos < len(text):
        end = len(text)
    if end == -1:
        return None
    if end == pos:
        pos = end + 1
        end = text.find('\n', pos)

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

    return str_line

def count_words(text):
    return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿА-Яа-яЁё]+", text))

def detect_translate(text: str, start_pos: int) -> str:
    is_translate = False
    one_word = False
    # начало строки поиска
    pos = None if start_pos == len(text) else start_pos
    if pos  is None:
        return is_translate, len(text)
    # конец строки поиска
    end = text.find('\n', pos)
    if end == -1 and pos < len(text):
        end = len(text)
    if end == -1:
        return is_translate, len(text)
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


#%%
def extract_quoted_substring(text: str, start_pos: int, pattern: str):
    """
    Ищет в строке text, начиная С позиции start_pos,
    подстроку вида: ' "текст"'.
    Возвращает:
        (substring, is_longer_than_30, closing_quote_pos)
    """
    # pattern1 = r'(?:\d{1,3}-\d{1,3})?[:),]'
    # pattern1 = r'(?:(?:\d{1,3}-\d{1,3})?[:),])?'

    # pattern1 = re.compile(pattern1)
    # if start_pos != 0:
    #     start_pos += 1
    # 1. Основной шаблон
    pattern = re.compile(pattern)

    match = pattern.search(text, start_pos)
    if not match:
        return None, None, len(text)
    # match = pattern1.search(text, match.end())
    # print(f"Найдено якорь по шаблону кавычек {match.group()}")
    # start_pos = match.end() + 1
    start_pos = match.end() - 2
    translate = False
    # open_seq = ' "'
    # # поиск открывающей кавычки начинается С start_pos
    # open_pos = text.find(open_seq, start_pos) + 1
    open_pos = find_double_quote(text, start_pos)
    if open_pos == -1:
        return None, None, len(text)
    # # позиция начала текста после открывающей кавычки "
    # quote_start = open_pos + 1
    # # ищем закрывающую кавычку "
    # quote_end = text.find('"', quote_start)
    # if quote_end == -1:
    #     return None, None, len(text)
    # # открывающая скобка
    # open_scob = text.find('(', quote_end)
    # if open_scob == -1:
    #     return None, None, len(text)
    # # открывающая скобка дальше закрывающей кавычки более чем на 3 символа
    # while open_scob - quote_end > 3:
    #     # ищем кавычки ниже
    #     open_pos = text.find(open_seq, quote_end) + 1
    #     if open_pos == -1:
    #         return None, None, len(text)
    #     quote_end = text.find('"', open_pos + 1)
    #     if quote_end == -1:
    #         return None, None, len(text)
    #     if open_scob - quote_end < 0:
    #         # опускаем скобки ниже кавычек
    #         while open_scob - quote_end < 0:
    #             open_scob = text.find('(', quote_end)
    #             if open_scob == -1:
    #                 return None, None, len(text)
    #
    # close_scob = text.find(')', open_scob)
    # maybe_translit = text[open_scob:close_scob]
    #
    # while not extract_transliteration(maybe_translit):
    #     start_pos = close_scob + 1
    #     # ищем кавычки после скобок
    #     open_pos = text.find(open_seq, start_pos) + 1
    #     if open_pos == -1:
    #         return None, None, len(text)
    #     quote_start = open_pos + 1
    #     quote_end = text.find('"', quote_start)
    #     if quote_end == -1:
    #         return None, None, len(text)
    #     open_scob = text.find('(', quote_end)
    #     if open_scob == -1:
    #         return None, None, len(text)
    #     close_scob = text.find(')', open_scob)
    #     maybe_translit = text[open_scob:close_scob]
    #
    #
    # distance_to_open = open_pos - start_pos
    # arr_mach = []
    # start_pos_prov = start_pos
    # match_prov = match
    # distance_to_open_prov = distance_to_open
    # # дистанция от конца якоря до открытой кавычки
    # while distance_to_open_prov > 10:
    #     match_prov = pattern.search(text, start_pos_prov)
    #     if not match_prov:
    #         return None, None, len(text)
    #     match_prov = pattern1.search(text, match_prov.end())
    #     if match_prov:
    #         arr_mach.append(match_prov.end())
    #         start_pos_prov = match_prov.end()
    #         distance_to_open_prov = open_pos - start_pos_prov
    #         # якорь проскочил за открытую кавычку
    #         if distance_to_open_prov < 0:
    #             open_pos = text.find(open_seq, open_pos) + 1
    #             if open_pos == -1:
    #                 return None, None, len(text)
    #             distance_to_open_prov = open_pos - start_pos_prov
    #             while distance_to_open_prov < 0:
    #                 open_pos = text.find(open_seq, open_pos) + 1
    #                 if open_pos == -1:
    #                     return None, None, len(text)
    #                 distance_to_open_prov = open_pos - start_pos_prov
    #     else:
    #         return None, None, len(text)
    # # max_match_prov = max(arr_mach)
    # # max_end = max(match.end(), max_match_prov)
    # start_pos = start_pos_prov
    # if open_pos - start_pos > 80:
    #     return None, None, start_pos + 8
    # if open_pos == -1:
    #     return None, None, start_pos
    # print(f"Найдено якорь по шаблону кавычек {match.group()}")
    # if match.group() == "1742: 26-29:":
    #     print("SLEDIM")
    # позиция начала текста после открывающей кавычки "
    quote_start = open_pos + 1

    # ищем закрывающую кавычку "
    # quote_end = text.find('"', quote_start)
    quote_end = find_double_quote(text, quote_start, False)

    if quote_end == -1:
        return None, None, len(text)

    # подстрока между кавычками
    substring = text[quote_start : quote_end]

    # dash_count = substring.count('-')
    # aleph_count = substring.count('ℵ')
    # if dash_count > 0 or aleph_count > 0:
    #     if dash_count > 0:
    #         dash_required = len(substring) / dash_count
    #     else:
    #         dash_required = 34
    #     # много символов транслитерации
    #     if dash_required < 25 or aleph_count > 0:
    if extract_transliteration(substring):
            return None, None, quote_end

    if len(substring) > 30:
        translate = True

    return substring, translate, quote_end

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

        return substring, flag, close_pos
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
    if diff >= 100:
        return -1
    else:
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
    # if start_search_pos != 0:
    #     start_search_pos += 1
    pattern = re.compile(pattern, re.MULTILINE)

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
    pos = text.find("\n", pos_middle + 1) + 1
    if pos == 0:
        return None, None, len(text)
    if pos == -1:
        return None, None, pos_middle
    if pos - pos_middle > 2:
        return None, None, pos_middle
    result = ""
    # pos = match.end() + 1
    # поиск начала транслитерации по строкам
    # следующая после якоря позиция строки
    num_row = 0
    while pos < len(text):
        # строка и её первая позиция
        n_l, next_first_pos = get_next_line_trl(text, pos)
        # if next_first_pos >= len(text):
        #     return result, True, pos

        if num_row > 1:
            return None, None, match.end()
        # num_row += 1
        line_trl = []
        if n_l:
            line_trl = extract_transliteration(n_l)
        end_translit = 0
        while line_trl:
            # сборная транслитерация
            result += ("\n".join(line_trl))
            end_translit = next_first_pos - 1
            # последняя позиция в строке \n
            # pos = pos + len(line_trl) + 1
            # первая позиция в следующей строке
            # pos = text.find("\n", next_first_pos) + 1
            # if pos >= len(text):
            #     return result, True, next_first_pos
            # строка
            n_l, next_first_pos = get_next_line_trl(text, next_first_pos)
            if next_first_pos == -1:
                return result, True, end_translit
            if n_l:
                line_trl = extract_transliteration(n_l)
            else:
                line_trl = ""
        num_row += 1
        if result:
            return result, True, end_translit
        pos = next_first_pos
    return result, None, next_first_pos - 1

    # ----------------------------------------------------
    pos = match.end() + 1

    limit = min(pos + 6, len(text))
    # поиск начала транслитерации
    start_pos = pos
    for i in range(start_pos, limit):
        if text[i].isdigit() or text[i] == '-' or text[i] == 'ℵ' or text[i] == "–" or text[i] == ":":
            start_pos = i + 1
        else:
            start_pos = i
            break
    if start_pos == len(text):
        return None, None, len(text)
    # --------------------------------------------
    # new_line_pos = 0
    # end = text.find('\\n', start_pos)
    end = text.find('\n', start_pos)
    substring = text[start_pos:] if end == -1 else text[start_pos:end]
    while 0 < len(substring) < 30:
        start_pos = end + 1
        # end = text.find('\\n', start_pos)
        nd = text.find('\n', start_pos)
        substring = text[start_pos:] if end == -1 else text[start_pos:end]
        if end == -1:
            return None, None, len(text)
    # new_line_pos = None if end == -1 else end + 1
    # if not new_line_pos:
    #     return None, None, None
    result = ""
    while end < len(text):
        count_empty = 0
        if len(substring) == 0:
            while len(substring) == 0 and count_empty < 3:
                new_line_pos = None if end == -1 else end + 1
                # end = text.find('\\n', new_line_pos)
                end = text.find('\n', new_line_pos)
                substring = text[new_line_pos:] if end == -1 else text[new_line_pos:end]
                if end == -1:
                    return None, None, len(text)
                count_empty += 1
            # две строки после якоря нет транслитерации
            if len(substring) == 0:
                if result:
                    qu_pos = find_single_quote(text, end)
                    if qu_pos:
                        return result, True, qu_pos
                    else:
                        return result, None, end
                else:
                    return None, None, end
        # else:
        #     end = text.find('\\n', new_line_pos)
        # new_line_pos = None if end == -1 else end + 1
        # Пропускаем разделители
        if SEPARATOR_RE.match(substring):
            continue
        # if result:
        #     qu_pos = find_single_quote(text, new_line_pos)
        #     if qu_pos:
        #         return result, True, qu_pos
        #     else:
        #         return result, None, None
        # else:
        #     return None, None, new_line_pos
        substring = substring.strip()
        # Проверка 1: Соответствует ли базовому формату транслитерации?
        has_basic_format = (
                TRANSLIT_LINE_RE.match(substring) and
                MORPHEME_SEP_RE.search(substring))
        # Проверка 2: Содержит ли иностранные слова?
        has_foreign_words = FOREIGN_WORD_RE.search(substring)
        # if has_foreign_words:
            # print("Найдено слово:", has_foreign_words.group())
            # sys.exit()
        # Проверка 3: Содержит ли явные признаки НЕ транслитерации?
        is_not_translit = NOT_TRANSLIT_RE.search(substring)
        # Проверка 4: Содержит ли признаки аккадской транслитерации?
        has_akkadian_indicators = AKKADIAN_INDICATOR_RE.search(substring)
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
            result = "".join(substring)
            new_line_pos = None if end == -1 else end + 1
            # end = text.find('\\n', new_line_pos)
            end = text.find('\n', new_line_pos)
            if end == -1:
                return None, None, len(text)
            substring = text[new_line_pos:] if end == -1 else text[new_line_pos:end]
        else:
            if result:
                qu_pos = find_single_quote(text, end + 1)
                if qu_pos:
                    # print(f"Выбрана транслитерация: {result}")
                    return result, True, qu_pos
                else:
                    return result, None, end
            else:
                return None, None, end
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
    # if start_pos != 0:
    #     start_pos += 1
    pattern = re.compile(pattern)

    match = pattern.search(text, start_pos)
    if not match:
        return None, None, len(text)
    print(f"Найден поисковый якорь Ankara: {match.group()}")
    # if match.end() >= len(text):
    #     return None, None, len(text)
    result = ""
    pos = match.end() + 1
    # поиск начала транслитерации
    # следующая после якоря позиция строки
    while pos < len(text):
        n_l, next_first_pos = get_next_line_trl(text, pos)
        if next_first_pos >= len(text):
            return result, True, pos
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

def extract_after_ankara(text: str, start_pos: int):
    result = ""
    is_trans: bool = False
    n_l: str = ""
    if start_pos < 0 or start_pos >= len(text):
        return None, None, start_pos
    # позиция начала строки
    pos = start_pos + 1
    while pos < len(text) and pos != -1:
        # строка
        n_l = get_next_line(text, pos)

        if n_l:
            if extract_transliteration(n_l) == []:
                if re.match(r'^\d+:\s*', n_l):
                    break
                # здесь проверка на окончание перевода
                # и на то, что это перевод
                # ---------------------------------
                is_trans, n_l = detect_translate(n_l, 0)
                # -----------------------------------
                # не транслитерация
                if is_trans:
                    result += (" " + n_l)
        # позиция начала следующей строки
        # pos = pos + len(n_l) + 1
        pos = text.find("\n", pos) + 1
    if result:
        return result, True, pos - 1
    return result, None, pos - 1


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

#%%

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
    # patterns3 = [r'ANKARA KÜLTEPE TABLETLERİ II']
    # список списков шаблонов поиска первого блока
    all_patterns = [pattern1, pattern2]
    len_arr = len(all_patterns)
    # len_arr = 1
    # список функций поиска первого блока соответствует списку списков шаблонов
    # extract_function_1 = [extract_quoted_substring, extract_letter_space_digit_colon_space, extract_ankara]
    extract_function_1 = [extract_quoted_substring, extract_letter_space_digit_colon_space]
    # список функций поиска второго блока соответствует списку функций поиска первого блока
    # extract_function_2 = [extract_parenthesized_substring, extract_single_quotes, extract_after_ankara]
    extract_function_2 = [extract_parenthesized_substring, extract_single_quotes]
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
                # поиск по круглым скобкам потом по одинарным кавычкам
                str_txt_1[i % len_arr], flag2, close_pos = extract_function_2[i % len_arr](text, next_pos)
                if flag2:
                    print("Найден 2 блок")
                    # double_txt, double_flag, double_next_pos = extract_function_1[i % len_arr](text, next_pos, pattern)
                    # if double_flag and double_next_pos < (close_pos - len(str_txt_1[i % len_arr])):
                    #     print(f"Найден уточняющий текст {double_txt}")
                    #     str_txt[i % len_arr] = double_txt
                    #     next_pos = double_next_pos
                    match i:
                        case 0:
                            translate_str = str_txt[i % len_arr]
                            accad_str = str_txt_1[i % len_arr]
                        case 1:
                            translate_str = str_txt_1[i % len_arr]
                            accad_str = str_txt[i % len_arr]
                        case 2:
                            translate_str = str_txt[i % len_arr]
                            accad_str = str_txt_1[i % len_arr]
                    # 1. Очистка перевода
                    t = translate_str.replace("\n", " ")

                    # 2. Очистка аккадского
                    a = accad_str.replace("\n", " ")
                    a = normalize_for_mt(a)

                    # 3. Токенизация перевода
                    t_sentences = sent_tokenize(t)

                    # 4. Выравнивание + маркеры
                    a = align_and_mark_sentences(a, t_sentences, marker="<sent>")

                    # 5. Склеиваем перевод обратно
                    t = " ".join(t_sentences)

                    # 6. CSV-экранирование (ОДИН РАЗ!)
                    a = a.replace('"', '""')
                    t = t.replace('"', '""')
                    print(f"\nТранслитерация{i + 1}\n {a}")
                    print(f"\nПеревод{i + 1}\n {t}")
                    print("-" * 50)
                    csv_rows.append(f'"{a}","{t}"\n')
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
thiscompteca = "D:/Projects/Python/Конкурсы/Old_accad_translate"
# thiscompteca = "G:/Visual Studio 2010/Projects/Python/Old_accad_translate/"
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
print('\n')

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

for i in idx:
    print(f"{num_i + 1} текст начинаем искать")
    print(f"{num + 1} пару блоков начинаем искать.\n")
    if i == 202410:
    # if i == 201336:
    # if i == 25:
    # if i == 130319:
        print("PROVERKA")
    # if i > 28:
        print("Текст всієї статті:\n", df_trnl.at[i, df_trnl.columns[2]])
    #     print("Текст всієї статті всі символи:\n", repr(df_trnl.at[i, df_trnl.columns[2]]))
    print(f"index = {i}")
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
