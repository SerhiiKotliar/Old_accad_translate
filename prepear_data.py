#%%
import sys
import pandas as pd
# import numpy as np
import re
import nltk
from langdetect import detect
from langdetect import DetectorFactory
from deep_translator import GoogleTranslator
from typing import Dict, List, Tuple, Match, Pattern
from collections import defaultdict

# from micro import pos_start_translate, pos_end_translate, flag_vyp

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
# nltk.download('punkt')
# nltk.download('punkt_tab')
from nltk.tokenize import sent_tokenize

Unfin_Data: dict = {"number":"", "trlit":"", "perevod":""}
Pattern_search_translate = ""
Pattern_search_trlit = ""
Pattern_search_translate_end = ""
Pattern_search_trlit_end = ""

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
       \b(?:DINGIR|LUGAL|EN|NIN|DUMU|SAL|MUNUS|GURUŠ|LU₂|AMA|AB|AḪ|ŠEŠ|NIN₉|E₂|KI|URU|KUR|ABZU|A|IM|UD|U₄|ITI|MU|GIŠ|DU₃|GAR|GUB|TUKU|ŠU₂|ZI|NAM|ME|ŠU|IGI|DIŠ|MIN|EŠ|LIMMU|IA|KIŠIB|LÚ|AŠ|ŠA|BABBAR|KÙ|NUMUN|SU.|U.BA|TUG|NIGIN|GIN|KÙ.|TA)\b  # формулы / логограммы
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
# AKKADIAN_INDICATOR_RE = re.compile(
#     r"[ŠšḪḥṢṣṬṭʾʿ⅀⅁ᲟᲠ]|"  # Аккадские специальные символы
#     r"\[.*?\]|"  # Квадратные скобки
#     r"\(.*?\)|"  # Круглые скобки
#     r"\{.*?\}|"  # Фигурные скобки
#     r"\b[A-Z][a-zšḫṭṣ]+ℵ[a-zšḫṭṣ]+\b|"  # Слова с ℵ, начинающиеся с заглавной
#     r"\b[a-zšḫṭṣ]+ℵ[a-zšḫṭṣ]+\b|"  # Слова с ℵ из строчных
#     r"\b[A-Z][a-zšḫṭṣ]+-[a-zšḫṭṣ]+\b|"  # Слова с дефисом, начинающиеся с заглавной
#     r"\b[a-zšḫṭṣ]+-[a-zšḫṭṣ]+\b|"  # Слова с дефисом из строчных
#     r"\b\d+[rv]\b|"  # Номера строк: 14r, 15v и т.д.
#     r"x\+|x\-|x\?|x=\d+|"  # Фрагменты табличек
#     r"\.\.\.|…|"  # Многоточия
#     r"\d+['ˈ]|"  # Числа с апострофом
#     r"–[^ ]"  # Длинное тире не после пробела
# )
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
# NOT_TRANSLIT_RE = re.compile(
# #     r"\b[A-Z][a-z]{3,} [A-Z][a-z]{3,}\b|"  # Два заглавных слова подряд (имя собственное)
# #     r"\b[a-z]{4,} [a-z]{4,} [a-z]{4,}\b|"  # Три длинных слова подряд (предложение)
# #     r"^\d+ [A-Z][a-z]|"  # Начинается с цифры и заглавной буквы
# #     r"[a-z]{5,}-[a-z]{4,}[^šḫṭṣʾʿ]|"  # Длинные английские слова с дефисом
# #     r"[a-zA-ZäöüÄÖÜß]{5,}-[a-zA-ZäöüÄÖÜß]{4,}|" # Длинные немецкие слова с дефисом
# #     r"[a-zA-ZçğıİöşüÇĞİÖŞÜ]{5,}-[a-zA-ZçğıİöşüÇĞİÖŞÜ]{4,}|" # Длинные турецкие слова с дефисом
# #     r", |; |: |\. [A-Z]|"  # Знаки пунктуации с пробелом
# #     r"\b(?:[A-Za-z]+ ){3,}[A-Za-z]+\b"  # Более 3 слов подряд
# # )
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

patterns_akt2 = {
        "paren_both": r"\(\s*(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+)|(?P<number>\d+))\s*\)"
,  # (12) (12-15)
        "paren_right": r"(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+)|(?P<number>\d+))\s*\)"
,  # 12) 12-15)
        "quote_right": r"(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+)|(?P<number>\d+))'"
,  # 12' 12-15 -15'
        "plain": r"(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+)|(?P<number>\d+))"
  # 12 12-15
    }
patterns_akt2_trl_s = {
        "start_trl": r"^[ÖG~]\.\s*y\.\r?\n",  # Ö. y.
        "start_trl_paren_right": r"\d+\s*\)",  # 12)
        "start_trl_dot": r"^\d+\s*\.",  # 12.
        "start_trl_paren_both": r'\(\s*(\d{1,3})\s*\)' # (34)

    }
patterns_akt2_per_s = {
        # "plain": r"(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+))",  # 12 12-15
        "plain": r'(?<![,])(?P<start>\b\d+)\s*[-–—]\s*(?P<end>\d+\b(?!:))|[-–—]\s*(?P<only_end>\d+\b(?!:))',
        "paren_both": r"\(\s*(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+))\s*\)"
,  # (12) (12-15)
    }
patterns_akt2_per_e = {
        "plain": r'^(?:\d{1,2},)?(?:\d{1,2},)?(\d{1,2}|\d+\s*[-–—]\s*\d+):'  # 1,2,12-15:
    }
patterns_akt = {
        "paren_digit_dot_digit": r'\((?:[A-Za-z]{1,2}\.\s)?\d{1,2}\)',  # (Az. 37)
        "plain": r'\s\d{1,2}\s*[-–—]\s*\d{1,2}\s*:\s',  # 1-12:
        "paren_both": r'\(\s*(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*\)|\(\s*(\d{1,3})\s*\)',  # (3) (12-15)
        "para_quote": r'\s\"' # "
    }
# r'^[A-Z]{1}[a-z]{2,8}\s*\d{1,4}:\s*(?:\d+[–\-]\d+|\d{1,4})\n'
patterns_withaut_diapason_s = {
        "start_trl": r'^[A-Z]{1,3}[a-z]{0,2}\s*(?:\d{1,3}/k|n/k|\d{1,2},)\s*\d{1,4}[a-z]{0,2}(?::\s*(?:\d+[–\-]\d+|:|\d{1,5}))?\n',
        "start_trl_sooname": r'^[A-Z]{1}[a-z]{2,8}\s*\d{1,4}:\s*(?:\d+[–\-]\d+|\d{1,4})\n',
        "start_per_quote": r"^[A-Z]{1,3}[a-z]{0,2}\s*(?:\d{1,3}/k|n/k)\s*\d{1,4}[a-z]{0,2}:\s'"
}
patterns_withaut_diapason_per_e = {"end_per_quote": r"'\s\(\d"}

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
                if len(line_trimmed) > 25 else TRANSLIT_LINE_RE.match(line_trimmed)
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

        # # Короткие токены (ša, ina, a-na и т.п.)
        is_tokens = False
        tokens = re.findall(r"[A-Za-zšṣṭḫʾʿ]+", line_trimmed.lower())
        if tokens:
            long_tokens = [t for t in tokens if len(t) >= 4]
            short_tokens = [t for t in tokens if len(t) <= 3]
            llt = len(long_tokens)
            lst = len(short_tokens)
            lt = len(tokens)
            des = lst/lt
            if des > 0.6:
            # if llt > 0:
            #     lline = len(line_trimmed)
            #     if lline/llt > 15:
                is_tokens = True
        else:
            is_tokens = False

        # Логика принятия решения:
        # 1. Должен быть базовый формат
        # 2. Не должен содержать иностранных слов ИЛИ должен иметь аккадские индикаторы
        # 3. Не должен быть явно НЕ транслитерацией
        is_transliteration = (
                has_basic_format and is_tokens and
                (not has_foreign_words or has_akkadian_indicators) and
                not is_not_translit
        )

        # # Особый случай: если есть аккадские индикаторы, принимаем даже с некоторыми иностранными словами
        # if has_akkadian_indicators and has_basic_format and not is_not_translit:
        #     is_transliteration = True

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
        if (num_div > 0 and len(line) / num_div - 1 > 16) or num_div == 0 and not has_basic_format:
            is_transliteration = False
        # проверка количества цифр в строке
        def more_than_half_digits(line_trimmed):
            digits = sum(ch.isdigit() for ch in line_trimmed)
            return digits > len(line_trimmed) / 2

        if more_than_half_digits(line_trimmed):
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

def find_translit_by_rows(text: str, pos: int=0, n_dop: int=2):
    """поиск транслитерации начиная с позиции после якоря по строкам
    возвращает транслитерацию или "" и её позиции конца и начала"""
    pos_end_of_line = 0
    pos_start_trlit = pos
    start_detect = pos_start_trlit
    pos_start_transliteration = ""
    result = ""
    num_row = 0
    while pos < len(text):
        # строка от её первой позиции и позиция конца строки
        n_l, pos_end_of_line = get_next_line(text, pos)
        # прекращение поиска транслитерации после 2 ложных строк
        if num_row > n_dop-1:
            return "", pos_end_of_line, pos_start_trlit
        line_trl = []
        if n_l:
            line_trl = extract_transliteration(n_l)
        end_translit = 0
        while line_trl:
            # сборная транслитерация
            result += "\n".join(line_trl) + "\n"
            end_translit = pos_end_of_line
            if (pos_end_of_line - len(n_l) - 1) > 0 and pos_start_trlit == start_detect:
                pos_start_transliteration = pos_end_of_line - len(n_l) - 1
                pos_start_trlit = pos_start_transliteration
            else:
                pos_start_trlit = pos_end_of_line - len(n_l) - 1
            # строка
            n_l, pos_end_of_line = get_next_line(text, pos_end_of_line)
            if pos_end_of_line == -1:
                return result, end_translit, pos_start_trlit
            if n_l:
                line_trl = extract_transliteration(n_l)
            else:
                line_trl = ""
        num_row += 1
        pos = pos_end_of_line
        if result:
            return result, end_translit, pos_start_transliteration


    return "", pos_end_of_line, pos_start_transliteration


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

    # # Частотные служебные слова аккадского
    # if sum(1 for t in tokens if t in AKKADIAN_FUNCTION_WORDS) >= 2:
    #     return False
    # якоря начала перевода, служебные пометки номеров каталогов
    all_anchors = []
    for key, value in patterns_withaut_diapason_s.items():
        anchors = re.findall(value, text)
        all_anchors.extend(anchors)
    if len(all_anchors) > 0:
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
    if end == -1 and pos <= len(text):
        end = len(text)
        return text[pos:end], end
    # позиция старта совпадает с переводом строки
    if pos == end and pos < len(text):
        # return text[pos:end+1], end+1
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
        return "", len(text)
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

    return str_line, end+1

def count_words(text):
    return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿА-Яа-яЁё]+", text))

def detect_translate(text: str, start_pos: int):
    """подтверждает что  очищенная строка есть перевод
    выводит флаг и строку"""
    global Pattern_search_translate
    is_translate = False
    one_word = False
    # # начало строки поиска
    # pos = None if start_pos == len(text) else start_pos
    # if pos  is None:
    #     return is_translate, ""
    # # конец строки поиска
    # end = text.find('\n', pos)
    # if end == -1 and pos < len(text):
    #     end = len(text)
    # # if end == -1:
    # #     return is_translate, text[pos:end]
    # str_line = text[pos:end]
    # уборка мусора
    str_line = text
    str_line = cleaning_from_ocr(str_line)

    # шаблон диапазона страниц
    # pattern = r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b'

    str_line, count = re.subn(Pattern_search_translate, '', str_line)
    if count_words(str_line) == 1:
        one_word = True
    if count > 2 and is_translation(str_line, one_word):
        is_translate = True
    # print("Количество замен:", count)

    return is_translate, str_line


# RANGE_RE = re.compile(
#     r'\(?\s*(\d{0,3})\s*[-–—]\s*(\d{1,3})\s*\)?'
# )
# NUMBER_RE = re.compile(r'\b\d{1,3}\b')

def clear_from_ocr_for_text(text: str) -> str:
    """Упорядочивает последовательно значения диапазонов
    и оборачивает в круглые скобки"""
    # --- 1. OCR-мусор: " 3A" → "3-A"
    global Pattern_search_translate
    # text = re.sub(r'(\s\d)\s*(\d\w)', r'\1-\2', text)

    # token_pattern = re.compile(
    #     r'\(?\s*\d{1,3}\s*[-–—]\s*\d{1,3}\s*\)?'
    #     r'|\(?\s*\d{1,3}\s*\)?'
    #     r'|(?<!\d)[–—-]\s*\d{1,3}'
    #     r'|\b\d{1,3}\b'    # шаблон отдельного числа
    # )
    token_pattern = re.compile(Pattern_search_translate)
    tokens = []
    for m in token_pattern.finditer(text):
        # # пропуск чисел без признаков нумерации
        # if not m.group().startswith("(") and not m.group().endswith(")") and not m.group().endswith("'"):
        #     continue
        # token = m.group()
        # # если найдено (N) → превратить в (N-N)
        # if (token.startswith("(") and token.endswith(")")) or (token.endswith(")"))  or (token.endswith("'")):
        # inner = token[1:-1].strip()
        # if inner.isdigit():
        #     token = f"({inner}-{inner})"
        # if m:
        if m.group("start"):
            # print("полный диапазон")
            # print(m.group("start"), m.group("end"))
            tokens.append({
                "type": "range",
                "a": m.group("start"),
                "b": m.group("end"),
                "start": m.span()[0],
                "end": m.span()[1],
                # "text": m.group()
                # "text": token
            })

        elif m.group("only_end"):
            # print("неполный диапазон")
            # print("end =", m.group("only_end"))
            tokens.append({
                "type": "broken",
                "b": m.group("only_end"),
                "start": m.span()[0],
                "end": m.span()[1],
                # "text": m.group()
                # "text": token
            })

        else:
            # print("одиночное число")
            # print("number =", m.group("number"))
            tokens.append({
                "type": "single",
                "a": m.group("number"),
                "start": m.span()[0],
                "end": m.span()[1],
                # "text": m.group()
                # "text": token
            })
        # tokens.append({
        #     "start": m.start(),
        #     "end": m.end(),
        #     # "text": m.group()
        #     "text": token
        # })
        # tokens.append({
        #     "start": m.group("start"),
        #     "end": m.group("end"),
        #     # "text": m.group()
        #     # "text": token
        # })
    parsed = tokens
    if len(tokens) == 0:
        return text
    # --- 2. Разбираем токены в диапазоны
    # parsed = []
    # for t in tokens:
    #     s = t["text"]
    #     # m = re.match(r'\(?\s*(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*\)?'r'|\(?\s*\d{1,3}\s*\)?', s)
    #     m = re.match(Pattern_search, s)
    #     # m = re.match(r'\(?[^\S\n]*(\d{1,3})[^\S\n]*[-–—][^\S\n]*(\d{1,3})[^\S\n]*\)?(?=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', s)
    #     if m:
    #         parsed.append({"type": "range", "a": int(m.group(1)), "b": int(m.group(2)), **t})
    #         continue
    #
    #     m = re.match(r'[–—-]\s*(\d{1,3})', s)
    #     if m:
    #         parsed.append({"type": "broken", "b": int(m.group(1)), **t})
    #         continue
    #     # ----------------------------------------------------------
    #     # собирает отдельные числа
    #     # parsed.append({"type": "single", "n": int(s), **t})
    #     parsed.append({"type": "single", "a": int(s), **t})
    #     # -------------------------------------------------------------
    # --- 3. Исправляем логику (исправляем ПРЕДЫДУЩИЙ диапазон)
    last_range = None
    merged = []
    del_items = []
    for i, item in enumerate(parsed):
        is_del = False
        if last_range is None:
            if item["type"] == "broken":
                item["a"] = "1"
                item["type"] = "range"
            elif item["type"] == "single":
                item["b"] = item["a"]
                item["type"] = "range"
            last_range = item
            continue
        if item["type"] != "broken":
            diff = int(item["a"]) - int(last_range["b"])
        if item["type"] == "range":
            # ----------------------------------------------------
            if diff == 0:
                if int(last_range["b"]) - int(last_range["a"]) > 0:
                    last_range["b"] = str(int(item["a"]) - 1)
                else:
                    item["a"] =str(int(item["a"]) + 1)

            elif diff < 0:
                if item["a"] == item["b"]:
                    if abs(diff) > 1:
                        del_items.append(i)
                        is_del = True
                    else:
                        item["a"] = str(int(last_range["b"]) + 1)
                        item["b"] = item["a"]
                else:
                    item["a"] = str(int(last_range["b"]) + 1)
                    if last_range["a"] == last_range["b"]:
                        last_range["b"] = str(int(last_range["a"]) + 1)
            elif diff > 0:
                if item["a"] == item["b"]:
                    if diff > 1:
                        del_items.append(i)
                        is_del = True
                else:
                    item["a"] = str(int(last_range["b"])  + 1)
        elif item["type"] == "broken": # and last_range:
            item["a"] = str(int(last_range["b"]) + 1)
            item["type"] = "range"
        elif item["type"] == "single":
            if len(item["a"]) > 2 or diff> 1 or diff < -1:
                del_items.append(i)
                is_del = True
            elif diff == -1 or diff == 0:
                item["a"] = str(int(last_range["b"]) + 1)
            item["type"] = "range"
            item["b"] = item["a"]
        merged.append(last_range)
        if not is_del:
            last_range = item
    merged.append(last_range)
    # parsed = [item for i, item in enumerate(parsed) if i not in del_items]
    merged = [item for i, item in enumerate(merged) if i not in del_items]
    parsed = merged
        # ------------------------------------------------------
    # --- 4. Точечная замена (справа налево!)
    chars = list(text)

    for item in reversed(parsed):
        # if item["type"] == "range":
        repl = f"{item['a']}-{item['b']}"
        # учитывает отдельные числа
        # elif item["type"] == "single":
        #     repl = str(item["n"])
        # else:
        #     continue

        chars[item["start"]:item["end"]] = repl
    result = "".join(chars)
    # # ----------------------------------------------------
    # # если не обёрнуты, оборачивает в скобки
    # # pattern = re.compile(r'\(?(\d+)-(\d+)\)?'r'|\b\d{1,3}\b')
    # pattern = re.compile(r'\(?(\d+)-(\d+)\)?')
    #
    # def wrap_if_no_parentheses(match: re.Match) -> str:
    #     full = match.group(0)  # всё совпадение
    #     a = match.group(1)
    #     b = match.group(2)
    #
    #     if full.startswith("(") and full.endswith(")"):
    #         return full  # уже в скобках — оставить как есть
    #     else:
    #         return f"({a}-{b})"  # обернуть
    #
    # result = pattern.sub(wrap_if_no_parentheses, result)
    return result



def clear_from_ocr_for_text_last(text: str) -> str:
    """Окончательно чистит мусор и форматирует по пробелам диапазоны"""
    global Pattern_search_translate
    # pattern = re.compile(
    #     r'\(?(\d+)\s*-\s*(\d+)\)?(\s+([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]+))?'
    #     r'|[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü.\s](\d{1,2})\s?\r?\n?[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]'
    # )
    pattern = re.compile(Pattern_search_translate)

    def range_repl(m):

        # count = sum(x is not None for x in m.groups())
        # if count == 1:
        #     gr_not_None = next(x for x in m.groups() if x is not None)
        #     # оставляем только цифры
        #     digits_only = re.sub(r'\D', '', gr_not_None)
        #     return f"{digits_only}-{digits_only} "
        # # -------------------------------------------------
        # if m and m.group("start"):
        if m and m.span()[0]:
            # if not m.group("start"):
            #     return m.group(0)
            # left = m.group("start")
            # right = m.group("end")
            left = m.span()[0]
            right = m.span()[1]
            # word = m.group(4)
            word = text[right:right+8]
            if int(right) - int(left) > 10:
                # если правая часть длиннее
                if len(right) > len(left):
                    # отрезаем излишек
                    main_right = right[:len(left)]
                    # излишек
                    extra = right[len(left):]

                    # смотрим что идёт после всего совпадения
                    rest = text[m.end():]

                    #  если справа дробь или число → удаляем extra
                    if re.match(r'\s*?\d+(?:\s*/\s*\d+)?', rest):
                        return f"{left}-{main_right} "

                    #  если справа слово (захваченное)
                    if word:
                        if word[0].islower():
                            extra_conv = (
                                extra.replace('1', 'I')
                                     .replace('0', 'O')
                                     .replace('5', 'S')
                                     .replace('4', 'A')
                            )
                            return f"{left}-{main_right} {extra_conv}{word}"

                    #  иначе просто удаляем extra
                    return f"{left}-{main_right} {word}"

        return m.group(0)

    text = pattern.sub(range_repl, text)
    text = re.sub(r'\s*i0\s*', r' 10 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    # # ----------------------------------------------------
    # # последовательная сортировка диапазонов
    #
    # pattern = re.compile(r'\(?(\d+)-(\d+)\)?')
    #
    # def merge_ranges(text: str) -> str:
    #     matches = list(pattern.finditer(text))
    #     if not matches:
    #         return text
    #
    #     merged = []
    #     last = None
    #
    #     for m in matches:
    #         a = int(m.group(1))
    #         b = int(m.group(2))
    #
    #         if last is None:
    #             last = [a, b]
    #             continue
    #
    #         diff = a - last[1]
    #
    #         if diff == 0:
    #             if last[1] - last[0] > 0:
    #                 last[1] = a - 1
    #             else:
    #                 a += 1
    #
    #         elif diff < 0:
    #             a = last[1] - diff + 1
    #
    #         elif diff >= 2:
    #             last[1] = a - 1
    #
    #         merged.append(tuple(last))
    #         last = [a, b]
    #
    #     merged.append(tuple(last))
    #
    #     # 🔹 заменяем диапазоны на новые, сохраняя текст
    #     result = text
    #     for m, (a, b) in zip(reversed(matches), reversed(merged)):
    #         result = result[:m.start()] + f"({a}-{b})" + result[m.end():]
    #
    #     return result
    # text = merge_ranges(text)

    # # если не обёрнуты, оборачивает в скобки
    # # pattern = re.compile(r'\(?(\d+)-(\d+)\)?'r'|\b\d{1,3}\b')
    # pattern = re.compile(r'\(?(\d+)-(\d+)\)?')
    # # last_range = None
    # def wrap_if_no_parentheses(match: re.Match) -> str:
    #     # nonlocal last_range
    #     full = match.group(0)  # всё совпадение
    #     a = match.group(1)
    #     b = match.group(2)
    #     # if last_range:
    #     #     diff = int(a) - int(last_range["b"])
    #     #     if diff > 2 or diff < -1:
    #     #         return full
    #     #     elif diff == 2:
    #     #         a = int(last_range["b"]) + 1
    #     #
    #     # last_range = {"b": b, "a": a}
    #
    #     if full.startswith("(") and full.endswith(")"):
    #         return full  # уже в скобках — оставить как есть
    #     else:
    #         return f"({a}-{b})"  # обернуть
    #
    # text = pattern.sub(wrap_if_no_parentheses, text)
    return text


def cleaning_from_ocr_prelim(text: str) -> str:
    text = re.sub(
        r'^\s*(?:S\.(?:\s*K\.)?|K\.(?:\s*)?|v|\. v)\s*',
        '',
        text,
        flags=re.MULTILINE
    )
    text = re.sub(r'^\w\.\s*K\.\s*\w+', '', text, flags=re.MULTILINE)
    subs = [
        (r'(?<=[A-Za-z])6(?=(?:-[A-Za-z]))', 'b'),
        (r'(\d+)([A-Za-z])', r'\g<1> \g<2>'),
        (r'([a-z])ı\s*', r'\g<1>i'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])ı([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>i\g<2> '),
        (r'\s*ı\s*ı', ' 11'),
        (r'\s*ı\s*(\d)', r' 1\g<1>'),
        (r'o', '0'),
        (r'ı', '1'),
        (r'4ssur', r'Assur'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])[-–—]1([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>-i\g<2>'),
        (r'\s5([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' S\g<1>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])15([-–—])', r' \g<1>lš\g<2>'),
        (r'\s0([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' O\g<1>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])0([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>O\g<2>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])5([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>S\g<2>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>i\g<2>'),
        (r'A1', 'Ai'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1\s', r'\g<1>i '),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]),(\d)', r'\g<1> \g<2>'),
        (r'\s[iI]\s?(\d+)', r'1\g<1>'),
        (r'(?<![A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])l\s*(\d)', r' 1\g<1>'),
        (r'(?<=\d)o', '0'),
        (r'(?<=\d)°', '0'),
        (r'S([-–—])(\d)', r'5\g<1>\g<2>'),
        (r'(\d)([-–—])S', r'\g<1>\g<2>5'),
        (r'‰', ''),
        (r'™', ''),
        (r'^.*?(\d+)\s*[-–—]\s*(\d+):', r'\g<2>:'),
        (r'([^\W\d_])4(-|[^\W\d_])', r'\g<1>h\g<2>'),
        (r'(\s\d\s*)4([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>h\g<2>'),
        (r'(\w)4([-–—]\w)', r'\g<1>h\g<2>'),
        (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\g<1>h'),
        (r"r'", "r"),
        (r'(\s\d+)\s(\d+)[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]', r'\g<1>-\g<2> '),
        (r'([^\d,\s])(\d+[\s\-–—]\d+)([^\d,:\n])', r'\g<1> \g<2> \g<3>'),
        (r'(\d+[\s\-–—]\d+)([^\d,:\s*])', r'\g<1> \g<2>'),
        (r'(\d+)\s*-\s*(\d+)', r'\g<1>-\g<2>'),
        (r'^.\.?\s?y\.\s?\r?\n?', ''),
        (r'(^\d,\s)(\d{1,2})\s(\d{1,2})(^\d\n)', r'\g<1> \g<2>-\g<3> \g<4>'),
        (r'\,\n', ''),
        (r'^\n', ''),
        (r"(\d{1,2})'[-–—]\s*(\d{1,2})",r'\g<1>1-\g<2>'),
        (r"[-–—]'(\d{1,2})", r'-\g<1>'),
        (r'(\d{1,2}\s*[-–—]\s*)IS', r'\g<1>18'),
        (r"\s'(\d)\s*[-–—]", r' 1\g<1>-'),
        (r'(\w)1(\w)', r'\g<1>i\g<2>'),
        (r'K Ù\.', r'KÙ\.'),
        (r'\"\'\"', ''),
        (r'(\d)i(\d)', r'\g<1>1\g<2>'),
        (r'^\d+\r?\n(?=Kt)', ''),
        # (r'^(\d+\.)\r?\n?', r'\g<1>'),
        (r'\s[ÖO](?=[A-ZÀ-ÖØİŞĞÇÜ])', r'0 '),
        (r'(\d{1,2}\s*)\'(\s*\d{1,2})', r'\g<1>-\g<2>'),
        (r'(\d+)\s*[-–—]\s*(\d+)', r' \g<1>-\g<2> '),
        (r'(\d+)\s*`\s*(\d+)', r' \g<1>-\g<2> '),
        (r'\s*(\d)\s*(\d)\s*[-–—]\s*(\d)\s*s', r' \g<1>\g<2>-\g<3>5 '),
        (r'\s*(\d)\'\'\s*(\d+)\s*', r' \g<1>7-\g<2> '),
        (r'\s*(l)\'\'\s*(\d+)\s*', r' 17-\g<2> '),
        (r'\s*\'\'\s*(\d+)\s*', r' 7-\g<1> '),
        (r'(\d)(^\d+)', r'\g<1> \g<2>'),
        (r'^(\d)\s(\d)(\.)', r'\g<1>\g<2>\g<3>'),
        (r'[-–—]\s*(\d)\s*(\d)\s*', r'-\g<1>\g<2> '),
    ]
    for pattern, repl in subs:
        text = re.sub(pattern, repl, text, flags=re.MULTILINE)
    # r'(\D\s*)(\d{1,2})\s+(\d{1,3})(\s*[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])'
    PATTERN: re.Pattern[str] = re.compile(
         r'(^\d,\s*)(\d{1,2})\s+(\d{1,3})(\s*[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])'

    )

    def transform_last_char(char: str) -> str:
        """
        Transform trailing digit:
            5 -> S
            1 -> I
            0 -> O
            other digit -> removed
            letter -> unchanged
        """
        if char.isdigit():
            if char == "5":
                return "S"
            if char == "1":
                return "I"
            if char == "0":
                return "O"
            return ""  # why: other digits must be removed
        return char

    def conditional_replace(match: re.Match[str]) -> str:
        len_left = len(match.group(2))
        len_right = len(match.group(3))
        right_out = len_right - len_left
        last_char: str = ""
        if right_out > 0:
            right: int = int(match.group(3)[:-right_out])
            last_char = transform_last_char(match.group(3)[-right_out:])
        else:
            right: int = int(match.group(3))
        left: int = int(match.group(2))
        next_str = match.group(4).strip()

        return f"{match.group(1)} {left}-{right} {last_char}{next_str}"

    def replace_if_left_less(text: str) -> str:
        """
        Replace space with dash only if left number < right number.
        """
        return PATTERN.sub(conditional_replace, text)

    text = replace_if_left_less(text)

    PATTERN1: re.Pattern[str] = re.compile(
        r'([^\d,\s])(\d)\s(\d)-(\d{2})([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])'
    )

    def conditional_replace1(match: re.Match[str]) -> str:
        left: int = int(match.group(2) + match.group(3))
        right: int = int(match.group(4))

        if left < right:
            return f"{match.group(1)} {left}-{right} {match.group(5)}"

        return match.group(0)  # why: preserve original if condition fails

    def process_text1(text: str) -> str:
        return PATTERN1.sub(conditional_replace1, text)
    text = process_text1(text)

    pattern3 = re.compile(r'^(\d+)\n', re.MULTILINE)

    def repl(match: re.Match) -> str:
        num = int(match.group(1))
        if num % 5 == 0 and num <= 40:
            return f"{num}."
        return ""  # иначе удаляем

    text = pattern3.sub(repl, text)

    pattern2 = re.compile(r'(\d\s*[-–—]?\s*)[\"“”«»„‟]([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])')

    def replace_func2(match):
        # первая группа: убрать пробелы, оставить цифру и тире
        first = re.sub(r'\s+', '', match.group(1))
        return f"{first}11 {match.group(2)}"

    text = pattern2.sub(replace_func2, text)

    pattern4 = re.compile(r'(?<![,])(\d+)\s*[-–—]\s*(\d+)(?!:)')
    def replace_func4(match):
        left = int(match.group(1))
        right = int(match.group(2))
        if left <= 9 and len(match.group(2)) > len(match.group(1)):
            ext = len(match.group(2)) - len(match.group(1))
            last_right = match.group(2)[:-ext]
            if int(last_right) > left and int(last_right) - left <= 10:
                return f" {left}-{last_right} {ext}"
        # first = re.sub(r'\s+', '', match.group(1))
        return f" {left}-{right} "

    text = pattern4.sub(replace_func4, text)

    return text


def cleaning_from_ocr(text: str, trlit: bool = True) -> str:
    if not isinstance(text, str):
        text = str(text)
   # уборка мусора
    if trlit:
        subs = [
            (r'\:', ' '),
            (r'\!', ''),
            (r'\?', ''),
            (r'\/', ''),
            (r"\'", ''),
            (r'\"', ''),
            (r"\'\'", ''),
            (r'<([^<>]+)>', r'\g<1>'),
            (r'^.\d{1,}\n', ''),
            (r'^.\.?\s?y\.\s?\r?\n?', ''),
            (r'(?<=[A-Za-z0-9]):(?=[A-Za-z0-9])', ' '),
            # (r'([^\W\d_])4(-|[^\W\d_])', r'\g<1>h\g<2>'),
            # (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\g<1>h'),
            (r'(.)\,(.)', r'\g<1>\g<2>'),
            (r'\,\n', ''),
            (r'\\', ''),
            (r'\s*i0\s*', r' 10 '),
        ]
    else:
        subs = [
            (r':', ''),
            (r'§', 'S'),
            (r'\$', '9'),
            (r'\:', ' '),
            (r'\!', ''),
            (r'\?', ''),
            (r"\'", ''),
            (r'\"', ''),
            (r"\'\'", ''),
            (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])([-\s])9([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>\g<2>\g<3>'),
            (r'(?<=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1(?=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'i'),
            (r'(\d)S([A-Z])', r'\g<1>5\g<2>'),
            (r'\s4([a-zа-яà-öø-ÿ])', r' h\g<1>'),
            (r'(\d)\s*/\s*(\d)', r'\g<1>/\g<2>'),
            (r'\s*i0\s*', r' 10 '),
            (r'(\d{1,2}\s*)\'(\s*\d{1,2})', r'\g<1>-\g<2>'),
            (r'(?<=[a-zø-ÿışğçü])0(?=[a-zø-ÿışğçü])', 'o'),
            (r'(?<=[A-ZÀ-ÖİŞĞÇÜ])0(?=[A-ZÀ-ÖİŞĞÇÜ])', 'O'),
            (r'(?<=[a-zø-ÿışğçü])0', 'o'),
            (r'(?<=[A-ZÀ-ÖİŞĞÇÜ])0', 'O'),
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

def choose_pattern(text: str, patterns: dict[str, str], no_diapason: bool = False):

    def detect_numbering_style(text):
        counts = {}
        for name, pat in patterns.items():
            matches = re.findall(pat, text)
            counts[name] = len(matches)
        return counts

    counts = detect_numbering_style(text)
    style = max(counts, key=counts.get)
    max_keys = [k for k, v in counts.items() if v == counts[style]]
    if "para_quote" in max_keys:
        style = "para_quote"

    # # ❌ если совпадений < 3 → перевода нет
    # if counts[style] < 3:
    #     return None, "no_translate_less_3"

    pattern = patterns[style]
    if no_diapason:
        return pattern, style, "is_translate"
    compiled = re.compile(pattern)

    # --- извлекаем диапазоны ---
    ranges = []

    for m in compiled.finditer(text):
        # if m.group("start") and m.group("end"):
        #     a = int(m.group("start"))
        #     b = int(m.group("end"))
        #     if b > a:
        #         ranges.append((a, b))
        if m.span()[0] and m.span()[1]:
            a = int(m.span()[0])
            b = int(m.span()[1])
            if b > a:
                ranges.append((a, b))

    if not ranges:
        return pattern, style, "no_translate_not_ranges"

    # сортируем диапазоны
    ranges.sort()
    del_item = []
    for i, rang in enumerate(ranges):
        start = rang[0]
        end = rang[1]
        diap = end - start
        if diap > 10:
            if len(str(end)) > len(str(start)):
                str_end = str(end)
                str_end = str_end[:-1]
                if len(str_end) > len(str(start)):
                    del_item.append(i)
                else:
                    if int(str_end) - int(start) > 10:
                        del_item.append(i)
            else:
                del_item.append(i)

    if len(del_item) > 0:
        ranges = [rang for i, rang in enumerate(ranges) if i not in del_item]
    if len(ranges) == 0:
        return pattern, style, "no_translate_not_ranges"
    if len(ranges) < 3:
        return pattern, style, "no_translate_less_3"
    return pattern, style, "is_translate"

def merge_if_consecutive(d1: dict, d2: dict):
    """складывает словари транслитераций"""
    if not d1 or not d2:
        return d2  # если первый пуст — просто вернуть второй

    max1 = max(d1)
    min2 = min(d2)

    if min2 == max1 + 1:
        merged = {**d1, **d2}
        return dict(sorted(merged.items()))
    else:
        return d2  # первый удаляется


def search_for_extract_ankara(text: str, pos_start: int):
    # ------------------------------------------------
    global Pattern_search_translate
    global Pattern_search_trlit
    global patterns_akt2_per_s
    global patterns_akt2_trl_s
    text_translate = ""
    text_transliterate = ""
    text_transliterate_prev = ""
    pos_end_translate = 0
    pos_start_translate = None
    pos_end = len(text)
    pos_end_tr = 0
    pos_start_tr = 0
    flag_vyp = False
    transl_from_past = False
    trlit_from_past = False
    # поиск шаблона нумерации предложений в переводе и транслитерации
    Pattern_search_translate, style, status_translate = choose_pattern(text, patterns_akt2_per_s)
    if status_translate == "is_translate":
        # pos_start_translate_match = re.search(Pattern_search_translate, text, flags=re.MULTILINE)
        Pattern_search_translate = re.compile(Pattern_search_translate)
        matches = [
            m for m in Pattern_search_translate.finditer(text)
            if m.group("start") and m.group("end")  # исключаем only_end
        ]
        pos_start_translate_match = next(
            (
                m1 for m1, m2 in zip(matches, matches[1:])
                if (
                    m1.group("start") is not None
                    and m1.group("end") is not None
                    and int(m1.group("start")) < int(m1.group("end"))
                    and int(m1.group("end")) - int(m1.group("start")) <= 10
                    and m2.start() - m1.end() < 150
            )
            ),
            None
        )
        if pos_start_translate_match is not None:
            pos_start_translate = pos_start_translate_match.start()
            if pos_start_translate < 50:
                pos_start_translate = pos_start
            # pos_end_translate_match = re.search(r'^(?:\d{1,2},)?(?:\d{1,2},)?(\d{1,2}|\d+\s*[-–—]\s*\d+):', text, flags=re.MULTILINE)
            pattern_end_translate = re.compile(r'^(?:\d{1,2},)?(?:\d{1,2},)?(\d{1,2}|\d+\s*[-–—]\s*\d+):', flags=re.MULTILINE)
            pos_end_translate_match = pattern_end_translate.search(text, pos=pos_start_translate)
            if pos_end_translate_match is not None:
                pos_end_translate = pos_end_translate_match.start()
            else:
                pos_end_translate = len(text)
            text_translate = text[pos_start_translate:pos_end_translate]
            # if detect_translate(text_translate, 0) == False:
            if is_translation(text_translate, 0) == False:
                text_translate = ""
            if text_translate != "" and pos_end_translate == len(text):
                Unfin_Data["perevod"] = text_translate
            else:
                if pos_start_translate == pos_start:
                    if Unfin_Data["perevod"] != "":
                        text_translate = Unfin_Data["perevod"] + text_translate
                        Unfin_Data["perevod"] = ""
                        transl_from_past = True
    if text_transliterate == "":
        Pattern_search_trlit, style, status_trlit = choose_pattern(text, patterns_akt2_trl_s)
        if status_trlit == "is_translate":
            pos_start_tr_match = re.search(Pattern_search_trlit, text, flags=re.MULTILINE)
            if pos_start_tr_match:
                pos_start_tr = pos_start_tr_match.start()
                if pos_start_tr < 150:
                    pos_start_tr = pos_start
                    text_transliterate, pos_end_tr, pos_start_tr = find_translit_by_rows(text, pos_start_tr, len(text))
                if pos_start_translate is not None:
                    # text_transliterate = text[pos_start_tr:pos_start_translate]
                    text_transliterate, pos_end_tr, pos_start_tr = find_translit_by_rows(text, pos_start_tr, len(text))
                    pos_end_tr = pos_start_translate
                if text_transliterate == "":
                    text_transliterate, pos_end_tr, pos_start_tr = find_translit_by_rows(text, pos_start, len(text))
        else:
            text_transliterate, pos_end_tr, pos_start_tr = find_translit_by_rows(text, pos_start, len(text))
    if Unfin_Data['trlit'] != "":
        if pos_start_tr == pos_start:
            text_transliterate_prev = Unfin_Data['trlit']
            trlit_from_past = True
            text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
        Unfin_Data['trlit'] = ""
    if text_transliterate != "":
        # очистка от мусора(уже очищено при поиске)
        text_transliterate = process_text(text_transliterate)
        # словарь транслитерации ключ номер строки и значение строка
        text_transliterate = renumber_trust_source(text_transliterate)
        text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
    else:
        text_transliterate = text_transliterate_prev
            # далее проверить предыдущую нумерацию и сравнить с нынешней
            # при совпадении соединить, в противном случае прошлую удалить
    if text_translate == "" and text_transliterate != "" and not trlit_from_past:
        # перевода нет, возможно он будет в следующем тексте
        # и понадобится эта транслитерация для него
        Unfin_Data['trlit'] = text_transliterate
        text_transliterate = ""
        pos_end = pos_end_tr
    if text_translate != "":
        # # очистка от мусора текста
        # text_translate = process_text(text_translate, False)
        if not looks_like_real_translation(text_translate):
            text_translate = ""
    if text_transliterate != "" or text_translate != "":
        flag_vyp = True
        if pos_end_tr < pos_end_translate:
            pos_end = pos_end_translate
        else:
            pos_end = pos_end_tr
    if text_translate != "" and transl_from_past and text_transliterate == "":
        flag_vyp = False
        text_translate = ""
    return flag_vyp, (text_translate, text_transliterate), pos_end


def is_tablet(text: str)-> tuple[bool, tuple[str, dict[int, str]], int]:
    """Ищет позицию предшествующую транслитерации с таблички
    и возвращает флаг находки, перевод, словарь транслитерации, позицию конца транслитерации"""
    pos_tablet = re.search(r'\s*tablet\.\n', text, flags=re.MULTILINE)
    dictionary_trlit = {}
    perevod:str = ""
    flag_vyp = False
    pos_end = len(text)
    if pos_tablet is not None:
        flag_vyp, (perevod, dictionary_trlit), pos_end = search_for_extract_ankara(text)
        # # словарь транслитерации ключ номер строки и значение строка
    return flag_vyp, (perevod, dictionary_trlit), pos_end


def translate_after_translite(text: str, start_pos: int = 0)-> tuple[bool, tuple[str, dict[int, str]], int]:
    """Ищет позицию первого диапазона предложений в переводе
    после транслитерации и возвращает транслитерацию, если найдёт её"""
    flag_vyp, (perevod, dictionary_trlit), pos_end = search_for_extract_ankara(text)

    return flag_vyp, (perevod, dictionary_trlit), pos_end


def pos_first_translite(text: str, start_pos: int = 0):
    """Ищет позицию начала транслитерации
    и возвращает её"""
    pos_first_trl = re.search(r'^.\.?\s*y\.\n', text, flags=re.MULTILINE)
    return pos_first_trl.start() if pos_first_trl is not None else -1



import re
from typing import Dict, List, Tuple


def _extract_number(text: str) -> int:
    """Извлекает первое число из строки."""
    match = re.search(r'\d+', text)
    return int(match.group()) if match else None


def _restore_sequence(
    anchors: List[Tuple[int, int]],
    total_length: int
) -> List[int]:
    """Восстанавливает номера по якорям."""
    result = [None] * total_length

    if not anchors:
        return list(range(1, total_length + 1))

    # ДО первого якоря
    first_idx, first_num = anchors[0]
    for i in range(first_idx, -1, -1):
        result[i] = first_num - (first_idx - i)

    # МЕЖДУ якорями
    for (i1, n1), (i2, n2) in zip(anchors, anchors[1:]):
        result[i1] = n1
        for i in range(i1 + 1, i2):
            result[i] = n1 + (i - i1)
        result[i2] = n2

    # ПОСЛЕ последнего якоря
    last_idx, last_num = anchors[-1]
    for i in range(last_idx, total_length):
        result[i] = last_num + (i - last_idx)

    return result


def renumber_trust_source(text: str) -> Dict[int, str]:
    """
    Преобразует текст с частичной нумерацией (кратной 5)
    в словарь {номер: строка}.

    Поддерживаются форматы:
    - 10.
    - 10:
    - (10)
    - (abc10xyz)
    - 10)
    - 10'

    Если строка не содержит номера,
    номер восстанавливается по ближайшим якорям.
    """
    global Pattern_search_trlit
    if not text.strip():
        return {}

    pattern_line_start = re.compile(r'^\s*(\d+)\s*[.:]')
    # inline_patterns = [
    #     r'\(([^)]*\d+[^)]*)\)',
    #     r'(\d+)\)',
    #     r'(\d+)\'',
    # ]
    inline_pattern = Pattern_search_trlit


    lines = text.splitlines()

    # --- РЕЖИМ 1: текст уже разбит на строки
    if any(pattern_line_start.match(line) for line in lines):
        anchors: List[Tuple[int, int]] = []

        for idx, line in enumerate(lines):
            m = pattern_line_start.match(line)
            if m:
                num = _extract_number(m.group(0))
                if num % 5 == 0:
                    anchors.append((idx, num))

        numbers = _restore_sequence(anchors, len(lines))

        result: Dict[int, str] = {}
        for num, line in zip(numbers, lines):
            content = pattern_line_start.sub('', line, count=1).strip()
            result[num] = content

        return result

    # --- РЕЖИМ 2: сплошной текст
    # for pat in inline_patterns:
    #     compiled = re.compile(pat)
    #     matches = list(compiled.finditer(text))
    #     if not matches:
    #         continue
    #
    #     segments = []
    #     anchors: List[Tuple[int, int]] = []
    #
    #     for i, match in enumerate(matches):
    #         num = _extract_number(match.group(0))
    #         start = match.end()
    #         end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
    #         content = text[start:end].strip()
    #         segments.append(content)
    #
    #         if num % 5 == 0:
    #             anchors.append((i, num))
    #
    #     numbers = _restore_sequence(anchors, len(segments))
    #
    #     return {num: seg for num, seg in zip(numbers, segments)}

    # for pat in inline_patterns:
    def with_parent(pattern)-> bool:
        pattern = re.compile(pattern)
        text = "(12-15)  18  ( 22 )  30-31"
        for m in pattern.finditer(text):
            start, end = m.span()
            inside_parentheses = (
                    start > 0 and end < len(text)
                    and text[start - 1] == '('
                    and text[end] == ')'
            )
            if not inside_parentheses:
                return False
                # print("без скобок:", m.group())
            else:
                return True
                # print("в скобках:", m.group())



    # if inline_pattern == r"(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+)|(?P<number>\d+))":
    # if inline_pattern == r"(?<!\()\b(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+)|(?P<number>\d+))\b(?!\))":
    if not with_parent(inline_pattern):
        return {1: text.strip()}
    compiled = re.compile(inline_pattern)
    matches = list(compiled.finditer(text))
    # matches = [m.group("number") for m in compiled.finditer(text) if m.group("number")]

    if not matches:
        return {1: text.strip()}

    segments = []
    anchors: List[Tuple[int, int]] = []

    for i, match in enumerate(matches):
        # num = _extract_number(match.group(0))
        num = _extract_number(match.group("number"))
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        segments.append(content)

        if num % 5 == 0:
            anchors.append((i, num))

    numbers = _restore_sequence(anchors, len(segments))

    return {num: seg for num, seg in zip(numbers, segments)}

    # return {1: text.strip()}




def process_text_last(text: str, lines_dict: Dict[int, str]) -> Tuple[List[str], List[str]]:
    # очистка от мусора текста
    # text = process_text(text, False)
    global Pattern_search_translate
    # форматирует диапазоны пробелами
    text = clear_from_ocr_for_text_last(text)
    # print("форматирует диапазоны пробелами")
    # print(text)
    # упорядочивает по порядку последовательности диапазонов
    text = clear_from_ocr_for_text(text)
    # print("упорядочивает по порядку последовательности диапазонов")
    # print(text)

    # range_pattern = re.compile(r'(\d{1,3})(?:\s*[-–—]\s*(\d{1,3}))?')
    # шаблон диапазона
    range_pattern = re.compile(Pattern_search_translate)
    # range_pattern = re.compile(r'\b(\d{1,3})\s*[-–—]\s*(\d{1,3})\b|(?<!\d)[-–—]\s*(\d{1,3})\b')
    # разделение текста по диапазонам
    matches = list(range_pattern.finditer(text))

    dict_results: List[str] = []
    text_results: List[str] = []
    if not isinstance(lines_dict, dict):
        print(repr(lines_dict))
        raise TypeError(f"Ожидался dict, получен {type(lines_dict)}")

    if not matches:
        merged = " ".join(lines_dict.values())
        # удаление из строк словаря нумерации(это, вероятно, уже сделано ранее)
        cleaned = re.sub(r'\d{1,2}[\.:]\s*', '', merged)
        dict_results.append(cleaned)
        text_results.append(text.replace("\n", " ").strip())
        return dict_results, text_results
    # перебор текста перевода по диапазонам
    for i, match in enumerate(matches):
        if match.group("start") and match.group("end"):
            # начало дапазона
            # start = int(match.group(1))
            start = int(match.group("start"))
            # конец диапазона
            # end = int(match.group(2)) if match.group(2) else start + 1
            # end = int(match.group(2)) if match.group(2) else start
            end = int(match.group("end")) if match.group("end") else start
            # для случая типа 1-4 5-6 7-8
            keys = range(start, end+1)
            # # для случая типа 1-4 4-7 7-10
            # keys = range(start, end)
            # пропуск тех строк транслитерации, что отсутствуют в диапазонах перевода
            # или присутствуют в неполном количестве
            if not all(k in lines_dict for k in keys):
                continue

            dict_results.append(" ".join(lines_dict[k] for k in keys))
            # сбор текста из участков с диапазонами,
            # которым соответствуют имеющиеся транслитерации
            text_start = match.end()
            text_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            # fragment = text[text_start:text_end].strip(" ()")
            fragment = text[text_start:text_end].strip()
            text_results.append(fragment)

    return dict_results, text_results



#%%
def extract_quoted_substring(text: str, start_pos: int, pattern: str):
    """
    Ищет в строке text, начиная С позиции start_pos,
    подстроку вида: ' "текст"'.
    Возвращает:
        (substring, is_longer_than_30, closing_quote_pos)
    """
    # 1. Основной шаблон якоря
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

    # if extract_transliteration(substring):
    #         return None, None, quote_end

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
    # if quote_pos == -1:
    #     return quote_pos
    # дистанция от транслитерации или от перевода
    # кавычки после текста
    # diff = quote_pos - start_pos
    # if diff >= 100:
    #     return -1
    # else:
    return quote_pos

#%%
def extract_letter_space_digit_colon_space(text: str, start_search_pos: int, pattern: str):
    """предполагается что переноса перевода с одного текста на следующий
    или транслитерации нет"""
    # # предварительная очистка
    # text = cleaning_from_ocr_prelim(text)
    text_translate = ""
    flag_vyp = False
    global Pattern_search_translate, Pattern_search_trlit
    pattern = re.compile(pattern, re.MULTILINE)
    match = pattern.search(text, 0)
    if match:
        Pattern_search_trlit, style, status_trlit = choose_pattern(text, patterns_withaut_diapason_s, True)
        if status_trlit == "is_translate":
            if style == "start_per_quote":
                Pattern_search_translate = Pattern_search_trlit
                Pattern_search_trlit = ""
                pattern = re.compile(Pattern_search_translate, re.MULTILINE)
                match = pattern.search(text, start_search_pos)
                if not match:
                    return ("", ""), flag_vyp, len(text)
                pos_start_translate = match.end()
                Pattern_search_translate_end, style, status_translate = choose_pattern(text, patterns_withaut_diapason_per_e, True)
                pattern = re.compile(Pattern_search_translate_end, re.MULTILINE)
                match = pattern.search(text, pos_start_translate)
                if not match:
                    return ("", ""), flag_vyp, len(text)
                pos_end_translate = match.start()
                text_translate = text[pos_start_translate:pos_end_translate]
            else:
                pattern = re.compile(Pattern_search_trlit, re.MULTILINE)
                match = pattern.search(text, start_search_pos)
    if not match:
        return None, None, len(text)
    print(f"Найден поисковый якорь: {match.group()}")

    # ---------------------------------------------------

    # позиция начала поиска
    pos = match.end() - 1
    # транслитерация
    text_transliterate, pos_end, pos_start = find_translit_by_rows(text, pos)
    if extract_transliteration(text_transliterate):
        if text_transliterate != "":
            # словарь транслитерации ключ номер строки и значение строка
            text_transliterate = renumber_trust_source(text_transliterate)
            if pos_end < len(text):
                pos_start_translate = pos_end
                match = pattern.search(text, pos_start_translate)
                if not match:
                    pos_end_translate = len(text)
                else:
                    pos_end_translate = match.start()
                text_translate = text[pos_start_translate:pos_end_translate]
                text_translate = process_text(text_translate, False)
        else:
            return (text_translate, text_transliterate), flag_vyp, len(text)
    else:
        pos_start_translate = find_single_quote(text, pos)
        pos_end_translate = find_single_quote(text, pos_start_translate+1, False)
        text_translate =text[pos_start_translate:pos_end_translate]
        text_transliterate, pos_end, pos_start = find_translit_by_rows(text, pos_end_translate)
        if text_transliterate != "":
            # словарь транслитерации ключ номер строки и значение строка
            text_transliterate = renumber_trust_source(text_transliterate)
        else:
            return (text_translate, text_transliterate), flag_vyp, len(text)
        text_translate = process_text(text_translate, False)
    # ------------------------------------------------------------
    if is_translation(text_translate) and looks_like_real_translation(text_translate) and text_transliterate != "":
        flag_vyp = True
        return (text_translate, text_transliterate), flag_vyp, pos_end_translate
    return ("", text_transliterate), flag_vyp, len(text)

def extract_after_letter_space_digit_colon_space(text_dict_tr: tuple[str, dict[int, str]], pos_s: int) -> Tuple[Tuple[List:str, List:str], bool, int]:
    # text_translate = text_dict_tr[0]
    # list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
    list_trl_transl = [text_dict_tr[0]], [text_dict_tr[1][1]]
    # кортеж списков перевода и транслитерации, флаг, конец перевода
    return list_trl_transl, True, pos_s


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
    pattern = re.compile(pattern, re.MULTILINE)
    match = pattern.search(text, start_pos)
    if not match:
        return None, None, len(text)
    print(f"Найден поисковый якорь Ankara: {match.group()}")
    flag_vyp, (perevod, dict_trlit), pos_end = search_for_extract_ankara(text, match.end())
    return (perevod, dict_trlit), flag_vyp, pos_end

def extract_after_ankara(text_dict_tr: tuple[str, dict[int, str]], pos_s: int) -> Tuple[Tuple[List:str, List:str], bool, int]:
    # text_translate = text_dict_tr[0]
    if text_dict_tr[1] == "":
        return (text_dict_tr[0], text_dict_tr[1]), False, pos_s
    list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
    # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, True, pos_s

def _normalize_newlines(text: str) -> str:
    """
    корректная склейка строк по правилам:
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
    pos_end_trlit = 0
    pos_start_translate = 0
    pos_end_translate = 0
    text_transliterate = ""
    text_translate = ""
    flag_vyp = False
    match_start_trl_main = None
    match_end_trl_main = None
    match_end_translate_main = None
    if start_pos < 0 or start_pos >= len(text):
        return None, None, start_pos
    end_pos = len(text)
    text = text.replace('TABLETLERİ!', 'TABLETLERİi')
    pattern = re.compile(pattern)
    match = pattern.search(text, start_pos)
    if not match:
        return None, None, len(text)
    print(f"Найден поисковый якорь Ankara: {match.group()}")
    text = text[match.end():]
    # шаблон поиска перевода
    Pattern_search_translate, style, status_trlat = choose_pattern(text, patterns_akt)
    # if any(Unfin_Data.values()):
    #     number_pred = Unfin_Data['number']
    #     trlit_pred = Unfin_Data['trlit']
    #     perevod_pred = Unfin_Data['perevod']
    # предварительная очистка
    text = cleaning_from_ocr_prelim(text)
    # pattern_start_trl = r'\((?:[A-Za-z]{1,2}\.\s)?\d{1,2}\)'
    # pattern_start_trl = re.compile(pattern_start_trl)

    # pattern_start_trl_1 = r'\s\d{1,2}-\d{1,2}\s*:\s'
    # pattern_start_trl_1 = re.compile(pattern_start_trl_1)
    # pattern_end_trl_1 = r'\s\"'
    # pattern_end_trl_1 = re.compile(pattern_end_trl_1)
    # pattern_end_translate_1 = найти закрывающие двойные кавыччки

    # pattern_start_trl_2 = r'\(\s*(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*\)|\(\s*(\d{1,3})\s*\)'
    # pattern_start_trl_2 = re.compile(pattern_start_trl_2)
    pattern_end_trl_2 = r'\(\s*(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*\)|\(\s*(\d{1,3})\s*\)'
    pattern_end_trl_2 = re.compile(pattern_end_trl_2)

    # patterns_start_trl = [pattern_start_trl_1, pattern_start_trl_2]
    # patterns_end_trl = [pattern_end_trl_1, pattern_end_trl_2]

    pattern_end_translate_2 = r'\.\n[A-Za-z]{1,2}\.\s*\d{1,2}:'
    pattern_end_translate_2 = re.compile(pattern_end_translate_2)
    pos_end_translate_2_1 = len(text)
    pattern_end_translate_2_2 = r'\.\r?\n?Zarf\r?\n?'
    pattern_end_translate_2_2 = re.compile(pattern_end_translate_2_2)
    patterns_end_translate = [pattern_end_translate_2, pattern_end_translate_2_2]

    # matches_start_trl = [pattern.search(text) for pattern in patterns_start_trl]
    # for i, mach_start in enumerate(matches_start_trl):
    #     if mach_start:
    #         match_start_trl_main = matches_start_trl[i]
    match_start_trl_main = Pattern_search_translate
    # matches_end_trl = [pattern.search(text) for pattern in patterns_end_trl]
    # for i, mach_end in enumerate(matches_end_trl):
    #     if mach_end:
    #         match_end_trl_main = mach_end
    # matches_end_translate = [pattern.search(text) for pattern in patterns_end_translate]
    # for i, mach_end_t in enumerate(matches_end_translate):
    #     if mach_end_t:
    #         match_end_translate_main = mach_end_t
    if match_start_trl_main:
        pos_start_trlit = match_start_trl_main.end()
        if match_end_trl_main:
            pos_end_trlit = match_end_trl_main.start()
        else:
            pos_end_trlit = len(text)
        if pos_end_trlit <= 0:
            pos_end_trlit = len(text)
        text_transliterate = text[pos_start_trlit:pos_end_trlit]
        if pos_end_trlit < len(text):
            pos_start_translate = match_end_trl_main.end()
            # match_end_translate_main = any(matches_end_translate)
            if match_end_translate_main:
                pos_end_translate = match_end_translate_main.start()
            else:
                pos_end_translate = find_double_quote(text, pos_start_translate, False)
                if pos_end_translate <= 0:
                    pos_end_translate = len(text)
            if pos_end_translate > 0:
                text_translate = text[pos_start_translate:pos_end_translate]
            else:
                text_translate = ""
    # if match_start_trl or match_start_trl_1 or match_start_trl_2:
    #     pos_start_trlit = match_start_trl.start() if match_start_trl else match_start_trl_1.end()
    #     if match_start_trl:
    #         match_end_trl = pattern_end_trl.search(text, pos_start_trlit)
    #         if match_end_trl:
    #             pos_end_trlit = match_end_trl.start()
    #         else:
    #             pos_end_trlit = len(text)
    #     elif match_start_trl_1:
    #         pos_end_trlit = find_double_quote(text, pos_start_trlit)
    #     if pos_end_trlit <= 0:
    #         pos_end_trlit = len(text)
    #     text_transliterate = text[pos_start_trlit:pos_end_trlit]
    #     pos_start_translate = pos_end_trlit
    #     match_end_translate = pattern_end_translate.search(text, pos_start_translate)
    #     if match_end_translate:
    #         # pos_end_translate = text.find("\n", match_end_translate.start(), pos_start_translate)
    #         pos_end_translate = text.rfind("\n", pos_start_translate, match_end_translate.start())
    #
    #     else:
    #         pos_end_translate = find_double_quote(text, pos_start_translate, False)
    #     if pos_end_translate > 0:
    #         text_translate = text[pos_start_translate:pos_end_translate]
    #     else:
    #         text_translate = ""
    else:
        if Unfin_Data['trlit'] != "":
            text_transliterate = Unfin_Data['trlit']
        if match_end_trl_main:
            pos_start_translate = match_end_trl_main.end()
            match_end_translate_main = any(matches_end_translate)
            if match_end_translate_main:
                pos_end_translate = match_end_translate_main.start()
            else:
                pos_end_translate = find_double_quote(text, pos_start_translate, False)
                if pos_end_translate <= 0:
                    pos_end_translate = len(text)
            if pos_end_translate > 0:
                text_translate = text[pos_start_translate:pos_end_translate]
            else:
                text_translate = ""



    #     # match_start_translate = pattern_end_trl.search(text, 0)
    #     if match_start_translate:
    #         pos_start_translate = match_start_translate.start()
    #     else:
    #         pos_start_translate = find_double_quote(text, 0)
    #     if pos_start_translate <= 0:
    #         text_translate = ""
    #     else:
    #         match_end_translate = pattern_end_translate.search(text, pos_start_translate)
    #         if match_end_translate:
    #             # pos_end_translate = text.find("\n", match_end_translate.start(), pos_start_translate)
    #             pos_end_translate = text.rfind("\n", pos_start_translate, match_end_translate.start())
    #         else:
    #             pos_end_translate = find_double_quote(text, pos_start_translate, False)
    #         if pos_end_translate > 0:
    #             text_translate = text[pos_start_translate:pos_end_translate]
    #         else:
    #             text_translate = ""
    # # -----------------------------------------------------------------------
    # if trlit_pred != "":
    #     pos_start_trlit = pos_first_translite(text, 0)
    # else:
    #     patterns_number_before = [r'No. \d{1,}?\n', r'Tablet\n']
    #     patterns_number_after = [r'No. \d{1,}?\n', r'Zarf\n']
    #     for i, pattern_number in enumerate(patterns_number_before):
    #         pattern_number_before = re.compile(pattern_number)
    #         # pattern_number_after = re.compile(patterns_number_after[i])
    #         match_number = pattern_number_before.search(text, start_pos)
    #         if match_number:
    #             match_number_before = match_number
    #             text = text[match_number_before.end():]
    #             Unfin_Data['number'] = match_number_before.group()
    #             break
    #     pattern_trlit = re.compile(r"\(([^)]*)\)")
    #     match_trlit = pattern_trlit.search(text, start_pos)
    #     if match_trlit:
    #         pos_start_trlit = match_trlit.start()
    # text = text[pos_start_trlit:]
    # # match_number = pattern_number.search(text, start_pos)
    # # транслитерация и первая позиция диапазона в переводе
    # text_trlit, pos_start_perevod = translate_after_translite(text)
    # if pos_start_perevod == len(text):
    #     Unfin_Data['trlit'] = text_trlit
    # if not text_trlit or not extract_transliteration(text_trlit) or not pos_start_perevod:
    #     return ("", ""), False, pos_start_perevod
    # if trlit_pred and extract_transliteration(trlit_pred):
    #         text_trlit = trlit_pred + text_trlit
    # # последняя позиция перевода
    # pos_end_perevod = re.search(r'(^St\.\s\d{1,2}:)', text, flags=re.MULTILINE)
    # if pos_end_perevod:
    #     pos_end_extract = pos_end_perevod.start() - 1
    # else:
    #     pos_end_extract = len(text)
    # # перевод
    # result = text[pos_start_perevod:pos_end_extract]
    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not detect_translate(text_translate, pos_start_translate):
            return ("", ""), False, end_pos
        if pos_end_translate < 0:
            Unfin_Data['perevod'] = text_translate
    if text_transliterate != "":
        # очистка от мусора транслитерации
        text_transliterate = process_text(text_transliterate)
        # словарь с ключами номерами и строками транслитерации
        text_transliterate = renumber_trust_source(text_transliterate)
        # result1 = parse_numbered_fragments(text_transliterate)
    if text_translate != "" or text_transliterate != "":
        flag_vyp =True
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, pos_end_translate

def extract_after_ankara_next(text_dict_tr: tuple, pos_s: int):
    text_translate = text_dict_tr[0]
    list_trl_transl = process_text_last(text_translate, text_dict_tr[1])
    # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, True, pos_s



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
    chars_to_remove = "!?/:.<>™‰˹˺[]⅁ᲟᲠᲢ"
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
    # if "." not in text:
    #     return False
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

def process_text_and_build_csv_rows(text: str):
    """
    Обрабатывает текст ячейкеи и возвращает список строк CSV
    (без заголовка)
    """
    # списки шаблонов поиска якорей для разных вариантов пар первого и второго блоков
    # перевод, транслитерация
    pattern1 = r'\d{2,}:\s*(?:\d+[-–—]\d+\s*[:,)]\,?\s*[\s\S]{0,80}?)?\s*"'
    # pattern1 = r'\d{2,}:\s*(?:\d+[-–—]\d+[:,)]\s*[^"]{0,80}?)?\s*"'
    # транслитерация, перевод и наоборот
    pattern2 = r'^'
    # и то и другое а потом выбирать
    pattern3 = r'ANKARA KÜLTEPE TABLETLER[İI] II\n'
    pattern4 = r'ANKARA KÜLTEPE TABLETLER[İIi]\n'
    # список списков шаблонов поиска первого блока
    all_patterns = [pattern1, pattern2, pattern3]
    len_arr = len(all_patterns)
    # len_arr = 1
    # список функций поиска первого блока соответствует списку списков шаблонов
    # extract_function_1 = [extract_quoted_substring, extract_letter_space_digit_colon_space, extract_ankara]
    extract_function_1 = [extract_quoted_substring, extract_letter_space_digit_colon_space, extract_ankara]
    # список функций поиска второго блока соответствует списку функций поиска первого блока
    # extract_function_2 = [extract_parenthesized_substring, extract_single_quotes, extract_after_ankara]
    extract_function_2 = [extract_parenthesized_substring, extract_after_letter_space_digit_colon_space, extract_after_ankara]
    str_txt = [""] * len_arr
    str_txt_1 = [""] * len_arr
    # предварительная очистка
    text = cleaning_from_ocr_prelim(text)

    i = 0
    csv_rows = []
    start_pos = 0

    while i < len_arr:
        pattern = all_patterns[i]
        print(f"Работаем с {i + 1} группой шаблонов")
        work = True
        while work:
            str_txt[i % len_arr], flag, next_pos = extract_function_1[i % len_arr](text, start_pos, pattern)

            if flag:
                print("Найден 1 блок")
                if isinstance(str_txt[i % len_arr], tuple):
                    text_tuple = str_txt[i % len_arr]
                    str_txt_1[i % len_arr], flag2, close_pos = extract_function_2[i % len_arr](text_tuple, next_pos)
                else:
                    str_txt_1[i % len_arr], flag2, close_pos = extract_function_2[i % len_arr](text, next_pos)
                if flag2:
                    print("Найден 2 блок")
                    translate_str_arr = []
                    accad_str_arr = []
                    match i:
                        case 0:
                            translate_str_arr = str_txt[i % len_arr]
                            accad_str_arr = str_txt_1[i % len_arr]
                        case 1:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][1]
                                translate_str_arr = str_txt_1[i % len_arr][0]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 2:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 3:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]

                    if isinstance(translate_str_arr, str):
                        translate_str_arr = [translate_str_arr]
                    if isinstance(accad_str_arr, str):
                        accad_str_arr = [accad_str_arr]
                    num_i = 1
                    for translate_str, accad_str in zip(translate_str_arr, accad_str_arr):
                        # 1. Очистка перевода
                        t = translate_str.replace("\n", " ")

                        # 2. Очистка аккадского
                        a = accad_str.replace("\n", " ")
                        a = normalize_for_mt(a)

                        # # 3. Токенизация перевода
                        # t_sentences = sent_tokenize(t)
                        # --------------------------------------------------------------
                        # # 3. Токенизация перевода
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
                        print("Ищем следующий 1 блок")
                        # меняем шаблон
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

# def print_file_head(path, n=5, encoding="utf-8"):
#     with open(path, "r", encoding=encoding) as f:
#         for i, line in enumerate(f):
#             if i >= n:
#                 break
#             print(f"{i}: {line.rstrip()}")


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
# pattern = r"Starke 1985: 68"
# pattern = re.compile(pattern, re.MULTILINE)
# match = pattern.search(df_trnl.columns[2], 0)
# if match:
#     print(idx)
#     sys.exit()
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
# idx = idx[:4000]
# for val in values:
# texts = ''
# with open("output4.txt", "a", encoding="utf-8", errors="replace") as f:
for i in idx:
    print(f"{num_i + 1} текст начинаем искать")
    print(f"{num + 1} пару блоков начинаем искать.\n")
    print(f"Index = {i}\n")
    # if i == 74880:
    if i == 5185:
    #     не печатает переводы
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
print(new_df.head(10))
print(f"Кількість статей з перекладом: {len(idx)}\n")
print(f"Кількість зроблених перекладів: {new_df.shape[0]}\n")
# print(type(new_df))
# print(new_df.shape)
# print(new_df.head(5))

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
