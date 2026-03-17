#%%
import sys
import pandas as pd
import re
import nltk
from langdetect import detect
from langdetect import DetectorFactory
from deep_translator import GoogleTranslator
translator = GoogleTranslator(source="auto", target="en")
from typing import Dict, List, Tuple, Match, Pattern
# from collections import defaultdict
import csv
from io import StringIO

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
Pattern_search_translate_re = None
Pattern_search_trlit_re = None
Pattern_search_translate_end_re = None
Pattern_search_trlit_end_re = None
# номер обрабатываемого текста
number_text = 0
# номер текста в котором искали предыдущий перевод
prov_prev_transl = 0

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
    # "ᴳᴵŠ": "{geš}",
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
TRANSLIT_LINE_RE = re.compile(r'''
^(?!\s*\d+\s*$)            |   # не начинается с чистого номера
(?=.*(
        -[a-z]            |   # дефисная слоговая морфология
        \d                |   # индексные цифры (Puzur4)
       \b(?:DINGIR|LUGAL|EN|NIN|DUMU|SAL|MUNUS|GURUŠ|LU₂|AMA|AB|AḪ|ŠEŠ|NIN₉|E₂|KI|URU|KUR|ABZU|A|IM|UD|U₄|ITI|MU|GIŠ|DU₃|GAR|GUB|TUKU|ŠU₂|ZI|NAM|ME|ŠU|IGI|DIŠ|MIN|EŠ|LIMMU|IA|KIŠIB|LÚ|AŠ|ŠA|BABBAR|KÙ|NUMUN|SU.|U.BA|TUG|NIGIN|GIN|KÙ.|TA|KB|URUDU)\b  # формулы / логограммы
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

BIBLIO_RE = re.compile(
    r"""
    No\.?\s*\d+           # No. 309
    |Nr\.?\s*\d+          # Nr. 309
    |\b\d+(?:/?[a-z]+)\s+\d+\b    # 88/k 595 или 88k 595
    |\d+?\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s*\d{4}   # Фамилия и год, напр. 49 Çeçen 1995
    """,
    re.VERBOSE
)

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
# "start_trl_paren_right": r"\d+\s*\)",  # 12)
# "start_trl_dot": r"^\d+\s*\.",  # 12.
# "start_trl": r"^[ÖG~]\.\s*y\.\r?\n",  # Ö. y.
# "start_trl_paren_both": r'\(\s*(\d{1,3})\s*\)', # (34)
# "start_trl": r'\d{2,}:\s*(?:\d+[-–—]\d+\s*[:,)]\,?\s*[\s\S]{0,80}?)?\s*"'
#1
patterns_individual_assirian = {"start_trl": r'\d{2,}:\s*(?:\d+[-–—])?(?:\d+\s*[:,)]\,?\s*[\s\S]{0,80}?)?\s*"'}

#2
patterns_akt_trl_s = {"start_trl": r"^[ÖO0G~]\.\s*y\.\r?\n"}  # Ö. y.
# patterns_akt_per_s = {"plain": r'\s*\d{1,2}\s*[-–—]\s*\d{1,2}\s*',}  # 1-12
patterns_akt_per_s = {"plain": r"(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+))"}  # 12 12-15
# patterns_akt_per_s = {"plain": r"(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+))",}  # 12-15
patterns_akt_per_e = {"plain": r'^(?:\d{1,2},)?(?:\d{1,2},)?(\d{1,2}|\d+\s*[-–—]\s*\d+):'}  # 1,2,12-15:
# r'^[A-Z]{1}[a-z]{2,8}\s*\d{1,4}:\s*(?:\d+[–\-]\d+|\d{1,4})\n'
patterns_akt = {
        "paren_digit_dot_digit": r'\((?:[A-Za-z]{1,2}\.\s)?\d{1,2}\)',  # (Az. 37)
        "plain": r'\s\d{1,2}\s*[-–—]\s*\d{1,2}\s*:\s',  # 1-12:
        "paren_both": r'\(\s*(\d{1,3})\s*[-–—]\s*(\d{1,3})\s*\)|\(\s*(\d{1,3})\s*\)',  # (3) (12-15)
        "para_quote": r'\s\"' # "
    }
#3
patterns_akt2_trl_s = {
        # "start_trl_tablet": r'^Tablet\n\(1\)',
        # "start_trl_paren_both": r'^[^\d\(]*\(\s*\d{1,3}\s*\)', # (34)
        "start_trl_tablet_or_paren": r'(?:^Tablet\n\(1\)|\(\s*\d{1,3}\s*\))'
    }
patterns_akt2_per_s = {
        # "plain": r"(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|[-–—]\s*(?P<only_end>\d+))",  # 12 12-15
        # "plain": r'(?<![,(])(?:(?P<start>\b\d+)\s*[-–—]\s*(?P<end>\d+)\b|[-–—]\s*(?P<only_end>\d+)\b)(?![:)])',
        "paren_both": r"(\(\s*(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|(?P<number>\d+))\s*\))"  # (12) (12-15)
# "paren_both": r"(\(\s*(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+))\s*\))"  # (12-15)
    }
patterns_akt2_per_e = {
        # "plain": r'^(?:\d{1,2},)?(?:\d{1,2},)?(\d{1,2}|\d+\s*[-–—]\s*\d+):',  # 1,2,12-15:
        # "zarf": r'^(?:Zarf\n|Zarf parçasi\n|Zarfin |St\.(?:\s\d,)?\s(?:\d+(?:[-–—]\d+)?):)'
        "zarf": r'^(?:Zarf\r?\n|Zarf par[çc]as[ıi]\r?\n|Zarfin\s|St\.(?:\s\d{1,3},)?\s\d+(?:[-–—]\d+)?:)'
    }

#4
patterns_numbs_and_diapasones_s = {"start_numb_trl": r"^(?:F\.|\d+(?:[’'])?\.|[ÖO0G~]\.\s*y\.\r?\n|Vs\.\n|\d+\n)"}
patterns_numbs_and_diapasones_per_s = {"start_numb_per": r"^\s*(?P<start>\d+)(?:[’')])?\s*[-–—]\s*(?P<end>\d+)(?:[’')])?"}
# patterns_numbs_and_diapasones_per_s = {"start_numb_per": r'^\s*\(?(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)\)?'}
patterns_numbs_and_diapasones_per_e = {"end_numb_per": r'^(?:NOTES\s*:?\n|(?:St\.\s*)?\d+:|\d+\s*[a-z]*?\.\s*kt\s*\d+\/k\s*\d+\n)'}

#5
patterns_numbs_and_diapasones_paren_s = {"start_numb_trl": r'^\(\d+\)'}
patterns_numbs_and_diapasones_paren_per_s = {"start_numb_per": r'^\s*\((?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)\)'}
# patterns_numbs_and_diapasones_per_s = {"start_numb_per": r'^\s*\(?(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)\)?'}
patterns_numbs_and_diapasones_paren_per_e = {"end_numb_per": r'^St\.\s*\d+:'}

#6
patterns_withaut_diapason_s = {
        "start_trl": r'^(?:[A-Z]{1,3}[a-z]{0,2}\s*(?:\d{1,3}/k|n/k|\d{1,2},)\s*\d{1,4}[a-z]{0,2}(?::\s*(?:\d+[–\-]\d+|:|\d{1,5}))?\n|[A-Z]{1}[a-z]{2,8}\s*\d{1,4}:\s*(?:\d+[–\-]\d+|\d{1,4})\n)',
        # "start_trl_sooname": r'^[A-Z]{1}[a-z]{2,8}\s*\d{1,4}:\s*(?:\d+[–\-]\d+|\d{1,4})\n',
        # "start_per_quote": r"^[A-Z]{1,3}[a-z]{0,2}\s*(?:\d{1,3}/k|n/k)\s*\d{1,4}[a-z]{0,2}:\s'",
}
patterns_withaut_diapason_per_s = {"start_per_quote": r"^[A-Z]{1,3}[a-z]{0,2}\s*(?:\d{1,3}/k|n/k)\s*\d{1,4}[a-z]{0,2}:\s'"}
patterns_withaut_diapason_per_e = {"end_per_quote": r"'\s\(\d"}

#7
patterns_salim_assur_s = {"salim_start_trl": r'^(?:\d+\s*[a-z]*?\.\s*kt\s*\d+\/k\s*\d+\n)'}
patterns_salim_assur_per_s = {"salim_start_per": r"^\d+\.\d+\.\s*(?:(?:\d+\.|[A-Za-z]*\.)?(?:e\.|r\.)|\d+(?:[’'])?)\n"}
patterns_salim_assur_per_e = {"salim_end_per": r'^Notes:\n'}

#8
patterns_sebahattin_s = {"sebahat_start_trl": r''}
patterns_sebahattin_per_s = {"sebahat_start_per": r'(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+))'}
patterns_sebahattin_per_e = {"sebahat_end_per": r'^(?:No\.\s*\d+:\s*\d+\/k\s*\d+(?:\/[a-z]*)?|[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]*:)\n'}

#9
patterns_kultepe_VII_s = {"VII_start_trl": r''}
patterns_kultepe_VII_per_s = {"VII_start_per": r'(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+))'}
patterns_kultepe_VII_per_e = {"VII_end_per": r'^(?:No\.\s*\d+:\s*\d+\/k\s*\d+(?:\/[a-z]*)?|[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]*:)\n'}

# 10
patterns_byram_s = {"byram_start_trl": r''}
patterns_byram_per_s = {"byram_start_per": r'(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+))'}
patterns_byram_per_e = {"byram_end_per": r'^(?:No\.\s*\d+:\s*\d+\/k\s*\d+(?:\/[a-z]*)?|[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]*:)\n'}

# 11
patterns_babylon_s = {"babylon_start_trl": r'^(?:[A-Za-z]{0,2}\s*)?(?P<start_per>\d{1,4}\.\s*[A-Z]{0,4}\s*\d{2,6}\.)'}
patterns_babylon_per_s = {"babylon_start_per": r''}
patterns_babylon_per_e = {"babylon_end_per": r'^\d{1,4}\.\s*[A-Z]{0,4}\s*\d{2,6}\.'}

#12
patterns_ninurta_s = {"ninurta_start_trl": r'^T\s*E\s*X\s*T'}
patterns_ninurta_per_s = {"ninurta_start_per": r'(\s*(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+))\s*\'?\))'}
patterns_ninurta_per_e = {"ninurta_end_per": r'[A-Za-z]{2,5}\]?:|^Lacuna'}

#13
patterns_nabu_s = {"nabu_start_trl": r''}
patterns_nabu_per_s = {"nabu_start_per": r'^Translation:'}
patterns_nabu_per_e = {"nabu_end_per": r'^[A-Za-z]*?\d+?\:'}
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
    log = False

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
        line_trimmed = line.strip()

        # библиографические ссылки и номера табличек
        if BIBLIO_RE.search(line_trimmed):
            break

        # Пропускаем разделители
        if SEPARATOR_RE.match(line):
            continue

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

        # Проверка 2: Содержит ли иностранные слова?
        has_foreign_words = FOREIGN_WORD_RE.search(line_trimmed)

        # Проверка 4: Содержит ли признаки аккадской транслитерации?
        has_akkadian_indicators = AKKADIAN_INDICATOR_RE.search(line_trimmed)

        # Проверка 3: Содержит ли явные признаки НЕ транслитерации?
        is_not_translit = False
        if not is_transliteration:
            is_not_translit = NOT_TRANSLIT_RE.search(line_trimmed)

        # # Короткие токены (ša, ina, a-na и т.п.)
        is_tokens = False

        # НЕ понижаем регистр сразу
        tokens_raw = re.findall(r"[A-Za-zšṣṭḫʾʿ]+", line_trimmed)
        if tokens_raw:
            # логограммы = полностью ВЕРХНИЙ регистр
            logograms = [t for t in tokens_raw if t.isupper()]
            if len(logograms) == len(tokens_raw) or len(logograms) / len(tokens_raw) >= 0.5:
                log = True

            # обычные токены (не логограммы)
            tokens = [t.lower() for t in tokens_raw if not t.isupper()]
            if tokens:  # считаем только реальные слова
                long_tokens = [t for t in tokens if len(t) >= 4]
                short_tokens = [t for t in tokens if len(t) <= 3]
                lt = len(tokens)
                lst = len(short_tokens)
                des = lst / lt
                if des > 0.6:
                    is_tokens = True
            else:
                # если строка состоит только из логограмм — тоже считаем токенной
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
        num_morf = line.count("ℵ")
        num_defis = line.count('-')
        num_div = num_morf + num_defis
        # мало дефисов в строке
        count_low = sum(1 for c in line if c.islower())
        if (num_div > 0 and count_low / num_div - 1 > 8) or num_div == 0 and not has_basic_format:
            is_transliteration = False
        # проверка количества цифр в строке
        def more_than_half_digits(line_trimmed):
            digits = sum(ch.isdigit() for ch in line_trimmed)
            return digits > len(line_trimmed) / 3

        if more_than_half_digits(line_trimmed):
            is_transliteration = False

        if is_transliteration or log:
            current.append(line_trimmed)
        else:
            break

    if current:
        blocks.append("\n".join(current).strip())

    return blocks

def find_translit_by_rows(text: str, pos: int=0, n_dop: int=2):
    """поиск транслитерации начиная с позиции после якоря по строкам
    возвращает транслитерацию или "" и её позиции конца и начала"""
    pos_end_of_line = 0
    pos_start_trlit = pos
    start_detect = pos_start_trlit
    pos_start_transliteration = 0
    end_translit = 0
    result = ""
    pattern_end = r"^\d+\.\d+\.\s*(?:(?:\d+\.|[A-Za-z]*\.)?(?:e\.|r\.)|\d+(?:[’'])?)"
    num_row = 0
    while pos < len(text):
        # строка от её первой позиции и позиция конца строки
        n_l, pos_end_of_line = get_next_line(text, pos)
        # конец транслитерации
        match_nl = re.compile(pattern_end).search(n_l)
        if match_nl:
            return result, end_translit, pos
        # прекращение поиска транслитерации после 2 ложных строк
        if num_row > n_dop-1:
            return "", pos_end_of_line, pos_start_trlit
        line_trl = []
        if n_l:
            line_trl = extract_transliteration(n_l)

        while line_trl:
            if pos_start_transliteration == 0:
                pos_start_transliteration = pos
            # сборная транслитерация
            result += "\n".join(line_trl) + "\n"
            end_translit = pos_end_of_line
            # строка
            n_l, pos_end_of_line = get_next_line(text, pos_end_of_line)
            pos_start_trlit = pos_end_of_line - len(n_l) - 1
            # конец транслитерации
            match_nl = re.compile(pattern_end).search(n_l)
            if match_nl:
                return result, end_translit, pos_start_transliteration
            if pos_end_of_line == -1:
                return result, end_translit, pos_start_transliteration
            if n_l:
                line_trl = extract_transliteration(n_l)
            else:
                line_trl = ""
        num_row += 1
        pos = pos_end_of_line
        if result:
            return result, end_translit, pos_start_transliteration

    return "", pos_end_of_line, pos_start_transliteration


def find_translate_by_rows(text: str, pos: int=0, n_dop: int=2):
    """поиск перевода начиная с позиции после якоря по строкам
    возвращает транслитерацию или "" и её позиции конца и начала"""
    pos_end_of_line = 0
    pos_start_per = pos
    pos_start_translate = 0
    end_translate = 0
    result = ""
    num_row = 0
    while pos < len(text):
        # строка от её первой позиции и позиция конца строки
        n_l, pos_end_of_line = get_next_line(text, pos)

        while n_l and is_clean_akkadian_translation(n_l):
            if pos_start_translate == 0:
                pos_start_translate = pos
            # сборная транслитерация
            result += "".join(n_l)
            end_translate = pos_end_of_line
            # строка
            n_l, pos_end_of_line = get_next_line(text, pos_end_of_line)
            if pos_end_of_line == -1:
                return result, end_translate, pos_start_translate
        num_row += 1
        pos = pos_end_of_line
        if result:
            return result, end_translate, pos_start_translate

    return "", len(text), pos_start_translate




# def is_translation(text: str, one_word: bool=False) -> bool:
#     """подтверждает что строка есть перевод"""
#     if not text or len(text) < 10:
#         return False
#
#     text = text.strip()
#
#     # Морфемные цепочки → почти наверняка транслитерация
#     if MORPHEME_CHAIN_RE.search(text):
#         return False
#
#     # Слова длиной ≥ 2
#     words = WORD_RE.findall(text)
#     if len(words)< 2 and not one_word:
#         return False
#
#     # # Короткие токены (ša, ina, a-na и т.п.)
#     tokens = re.findall(r"[A-Za-zšṣṭḫʾʿ]+", text.lower())
#     if tokens:
#         short_tokens = [t for t in tokens if len(t) <= 3]
#         if len(short_tokens) / len(tokens) > 0.6:
#             return False
#
#     # # Частотные служебные слова аккадского
#     # if sum(1 for t in tokens if t in AKKADIAN_FUNCTION_WORDS) >= 2:
#     #     return False
#     # якоря начала перевода, служебные пометки номеров каталогов
#     all_anchors = []
#     for key, value in patterns_withaut_diapason_s.items():
#         anchors = re.findall(value, text)
#         all_anchors.extend(anchors)
#     if len(all_anchors) > 0:
#         return False
#
#     return True



def is_clean_akkadian_translation(text: str) -> bool:
    """
    Возвращает True, если текст похож на чистый перевод с аккадского
    без транслитерации и научных комментариев.
    """

    if not text or not text.strip():
        return False

    # 1. Признаки аккадской транслитерации
    transliteration_patterns = [
        r'\b[a-z]+(?:-[a-z]+){5,}\b',           # a-na, i-na, ša-ma
        # r'[ŠšḪḫṬṭṢṣĀāĒēĪīŪū]',          # диакритика
        r'\b[A-Z]+\.[A-Z.]+\b',         # KÙ.BABBAR
    ]

    for pattern in transliteration_patterns:
        if re.search(pattern, text):
            return False

    # 2. Научные ссылки
    scholarly_patterns = [
        r'\b\d{4}\b',                   # год (1985)
        r'\b(cf\.?|see|vgl\.?)\b',      # cf., see
        r'\b[Kk][Tt]\s*n/?k\b',         # KT n/k
        r'\b[Kk][Bb][Oo]\b',            # KBo
    ]

    for pattern in scholarly_patterns:
        if re.search(pattern, text):
            return False

    # 3. Слишком много заглавных слов (как в каталогах)
    uppercase_words = re.findall(r'\b[A-Z]{3,}\b', text)
    if len(uppercase_words) > 3:
        return False

    # 4. Проверка что текст состоит в основном из букв и обычной пунктуации
    allowed_ratio = len(re.findall(r'[A-Za-zА-Яа-я ,.\-–—?!:;()\n]', text)) / len(text)
    if allowed_ratio < 0.75:
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
        pos = end + 1
        end = text.find('\n', pos)
        if end == -1 and pos <= len(text):
            end = len(text)
            return text[pos:end], end
    # достигнут конец текста
    if end == pos and pos >= len(text):
        return "", len(text)
    str_line = text[pos:end]

    return str_line, end+1

def count_words(text):
    return len(re.findall(r"[A-Za-zÀ-ÖØ-öø-ÿА-Яа-яЁё]+", text))

# def detect_translate(text: str, start_pos: int):
#     """подтверждает что  очищенная строка есть перевод
#     выводит флаг и строку"""
#     global Pattern_search_translate
#     is_translate = False
#     one_word = False
#     # # начало строки поиска
#     # pos = None if start_pos == len(text) else start_pos
#     # if pos  is None:
#     #     return is_translate, ""
#     # # конец строки поиска
#     # end = text.find('\n', pos)
#     # if end == -1 and pos < len(text):
#     #     end = len(text)
#     # # if end == -1:
#     # #     return is_translate, text[pos:end]
#     # str_line = text[pos:end]
#     # уборка мусора
#     str_line = text
#     str_line = cleaning_from_ocr(str_line)
#
#     # шаблон диапазона страниц
#     # pattern = r'\b\d{1,3}\s*[-–—-]\s*\d{1,3}\b'
#
#     str_line, count = re.subn(Pattern_search_translate, '', str_line)
#     if count_words(str_line) == 1:
#         one_word = True
#     if count > 2 and is_translation(str_line, one_word):
#         is_translate = True
#     # print("Количество замен:", count)
#
#     return is_translate, str_line


# RANGE_RE = re.compile(
#     r'\(?\s*(\d{0,3})\s*[-–—]\s*(\d{1,3})\s*\)?'
# )
# NUMBER_RE = re.compile(r'\b\d{1,3}\b')

def clear_from_ocr_for_text(text: str) -> str:
    """Упорядочивает последовательно значения диапазонов
    и оборачивает в круглые скобки"""
    # --- 1. OCR-мусор: " 3A" → "3-A"
    global Pattern_search_translate_re

    token_pattern = Pattern_search_translate_re
    tokens = []
    for m in token_pattern.finditer(text):
        if "start" in m.re.groupindex and m.group("start"):
            tokens.append({
                "type": "range",
                "a": m.group("start"),
                "b": m.group("end"),
                "start": m.span()[0],
                "end": m.span()[1],
            })
        elif "only_end" in m.re.groupindex and m.group("only_end"):
            tokens.append({
                "type": "broken",
                "b": m.group("only_end"),
                "start": m.span()[0],
                "end": m.span()[1],
            })
        else:
            tokens.append({
                "type": "single",
                "a": m.group("number"),
                "start": m.span()[0],
                "end": m.span()[1],
            })
    parsed = tokens
    if len(tokens) == 0:
        return text

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
                    if last_range["a"] == last_range["b"]:
                        last_range["b"] = str(int(last_range["a"]) + 1)
                    item["a"] = str(int(last_range["b"]) + 1)
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
    merged = [item for i, item in enumerate(merged) if i not in del_items]
    parsed = merged
        # ------------------------------------------------------
    # --- 4. Точечная замена (справа налево!)
    chars = list(text)

    for item in reversed(parsed):
        repl = f"{item['a']}-{item['b']}"
        chars[item["start"]:item["end"]] = repl
    result = "".join(chars)
    # ----------------------------------------------------
    # если не обёрнуты, оборачивает в скобки
    # pattern = re.compile(r'\(?(\d+)-(\d+)\)?'r'|\b\d{1,3}\b')
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
    # pattern = re.compile(r'\(?(\d+)-(\d+)\)?')
    #
    # def wrap_if_no_parentheses(match: re.Match) -> str:
    #     full = match.group(0)  # всё совпадение
    #     a = match.group(1)
    #     b = match.group(2)
    #
    #     # если нет открывающей скобки слева — оборачиваем
    #     if not full.startswith("("):
    #         return f"({a}-{b})"
    #     else:
    #         return full  # оставляем как есть
    return result



def clear_from_ocr_for_text_last(text: str) -> str:
    """форматирует по пробелам диапазоны"""
    global Pattern_search_translate_re
    # pattern = re.compile(Pattern_search_translate)
    pattern = Pattern_search_translate_re
    def range_repl(m):
        # if not m.group("start"):
        #     return m.group(0)
        if m and m.span()[0] and m.groupdict().get("start") and m.groupdict().get("end"):
            # if not m.group("start"):
            #     return m.group(0)
            left_gr = m.group("start")
            right_gr = m.group("end")
            left = m.span()[0]
            right = m.span()[1]
            # word = m.group(4)
            word = text[int(right):int(right)+8]
            if int(right_gr) - int(left_gr) > 10:
                # если правая часть длиннее
                if len(right_gr) > len(left_gr):
                    # отрезаем излишек
                    main_right = right_gr[:len(left_gr)]
                    # излишек
                    extra = right_gr[len(left_gr):]

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
                            return f"{left_gr}-{main_right} {extra_conv}{word}"

                        #  иначе просто удаляем extra
                        return f"{left_gr}-{main_right} {word}"

        return m.group(0)

    text = pattern.sub(range_repl, text)
    text = re.sub(r'\s*i0\s*', r' 10 ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'\(\s*(\d+)\s*[-–—]?\s*(\d+)?\s*\)',
                  lambda m: f"({m.group(1)}-{m.group(2)})" if m.group(2) else f"({m.group(1)})",
                  text)
    # # ----------------------------------------------------
    return text


def cleaning_from_ocr_prelim(text: str) -> str:
    text = re.sub(
        r'^(?:\s*(?:S\.(?:\s*K\.)?|K\.(?:\s*)?|v|V|• v|V ~|\. v)\s*)\n',
        '',
        text,
        flags=re.MULTILINE
    )
    text = re.sub(r'^\w\.\s*K\.\s*\w+', '', text, flags=re.MULTILINE)
    subs = [
        # (r'\s+', ' '),
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
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])0([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>o\g<2>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])5([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>s\g<2>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>i\g<2>'),
        (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])6([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r' \g<1>b\g<2>'),
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
        (r'¥', 'y'),
        (r'ª', 'a'),
        (r'#', 'h'),
        (r'\s*°', ' '),
        (r'¿', 'š'),
        (r'§', 'S'),
        # (r'\$', 's'),
        (r'\(obv\.\)', ''),
        (r'\(Vs\.\)', ''),
        (r'\st\s*0\s', ' to '),
        # (r'^.*?(\d+)\s*[-–—]\s*(\d+):', r'\g<2>:'),
        (r'(\d+)\s*[-–—]\s*(\d+):', r'\g<2>:'),
        (r'([^\W\d_])4(-|[^\W\d_])', r'\g<1>h\g<2>'),
        (r'(\s\d\s*)4([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>h\g<2>'),
        (r'(\w)4([-–—]\w)', r'\g<1>h\g<2>'),
        (r'(?<!\d)([^\W\d_])4(?=[-–—])', r'\g<1>h'),
        (r"r'", "r"),
        (r'(\s\d+)\s(\d+)[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]', r'\g<1>-\g<2> '),
        (r'([^\d,\s])(\d+[\s\-–—]\d+)([^\d,:\n])', r'\g<1> \g<2> \g<3>'),
        (r'(\d+[\s\-–—]\d+)([^\d,:\s*])', r'\g<1> \g<2>'),
        (r'(\d+)\s*-\s*(\d+)', r'\g<1>-\g<2>'),
        (r'^\*Met.: tám', ''),
        (r'^A\.\s?y\.\s?\r?\n?', ''),
        (r'^S\.\s?K\.\s?\r?\n?', ''),
        (r'^K\.\s?\r?\n?', ''),
        (r'(^\d,\s)(\d{1,2})\s(\d{1,2})(^\d\n)', r'\g<1> \g<2>-\g<3> \g<4>'),
        (r'^[A-Za-z]{1,3}\.(\d{1,2})', r'\g<1>'),
        (r'\,\n', ''),
        (r'^\s*\n', ''),
        (r'^\n', ''),
        (r'^\s*[^A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]?[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü]{1,3}\s*\n', ''),
        (r"(\d{1,2})'[-–—]\s*(\d{1,2})", r'\g<1>1-\g<2>'),
        (r"[-–—]'(\d{1,2})", r'-\g<1>'),
        (r'(\d{1,2}\s*[-–—]\s*)IS', r'\g<1>18'),
        (r"\s'(\d)\s*[-–—]", r' 1\g<1>-'),
        (r'(\w)1(\w)', r'\g<1>i\g<2>'),
        (r'K Ù\.', r'KÙ\.'),
        (r'\"\'\"', ''),
        (r'(\d)i(\d)', r'\g<1>1\g<2>'),
        (r'^\d+\r?\n(?=Kt)', ''),
        (r'^(\d+)\r?\n(?=[^\n]*\w-\w)', r'\1.'),
        (r'^(\d+[\.\)\'])\r?\n?', r'\g<1>'),
        (r'^(\d+\'\))\r?\n?', r'\g<1>'),
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
        (r'(\s\d)\s\/', r'\g<1>/'),
        (r'^...\s*\n', '...')
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
            # (r'§', 'S'),
            # (r'\$', '9'),
            (r'\$', 's'),
            (r'\:', ' '),
            (r'\!', ''),
            (r'\?', ''),
            (r"\'", ''),
            (r'\"', ''),
            (r"\'\'", ''),
            (r'([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])([-\s])9([A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'\g<1>\g<2>g\g<3>'),
            (r'(?<=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])1(?=[A-Za-zÀ-ÖØ-öø-ÿİıŞşĞğÇçÜü])', r'i'),
            (r'(\d)S([A-Z])', r'\g<1>5\g<2>'),
            (r'\s4([a-zа-яà-öø-ÿ])', r' h\g<1>'),
            (r'(\d)\s*/\s*(\d)', r'\g<1>/\g<2>'),
            (r'(\d+)h', r'\g<1>4'),
            (r'\s*i0\s*', r' 10 '),
            (r'(\d{1,2}\s*)\'(\s*\d{1,2})', r'\g<1>-\g<2>'),
            (r'(?<=[a-zø-ÿışğçü])0(?=[a-zø-ÿışğçü])', 'o'),
            (r'(?<=[A-ZÀ-ÖİŞĞÇÜ])0(?=[A-ZÀ-ÖİŞĞÇÜ])', 'O'),
            (r'(?<=[a-zø-ÿışğçü])0', 'o'),
            (r'(?<=[A-ZÀ-ÖİŞĞÇÜ])0', 'O'),
            (r'(\d+)\s*\)', r'\g<1>)'),
            ]
    for pattern, repl in subs:
        text = re.sub(pattern, repl, text)
    text = re.sub(r'\(\s*(\d+)\s*([-–—])\s*(\d+)\s*\)', r'(\g<1>\g<2>\g<3>)', text)
    text = re.sub(r'\(\s*(\d+)\s*\)', r'(\g<1>)', text)
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

    max1 = max(int(k) for k in d1.keys())
    min2 = min(int(k) for k in d2.keys())

    if min2 == max1 + 1:
        merged = {**d1, **d2}
        return dict(sorted(merged.items(), key=lambda x: int(x[0])))
    else:
        return d2  # первый удаляется

def search_for_extract_ankara(text: str, pos_start: int, ind: int):
    # ------------------------------------------------
    global Pattern_search_translate
    global Pattern_search_trlit
    global Pattern_search_translate_re
    global Pattern_search_trlit_re
    global Pattern_search_translate_end
    global  Pattern_search_translate_end_re
    global patterns_akt_per_s
    global patterns_akt_trl_s
    global patterns_akt_per_e
    global number_text
    global prov_prev_transl
    text_translate = ""
    text_transliterate = ""
    text_transliterate_prev = ""
    text_translate_prev = ""
    pos_end_translate = 0
    pos_start_translate = None
    pos_end = len(text)
    pos_end_tr = 0
    pos_start_tr = 0
    flag_vyp = False
    transl_from_past = False
    trlit_from_past = False
    trlit_to_past = False
    end_pos = len(text)
    Pattern_search_trlit = patterns_akt_trl_s["start_trl"]
    Pattern_search_trlit_re = re.compile(Pattern_search_trlit, re.MULTILINE)
    match_trlit = Pattern_search_trlit_re.search(text, pos_start)
    if match_trlit:
        pos_start_trlit = match_trlit.start()
        text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, pos_start_trlit, len(text))
    # else:
    #     text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text[:5], pos_start, len(text))
    #     if text_transliterate != "":
    #         text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, pos_start, len(text))
    # --------------------------------------------------------------------------------------------------
        # проверка наличия перевода в начале текста для зарезервированной транслитерации
        if pos_start_trlit > 200 and pos_start < 75 and Unfin_Data['trlit'] != "" and prov_prev_transl != number_text:
            prov_prev_transl = number_text
            Pattern_search_translate = patterns_akt_per_s["plain"]
            Pattern_search_translate_re = re.compile(Pattern_search_translate)
            match_translate = Pattern_search_translate_re.search(text, pos_start)
            if match_translate:
                pos_start_translate = match_translate.start()
                if pos_start_translate < pos_start_trlit:
                    Pattern_search_translate_end = patterns_akt_per_e["plain"]
                    Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
                    match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
                    if match_translate_end:
                        pos_end_translate = match_translate_end.start()
                        text_translate = text[pos_start_translate:pos_end_translate]
                    else:
                        text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text,
                                                                                                        pos_start_translate,
                                                                                                        len(text))
                    if text_translate != "" and pos_end_translate <= pos_start_trlit:
                        text_translate_prev = text_translate
            if text_translate_prev != "" and Unfin_Data['trlit'] != "":
                text_transliterate_prev = Unfin_Data['trlit']
                flag_vyp = True
                Unfin_Data['trlit'] = ""
                return (text_translate_prev, text_transliterate_prev), flag_vyp, pos_end_trlit
    # ----------------------------------------------------------------------------------------------
        if pos_end_trlit == len(text) or pos_end_trlit == -1 or get_next_line(text, pos_end_trlit)[1] ==len(text):
            pos_end_trlit = len(text)
            end_pos = pos_end_trlit
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            Unfin_Data['trlit'] = text_transliterate
            trlit_to_past = True
    if Unfin_Data['trlit'] != "" and not trlit_to_past:
        if text_transliterate_prev == "":
            text_transliterate_prev = Unfin_Data['trlit']
            trlit_from_past = True
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
            Unfin_Data['trlit'] = ""
    if text_transliterate == "" or text_transliterate == {}:
        return (text_translate, text_transliterate), flag_vyp, end_pos

    Pattern_search_translate = patterns_akt_per_s["plain"]
    Pattern_search_translate_re = re.compile(Pattern_search_translate)
    match_translate = Pattern_search_translate_re.search(text, pos_end_trlit)
    if match_translate:
        pos_start_translate = match_translate.start()
        Pattern_search_translate_end = patterns_akt_per_e["plain"]
        Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        if match_translate_end:
            pos_end_translate = match_translate_end.start()
            text_translate = text[pos_start_translate:pos_end_translate]
        else:
            text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate,
                                                                                        len(text))
            if text_translate == "":
                pos_end_translate = len(text)
    else:
        pos_end_translate = len(text)
    end_pos = pos_end_translate

    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not is_clean_akkadian_translation(text_translate):
            text_translate = ""
            end_pos = len(text)
        # else:
        #     if pos_end_translate == len(text):
        #         Unfin_Data["perevod"] = text_translate
    if text_transliterate != "":
        if not trlit_from_past and not trlit_to_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            if text_transliterate == {}:
                end_pos = len(text)
    if text_translate != "" and text_transliterate != {}:
        flag_vyp =True
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, end_pos


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


# def translate_after_translite(text: str, start_pos: int = 0)-> tuple[bool, tuple[str, dict[int, str]], int]:
#     """Ищет позицию первого диапазона предложений в переводе
#     после транслитерации и возвращает транслитерацию, если найдёт её"""
#     flag_vyp, (perevod, dictionary_trlit), pos_end = search_for_extract_ankara(text)
#
#     return flag_vyp, (perevod, dictionary_trlit), pos_end


# def pos_first_translite(text: str, start_pos: int = 0):
#     """Ищет позицию начала транслитерации
#     и возвращает её"""
#     pos_first_trl = re.search(r'^.\.?\s*y\.\n', text, flags=re.MULTILINE)
#     return pos_first_trl.start() if pos_first_trl is not None else -1



# import re
# from typing import Dict, List, Tuple


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
#    #^\s*              # начало строки + пробелы

pattern_line_start = re.compile(
    r"""
    \(?                # необязательная открывающая скобка
    \s*               
    (?P<number>\d{1,3})  # число (1-3 цифры)
    \s*               
    (?:               # незахватывающая группа для окончания
        [`'’‘ʼʹˈ]   # любой апостроф/кавычка, 0 или 1
        \s*          # пробел
        \.           # точка
        |             # или
        [`'’‘ʼʹˈ]   # любой апостроф/кавычка, 0 или 1
        \s*          # пробел
        \)           # скобка
        |            # или
        [`'’‘ʼʹˈ]   # любой апостроф/кавычка, 0 или 1
        |             # или
        \.            # точка
        |             # или
        :             # двоеточие
        |             # или
        \)           # необязательная закрывающая скобка
    )
    """,
    re.VERBOSE
)

pattern_line_start_r = re.compile(
    r"""
    ^\s*                  # начало строки + пробелы
    \(?                    # необязательная открывающая скобка
    \s*                    
    (?P<number>\d{1,3})    # число (1-3 цифры)
    \s*                    
    (?:                    # незахватывающая группа для окончания
        \.                 # точка
        |                   # или
        [`'’‘ʼʹˈ]         # любой апостроф/кавычка, 0 или 1
        |                   # или
        [`'’‘ʼʹˈ]   # любой апостроф/кавычка, 0 или 1
        \s*          # пробел
        \)           # скобка
        |                 # или
        [`'’‘ʼʹˈ]         # любой апостроф/кавычка, 0 или 1
        \s*                 # пробелы после кавычки
        \.                 # точка
        |                   # или
        :                   # двоеточие
        |                   # или
        \)                  # закрывающая скобка
    )
    """,
    re.VERBOSE
)


def renumber_trust_source(text: str) -> dict:
    if not text.strip():
        return {}

    lines = text.splitlines()
    pattern = pattern_line_start_r  # твой универсальный шаблон

    found_numbers: List[int] = []

    # ищем все номера в начале строк
    for line in lines:
        m = pattern.match(line)
        if m:
            found_numbers.append(int(m.group("number")))

    result: dict = {}

    if found_numbers:
        # --- Все номера кратные 5
        if all(n % 5 == 0 for n in found_numbers):
            anchors: list[tuple[int, int]] = []
            for idx, line in enumerate(lines):
                m = pattern.match(line)
                if m:
                    num = int(m.group("number"))
                    anchors.append((idx, num))
            numbers = _restore_sequence(anchors, len(lines))  # твоя существующая логика
            for num, line in zip(numbers, lines):
                content = pattern.sub('', line, count=1).strip()
                result[num] = content

        # --- Есть обычные номера (не кратные 5)
        else:
            result = {}
            # собираем все позиции с номерами
            anchors = [(idx, int(pattern.match(line).group("number")))
                       for idx, line in enumerate(lines) if pattern.match(line)]

            if not anchors:
                # вообще нет номеров → обычная последовательность
                for i, line in enumerate(lines, start=1):
                    result[i] = line.strip()
            else:
                # восстанавливаем нумерацию для всех строк
                for idx, line in enumerate(lines):
                    # ищем ближайший «якорь» впереди
                    next_anchor = next(((a_idx, num) for a_idx, num in anchors if a_idx >= idx), None)
                    if next_anchor:
                        a_idx, num = next_anchor
                        key = num - (a_idx - idx)  # уменьшаем от ближайшего номера
                    else:
                        # если впереди нет якоря — продолжаем с последнего найденного
                        last_anchor_num = anchors[-1][1]
                        key = last_anchor_num + (idx - anchors[-1][0])

                    # удаляем номер из строки, если он есть
                    m = pattern.match(line)
                    content = pattern.sub('', line, count=1).strip() if m else line.strip()
                    result[key] = content
    # --- Номеров нет совсем
    else:
        for i, line in enumerate(lines, start=1):
            result[i] = line.strip()

    # проверка на один элемент
    if len(result) == 1:
        only_key = next(iter(result))
        if only_key != 1:
            result = {1: result[only_key]}

    return result

#
# # Регулярка для нумерации с разными вариантами кавычек и суффиксов
# pattern_numbers = re.compile(
#     r'\b\(?\d+\)?(?:[`\'’‘ʼʹˈ]\.|[.`\'’‘ʼʹˈ)])?\s+'
# )
#
# # Регулярка для нумерации с кавычками и суффиксами
# pattern_numbers_1 = re.compile(
#     r'\b(\(?\d+\)?(?:[`\'’‘ʼʹˈ]\.|[.`\'’‘ʼʹˈ)]?))\s+'
# )

def split_numbered_text_with_intro(text) -> dict:
    """Выводит словарь из текста с последовательно нумерованными участками.
       Шаблонные номера удаляются из строк."""

    matches = list(pattern_line_start.finditer(text))
    result = {}

    if matches:
        first_num = int(matches[0].group("number"))

        # текст перед первой нумерацией
        if matches[0].start() > 0:
            result[first_num - 1] = text[:matches[0].start()].strip()

        # основной цикл по найденной нумерации
        for i, m in enumerate(matches):
            key = int(m.group("number"))
            start = m.end()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            content = text[start:end].strip()
            # удаляем номер из строки
            content_clean = pattern_line_start.sub('', content, count=1).strip()
            result[key] = content_clean
    else:
        return {1: text.strip()}
    if len(result) == 1:
        # получаем единственный ключ
        only_key = next(iter(result))
        if only_key != 1:
            result = {1: result[only_key]}

    return result



def process_text_last(text: str, lines_dict: Dict[int, str]) -> Tuple[List[str], List[str]]:
    """разделение перевода по диапазонам и совмещение с транслитерацией"""

    def split_by_words(start, end, text):
        words = text.split()
        n = max(0, end - start + 1)
        if n == 0:
            return []
        k, m = divmod(len(words), n)
        return [
            " ".join(words[i * k + min(i, m):(i + 1) * k + min(i + 1, m)])
            for i in range(n)
        ]


    matches = []
    # форматирует диапазоны пробелами
    text = clear_from_ocr_for_text_last(text)
    # упорядочивает по порядку последовательности диапазонов
    text = clear_from_ocr_for_text(text)
    # шаблон диапазона
    # range_pattern = re.compile(Pattern_search_translate)
    range_pattern_str = r"(\(?\s*(?:(?P<start>\d+)\s*[-–—]\s*(?P<end>\d+)|(?P<number>\d+))\s*\'?\)?)"
    range_pattern_re = re.compile(range_pattern_str)
    # разделение текста по диапазонам
    if len(lines_dict) > 1:
        matches = list(range_pattern_re.finditer(text))

    dict_results: List[str] = []
    text_results: List[str] = []
    if not isinstance(lines_dict, dict):
        print(repr(lines_dict))
        raise TypeError(f"Ожидался dict, получен {type(lines_dict)}")

    if not matches:
        merged = " ".join(lines_dict.values())
        # удаление из строк словаря нумерации(это, вероятно, уже сделано ранее)
        # cleaned = re.sub(r'\d{1,2}[\.:]\s*', '', merged)
        # dict_results.append(cleaned)
        dict_results.append(merged)
        text = range_pattern_re.sub("", text).strip()
        text_results.append(text.replace("\n", " ").strip())
        return dict_results, text_results
    # перебор текста перевода по диапазонам
    for i, match in enumerate(matches):
        if match.group("start") and match.group("end"):
            start = int(match.group("start"))
            end = int(match.group("end")) if match.group("end") else start
            keys = range(start, end + 1)
            if len(matches) == 1:
                text = range_pattern_re.sub("", text).strip()
                keys_l = range(next(iter(lines_dict)), next(reversed(lines_dict))+1)
                text_l = split_by_words(next(iter(lines_dict)), next(reversed(lines_dict)), text)
                i_num = 0
                for k_l in keys_l:
                    # проверка: все ключи должны быть в lines_dict
                    if k_l in lines_dict:
                        dict_results.append("".join(lines_dict[k_l]))
                        # if i_num == 0:
                        #     text_l[i_num] = range_pattern_re.sub("", text[i_num]).strip()
                        text_results.append(text_l[i_num])
                        i_num += 1
            else:
                # сбор текста фрагмента без диапазонов
                text_start = match.end() + 1
                text_end = matches[i + 1].start() - 1 if i + 1 < len(matches) else len(text)
                fragment = range_pattern_re.sub("", text[text_start:text_end]).strip()
                # пропускаем пустые фрагменты
                if not fragment:
                    continue
                fragment_l = split_by_words(start, end + 1, fragment)
                i_num = 0
                for k in keys:
                    # проверка: все ключи должны быть в lines_dict
                    if k in lines_dict:
                        dict_results.append("".join(lines_dict[k]))
                        # только если фрагмент непустой, добавляем в результаты
                        # dict_results.append(" ".join(lines_dict[k] for k in keys))
                        text_results.append(fragment_l[i_num])
                        i_num += 1

    return dict_results, text_results



#%%
def extract_quoted_substring(text: str, start_pos: int, pattern: str, ind: int):
    """
    Ищет в строке text, начиная С позиции start_pos,
    подстроку вида: ' "текст"'.
    Возвращает:
        (substring, is_longer_than_30, closing_quote_pos)
    """
    if start_pos < 0 or start_pos >= len(text):
        return None, None, start_pos
    if start_pos == 0:
        # # 1. Основной шаблон якоря
        pattern = re.compile(pattern, re.MULTILINE)
        match = pattern.search(text, start_pos)
        if not match:
            return None, None, len(text)
        print(f"Найден поисковый якорь Assirian: {match.group()}")
        # text = text[match.end():]
        start_pos = match.end()

    # шаблон поиска перевода
    # Pattern_search_trlit, style, status_patterns_akt2_trl_slit = choose_pattern(text, patterns_akt2_trl_s)
    Pattern_search_trlit = patterns_individual_assirian["start_trl"]
    Pattern_search_trlit_re = re.compile(Pattern_search_trlit)
    match_trlit = Pattern_search_trlit_re.search(text, start_pos)
    if match_trlit:
        start_pos = match_trlit.end() - 2
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
    return None, None, len(text)

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
def extract_letter_space_digit_colon_space(text: str, start_search_pos: int, pattern: str, ind: int):
    """предполагается что переноса перевода с одного текста на следующий
    или транслитерации нет"""
    text_translate = ""
    text_translate_prev = ""
    flag_vyp = False
    trlit_from_past = False
    trlit_to_past = False
    transl_from_past = False
    text_transliterate = ""
    pos_end_trlit = 0
    pos_start_trlit = 0
    end_pos = len(text)
    global Pattern_search_translate, Pattern_search_trlit, Pattern_search_translate_re, Pattern_search_trlit_re, Pattern_search_translate_end, Pattern_search_translate_end_re
    global patterns_withaut_diapason_s, patterns_withaut_diapason_per_s, patterns_withaut_diapason_per_e
    global number_text, prov_prev_transl
    if start_search_pos == 0:
        pattern = re.compile(pattern, re.MULTILINE)
        match = pattern.search(text, 0)
        if not match:
            return (text_translate, text_transliterate), flag_vyp, end_pos
        print(f"Найден поисковый якорь Ankara: {match.group()}")
        start_search_pos = match.end()
    # Pattern_search_trlit, style, status_trlit = choose_pattern(text, patterns_withaut_diapason_s, True)
    Pattern_search_trlit = patterns_withaut_diapason_s["start_trl"]
    Pattern_search_trlit_re = re.compile(Pattern_search_trlit, re.MULTILINE)
    match_trlit = Pattern_search_trlit_re.search(text, start_search_pos)
    if match_trlit:
        pos_start_trlit = match_trlit.start()
        text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, pos_start_trlit, len(text))
    # else:
    #     text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text[:5], start_search_pos, len(text))
    #     if text_transliterate != "":
    #         text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, start_search_pos, len(text))
    # --------------------------------------------------------------------------------------------------
        # проверка наличия перевода в начале текста для зарезервированной транслитерации
        if pos_start_trlit > 200 and start_search_pos < 75 and Unfin_Data['trlit'] != "" and prov_prev_transl != number_text:
            prov_prev_transl = number_text
            Pattern_search_translate = patterns_withaut_diapason_per_s["start_per_quote"]
            Pattern_search_translate_re = re.compile(Pattern_search_translate)
            match_translate = Pattern_search_translate_re.search(text, start_search_pos)
            if match_translate:
                pos_start_translate = match_translate.start()
                if pos_start_translate < pos_start_trlit:
                    Pattern_search_translate_end = patterns_withaut_diapason_per_e["end_per_quote"]
                    Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
                    match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
                    if match_translate_end:
                        pos_end_translate = match_translate_end.start()
                        text_translate = text[pos_start_translate:pos_end_translate]
                    else:
                        text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text,
                                                                                                        pos_start_translate,
                                                                                                        len(text))
                    if text_translate != "" and pos_end_translate <= pos_start_trlit:
                        text_translate_prev = text_translate
            if text_translate_prev != "" and Unfin_Data['trlit'] != "":
                text_transliterate_prev = Unfin_Data['trlit']
                flag_vyp = True
                Unfin_Data['trlit'] = ""
                return (text_translate_prev, text_transliterate_prev), flag_vyp, pos_end_trlit
        # ----------------------------------------------------------------------------------------------
        if pos_end_trlit == len(text) or pos_end_trlit == -1 or get_next_line(text, pos_end_trlit)[1] ==len(text):
            pos_end_trlit = len(text)
            end_pos = pos_end_trlit
            # if not trlit_from_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            Unfin_Data['trlit'] = text_transliterate
            trlit_to_past = True
    if Unfin_Data['trlit'] != "" and not trlit_to_past:
        # if pos_start_trlit == pos_start:
        text_transliterate_prev = Unfin_Data['trlit']
        trlit_from_past = True
        if text_transliterate != "":
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
        else:
            text_transliterate = text_transliterate_prev
        Unfin_Data['trlit'] = ""
    if text_transliterate == "" or text_transliterate == {}:
        return (text_translate, text_transliterate), flag_vyp, end_pos

    Pattern_search_translate = patterns_withaut_diapason_per_s["start_per_quote"]
    Pattern_search_translate_re = re.compile(Pattern_search_translate, re.MULTILINE)
    match_translate = Pattern_search_translate_re.search(text, pos_end_trlit)
    if match_translate:
        # pos_end_trlit = match_translate.start()
        # text_transliterate = text[pos_start_trlit:pos_end_trlit]
        pos_start_translate = match_translate.start()
        # text_translate, pos_start_translate, pos_end_translate = find_translate_by_rows(text, pos_start_translate,
        #                                                                                 len(text))
        # Pattern_search_translate_end, style, status_trlat = choose_pattern(text, patterns_akt2_per_e)
        Pattern_search_translate_end = patterns_withaut_diapason_per_e["end_per_quote"]
        Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        if match_translate_end:
            pos_end_translate = match_translate_end.start()
            text_translate = text[pos_start_translate:pos_end_translate]
        else:
            #     pos_end_translate = len(text)
            text_translate, pos_start_translate, pos_end_translate = find_translate_by_rows(text, pos_start_translate, len(text))
            if text_translate == "":
                pos_end_translate = len(text)
            # text_translate = text[pos_start_translate:pos_end_translate]
    else:
        pos_end_translate = len(text)
        # if Unfin_Data["perevod"] != "":
        #     text_translate = Unfin_Data["perevod"]
        #     Unfin_Data["perevod"] = ""
        #     transl_from_past = True
    end_pos = pos_end_translate

    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not is_clean_akkadian_translation(text_translate):
            text_translate = ""
            end_pos = len(text)
        # else:
        #     if pos_end_translate == len(text) and not transl_from_past:
        #         Unfin_Data["perevod"] = text_translate
    if text_transliterate != "":
        if not trlit_from_past and not trlit_to_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            if text_transliterate == {}:
                end_pos = len(text)
    if text_translate != "" and text_transliterate != {}:
        flag_vyp = True
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, end_pos


def extract_after_letter_space_digit_colon_space(text_dict_tr: tuple[str, dict[int, str]], pos_s: int) -> Tuple[Tuple[List:str, List:str], bool, int]:
    flag_vyp = False
    list_trl_transl = None
    if all(text_dict_tr):
        list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
        if all(list_trl_transl):
            flag_vyp = True
    # # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, flag_vyp, pos_s
        # кортеж списков перевода и транслитерации, флаг, конец перевода
    # return list_trl_transl, flag_vyp, pos_s


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

def extract_ankara(text: str, start_pos: int, pattern: str, ind: int):
    if start_pos < 0 or start_pos >= len(text):
        return ("", ""), False, start_pos
    if start_pos == 0:
        pattern = re.compile(pattern, re.MULTILINE)
        match = pattern.search(text, start_pos)
        if not match:
            return ("", ""), False, len(text)
        print(f"Найден поисковый якорь Ankara: {match.group()}")
    # text = text[match.end():]
        start_pos = match.end()
    # start_pos = match.end() if start_pos <= match.end() else start_pos
    (perevod, dict_trlit), flag_vyp, pos_end = search_for_extract_ankara(text, start_pos, ind)
    return (perevod, dict_trlit), flag_vyp, pos_end

def extract_after_ankara(text_dict_tr: tuple[str, dict[int, str]], pos_s: int) -> Tuple[Tuple[List:str, List:str], bool, int]:
    flag_vyp = False
    list_trl_transl = None
    if all(text_dict_tr):
        list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
        if all(list_trl_transl):
            flag_vyp = True
    # # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, flag_vyp, pos_s

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



def extract_ankara_next(text: str, start_pos: int, pattern: str, ind: int):
    text_translate = ""
    text_transliterate = ""
    text_translate_prev = ""
    flag_vyp = False
    transl_from_past = False
    trlit_from_past = False
    trlit_to_past = False
    pos_end_trlit = 0
    global Pattern_search_trlit, Pattern_search_trlit_re, Pattern_search_translate, Pattern_search_translate_re, Pattern_search_translate_end, Pattern_search_translate_end_re
    global patterns_akt2_trl_s, patterns_akt2_per_s, patterns_akt2_per_e, number_text, prov_prev_transl
    if start_pos < 0 or start_pos >= len(text):
        return ("", ""), flag_vyp, start_pos
    end_pos = len(text)
    if start_pos == 0:
        pattern = re.compile(pattern)
        match = pattern.search(text, start_pos)
        if not match:
            return (text_translate, ""), flag_vyp, end_pos
        print(f"Найден поисковый якорь Ankara: {match.group()}")
        start_pos = match.end()

    # text = text[match.end():]
    # start_pos += match.end()
    # start_pos = match.end() if start_pos <= match.end() else start_pos
    # шаблон поиска перевода
    # Pattern_search_trlit, style, status_patterns_akt2_trl_slit = choose_pattern(text, patterns_akt2_trl_s)
    Pattern_search_trlit = patterns_akt2_trl_s["start_trl_tablet_or_paren"]
    Pattern_search_trlit_re = re.compile(Pattern_search_trlit, re.MULTILINE)
    match_trlit = Pattern_search_trlit_re.search(text, start_pos)
    if match_trlit:
        pos_start_trlit = match_trlit.start()
        text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, pos_start_trlit, len(text))
    # else:
    #     text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text[:5], start_pos,
    #                                                                                len(text))
    #     if text_transliterate != "":
    #         text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, start_pos,
    #                                                                                    len(text))
        # --------------------------------------------------------------------------------------------------
        # проверка наличия перевода в начале текста для зарезервированной транслитерации
        if pos_start_trlit > 200 and start_pos < 75 and Unfin_Data['trlit'] != "" and prov_prev_transl != number_text:
            prov_prev_transl = number_text
            Pattern_search_translate = patterns_akt2_per_s["paren_both"]
            Pattern_search_translate_re = re.compile(Pattern_search_translate)
            match_translate = Pattern_search_translate_re.search(text, start_pos)
            if match_translate:
                pos_start_translate = match_translate.start()
                if pos_start_translate < pos_start_trlit:
                    Pattern_search_translate_end = patterns_akt2_per_e["zarf"]
                    Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
                    match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
                    if match_translate_end:
                        pos_end_translate = match_translate_end.start()
                        text_translate = text[pos_start_translate:pos_end_translate]
                    else:
                        text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text,
                                                                                                        pos_start_translate,
                                                                                                        len(text))
                    if text_translate != "" and pos_end_translate <= pos_start_trlit:
                        text_translate_prev = text_translate
            if text_translate_prev != "" and Unfin_Data['trlit'] != "":
                text_transliterate_prev = Unfin_Data['trlit']
                flag_vyp = True
                Unfin_Data['trlit'] = ""
                return (text_translate_prev, text_transliterate_prev), flag_vyp, pos_end_trlit
        # ----------------------------------------------------------------------------------------------
        if pos_end_trlit == len(text) or pos_end_trlit == -1 or get_next_line(text, pos_end_trlit)[1] ==len(text):
            pos_end_trlit = len(text)
            end_pos = pos_end_trlit
            # if not trlit_from_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            # text_transliterate = renumber_trust_source(text_transliterate)
            text_transliterate = parse_numbered_fragments(text_transliterate)
            Unfin_Data['trlit'] = text_transliterate
            trlit_to_past = True
    if Unfin_Data['trlit'] != "" and not trlit_to_past:
        # if pos_start_trlit == pos_start:
        text_transliterate_prev = Unfin_Data['trlit']
        trlit_from_past = True
        if text_transliterate != "":
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            # text_transliterate = renumber_trust_source(text_transliterate)
            text_transliterate = parse_numbered_fragments(text_transliterate)
            text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
        else:
            text_transliterate = text_transliterate_prev
        Unfin_Data['trlit'] = ""
    if text_transliterate == "" or text_transliterate == {}:
        return (text_translate, text_transliterate), flag_vyp, end_pos

    # text_transliterate = extract_transliteration(text[pos_start_trlit:])
    # Pattern_search_translate, style, status_trlat = choose_pattern(text, patterns_akt2_per_s)
    Pattern_search_translate = patterns_akt2_per_s["paren_both"]
    Pattern_search_translate_re = re.compile(Pattern_search_translate, re.MULTILINE)
    match_translate = Pattern_search_translate_re.search(text, pos_end_trlit)
    if match_translate:
        # pos_end_trlit = match_translate.start()
        # text_transliterate = text[pos_start_trlit:pos_end_trlit]
        pos_start_translate = match_translate.start()
        # text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate, len(text))
        # Pattern_search_translate_end, style, status_trlat = choose_pattern(text, patterns_akt2_per_e)
        Pattern_search_translate_end = patterns_akt2_per_e["zarf"]
        Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        if match_translate_end:
            pos_end_translate = match_translate_end.start()
            text_translate = text[pos_start_translate:pos_end_translate]
        else:
            #     pos_end_translate = len(text)
            # text_translate = text[pos_start_translate:pos_end_translate]
            text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate,
                                                                                        len(text))
            if text_translate == "":
                pos_end_translate = len(text)
    else:
        pos_end_translate = len(text)
        # if Unfin_Data["perevod"] != "":
        #     text_translate = Unfin_Data["perevod"]
        #     Unfin_Data["perevod"] = ""
        #     transl_from_past = True
    end_pos = pos_end_translate

    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not is_clean_akkadian_translation(text_translate):
            text_translate = ""
            end_pos = len(text)
        # else:
        #     if pos_end_translate == len(text):
        #         Unfin_Data["perevod"] = text_translate
    if text_transliterate != "":
        if not trlit_from_past and not trlit_to_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            # text_transliterate = renumber_trust_source(text_transliterate)
            text_transliterate = parse_numbered_fragments(text_transliterate)
            if text_transliterate == {}:
                end_pos = len(text)
    if text_translate != "" and text_transliterate != {}:
        flag_vyp =True
    # if text_translate != "" and transl_from_past and text_transliterate == "":
    #     flag_vyp = False
    #     text_translate = ""
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, end_pos

def extract_after_ankara_next(text_dict_tr: tuple, pos_s: int):
    flag_vyp = False
    list_trl_transl = None
    if all(text_dict_tr):
        list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
        if all(list_trl_transl):
            flag_vyp = True
    # # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, flag_vyp, pos_s

def extract_numbs_and_diapasons(text: str, start_pos: int, pattern: str, ind: int):
    text_translate = ""
    text_transliterate = ""
    text_translate_prev = ""
    pos_end_trlit = 0
    flag_vyp = False
    transl_from_past = False
    trlit_from_past = False
    trlit_to_past = False
    global Pattern_search_trlit, Pattern_search_trlit_re, Pattern_search_translate, Pattern_search_translate_re, Pattern_search_translate_end, Pattern_search_translate_end_re
    global patterns_numbs_and_diapasones_s, patterns_numbs_and_diapasones_paren_s, patterns_numbs_and_diapasones_per_s, patterns_numbs_and_diapasones_paren_per_s, patterns_numbs_and_diapasones_per_e, patterns_numbs_and_diapasones_paren_per_e
    global number_text, prov_prev_transl
    if start_pos < 0 or start_pos >= len(text):
        return (text_translate, ""), flag_vyp, start_pos
    end_pos = len(text)
    if start_pos == 0:
        pattern = re.compile(pattern)
        match = pattern.search(text, start_pos)
        if not match:
            return (text_translate, ""), flag_vyp, end_pos
        print(f"Найден поисковый якорь: {match.group()}")
        # text = text[match.end():]
        start_pos = match.end()
    match ind:
        case 4:
            Pattern_search_trlit = patterns_numbs_and_diapasones_s["start_numb_trl"]
        case 5:
            Pattern_search_trlit = patterns_numbs_and_diapasones_paren_s["start_numb_trl"]
    Pattern_search_trlit_re = re.compile(Pattern_search_trlit, re.MULTILINE)
    match_trlit = Pattern_search_trlit_re.search(text, start_pos)
    if match_trlit:
        if ind == 4:
            if str(match_trlit.group(0))[:3] == "Vs." or str(match_trlit.group(0))[2:4] == "y.":
                pos_start_trlit = match_trlit.end()
            elif str(match_trlit.group(0))== "F.":
                pos_start_trlit = match_trlit.start()
                end = match_trlit.end()
                # заменяем в тексте
                text = text[:pos_start_trlit] + "  " + text[end:]
            else:
                pos_start_trlit = match_trlit.start()
        else:
            pos_start_trlit = match_trlit.start()
        text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, pos_start_trlit, len(text))
    # else:
    #     text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text[:5], start_pos, len(text))
    #     if text_transliterate != "":
    #         text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, start_pos, len(text))
    # --------------------------------------------------------------------------------------------------
        # проверка наличия перевода в начале текста для зарезервированной транслитерации
        if pos_start_trlit > 200 and start_pos < 75 and Unfin_Data['trlit'] != "" and prov_prev_transl != number_text:
            prov_prev_transl = number_text
            match ind:
                case 4:
                    Pattern_search_translate = patterns_numbs_and_diapasones_per_s["start_numb_per"]
                case 5:
                    Pattern_search_translate = patterns_numbs_and_diapasones_paren_per_s["start_numb_per"]
            Pattern_search_translate_re = re.compile(Pattern_search_translate)
            match_translate = Pattern_search_translate_re.search(text, start_pos)
            if match_translate:
                pos_start_translate = match_translate.start()
                if pos_start_translate < pos_start_trlit:
                    match ind:
                        case 4:
                            Pattern_search_translate_end = patterns_numbs_and_diapasones_per_e["end_numb_per"]
                        case 5:
                            Pattern_search_translate_end = patterns_numbs_and_diapasones_paren_per_e["end_numb_per"]
                    Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
                    match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
                    if match_translate_end:
                        pos_end_translate = match_translate_end.start()
                        text_translate = text[pos_start_translate:pos_end_translate]
                    else:
                        text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text,
                                                                                                        pos_start_translate,
                                                                                                        len(text))
                    if text_translate != "" and pos_end_translate <= pos_start_trlit:
                        text_translate_prev = text_translate
            if text_translate_prev != "" and Unfin_Data['trlit'] != "":
                text_transliterate_prev = Unfin_Data['trlit']
                flag_vyp = True
                Unfin_Data['trlit'] = ""
                return (text_translate_prev, text_transliterate_prev), flag_vyp, pos_end_trlit
        # ----------------------------------------------------------------------------------------------

        if pos_end_trlit == len(text) or pos_end_trlit == -1 or get_next_line(text, pos_end_trlit)[1] ==len(text):
            pos_end_trlit = len(text)
            end_pos = pos_end_trlit
            # if not trlit_from_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            match ind:
                case 4:
                    text_transliterate = renumber_trust_source(text_transliterate)
                case 5:
                    text_transliterate = parse_numbered_fragments(text_transliterate)
            Unfin_Data['trlit'] = text_transliterate
            text_transliterate = ""
            trlit_to_past = True
    if Unfin_Data['trlit'] != "" and not trlit_to_past:
        # if pos_start_trlit == pos_start:
        text_transliterate_prev = Unfin_Data['trlit']
        trlit_from_past = True
        if text_transliterate != "":
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            match ind:
                case 4:
                    text_transliterate = renumber_trust_source(text_transliterate)
                case 5:
                    text_transliterate = parse_numbered_fragments(text_transliterate)
            text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
        else:
            text_transliterate = text_transliterate_prev
        Unfin_Data['trlit'] = ""

    if text_transliterate == "" or text_transliterate == {}:
        return (text_translate, text_transliterate), flag_vyp, end_pos

    match ind:
        case 4:
            Pattern_search_translate = patterns_numbs_and_diapasones_per_s["start_numb_per"]
        case 5:
            Pattern_search_translate = patterns_numbs_and_diapasones_paren_per_s["start_numb_per"]
    Pattern_search_translate_re = re.compile(Pattern_search_translate, re.MULTILINE)
    match_translate = Pattern_search_translate_re.search(text, pos_end_trlit)
    if match_translate:
        # pos_end_trlit = match_translate.start()
        # text_transliterate = text[pos_start_trlit:pos_end_trlit]
        pos_start_translate = match_translate.start()
        # text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate,
        #                                                                                 len(text))
        # Pattern_search_translate_end, style, status_trlat = choose_pattern(text, patterns_akt2_per_e)
        match ind:
            case 4:
                Pattern_search_translate_end = patterns_numbs_and_diapasones_per_e["end_numb_per"]
            case 5:
                Pattern_search_translate_end = patterns_numbs_and_diapasones_paren_per_e["end_numb_per"]
        Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        if match_translate_end:
            pos_end_translate = match_translate_end.start()
            text_translate = text[pos_start_translate:pos_end_translate]
        else:
            #     pos_end_translate = len(text)

            text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate,
                                                                                            len(text))
            if text_translate == "":
                pos_end_translate = len(text)
    else:
        pos_end_translate = len(text)
        # if Unfin_Data["perevod"] != "":
        #     text_translate = Unfin_Data["perevod"]
        #     Unfin_Data["perevod"] = ""
        #     transl_from_past = True
    end_pos = pos_end_translate

    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not is_clean_akkadian_translation(text_translate):
            text_translate = ""
            end_pos = len(text)
        # else:
        #     if pos_end_translate == len(text):
        #         Unfin_Data["perevod"] = text_translate
    if text_transliterate != "":
        if not trlit_from_past and not trlit_to_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            if text_transliterate == {}:
                end_pos = len(text)
    if text_translate != "" and text_transliterate != {}:
        flag_vyp = True
    # if text_translate != "" and transl_from_past and text_transliterate == "":
    #     flag_vyp = False
    #     text_translate = ""
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, end_pos

def after_extract_numbs_and_diapasones(text_dict_tr: tuple, pos_s: int):
    flag_vyp = False
    list_trl_transl = None
    if all(text_dict_tr):
        list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
        if all(list_trl_transl):
            flag_vyp = True
    # # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, flag_vyp, pos_s


def extract_salim_assur(text: str, start_pos: int, pattern: str, ind: int):
    text_translate = ""
    text_transliterate = ""
    pos_start_trlit = 0
    pos_end_trlit = 0
    flag_vyp = False
    transl_from_past = False
    trlit_from_past = False
    trlit_to_past = False
    text_translate_prev = ""
    global Pattern_search_trlit, Pattern_search_trlit_re, Pattern_search_translate, Pattern_search_translate_re, Pattern_search_translate_end, Pattern_search_translate_end_re
    global patterns_salim_assur_s, patterns_salim_assur_per_s, patterns_salim_assur_per_e, number_text, prov_prev_transl
    if start_pos < 0 or start_pos >= len(text):
        return (text_translate, ""), flag_vyp, start_pos
    end_pos = len(text)
    if start_pos == 0:
        pattern = re.compile(pattern)
        match = pattern.search(text, start_pos)
        if not match:
            return (text_translate, ""), flag_vyp, end_pos
        print(f"Найден поисковый якорь: {match.group()}")
        start_pos = text.find("\n", match.end())
        # text = text[match.end():]
        # start_pos = match.end()
    Pattern_search_trlit = patterns_salim_assur_s["salim_start_trl"]
    Pattern_search_trlit_re = re.compile(Pattern_search_trlit, re.MULTILINE)
    match_trlit = Pattern_search_trlit_re.search(text, start_pos)
    if match_trlit:
        pos_start_trlit = match_trlit.end()
        text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, pos_start_trlit, len(text))
    else:
        text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text[:5], start_pos, len(text))
        if text_transliterate != "":
            text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, start_pos, len(text))

        # --------------------------------------------------------------------------------------------------
        # проверка наличия перевода в начале текста для зарезервированной транслитерации
        if pos_start_trlit > 200 and start_pos < 75 and Unfin_Data['trlit'] != "" and prov_prev_transl != number_text:
            prov_prev_transl = number_text
            Pattern_search_translate = patterns_salim_assur_per_s["salim_start_per"]
            Pattern_search_translate_re = re.compile(Pattern_search_translate)
            match_translate = Pattern_search_translate_re.search(text, start_pos)
            if match_translate:
                pos_start_translate = match_translate.start()
                if pos_start_translate < pos_start_trlit:
                    Pattern_search_translate_end = patterns_salim_assur_per_e["salim_end_per"]
                    Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
                    match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
                    if match_translate_end:
                        pos_end_translate = match_translate_end.start()
                        text_translate = text[pos_start_translate:pos_end_translate]
                    else:
                        text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text,
                                                                                                        pos_start_translate,
                                                                                                        len(text))
                    if text_translate != "" and pos_end_translate <= pos_start_trlit:
                        text_translate_prev = text_translate
            if text_translate_prev != "" and Unfin_Data['trlit'] != "":
                text_transliterate_prev = Unfin_Data['trlit']
                flag_vyp = True
                Unfin_Data['trlit'] = ""
                return (text_translate_prev, text_transliterate_prev), flag_vyp, pos_end_trlit
        # ----------------------------------------------------------------------------------------------

        if pos_end_trlit == len(text) or pos_end_trlit == -1 or get_next_line(text, pos_end_trlit)[1] ==len(text):
            pos_end_trlit = len(text)
            end_pos = pos_end_trlit
            # if not trlit_from_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            Unfin_Data['trlit'] = text_transliterate
            trlit_to_past = True
    if Unfin_Data['trlit'] != "" and not trlit_to_past:
        # if pos_start_trlit == pos_start:
        text_transliterate_prev = Unfin_Data['trlit']
        trlit_from_past = True
        if text_transliterate != "":
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
        else:
            text_transliterate = text_transliterate_prev
        Unfin_Data['trlit'] = ""
    if text_transliterate == "" or text_transliterate == {}:
        return (text_translate, text_transliterate), flag_vyp, end_pos

    Pattern_search_translate = patterns_salim_assur_per_s["salim_start_per"]
    Pattern_search_translate_re = re.compile(Pattern_search_translate, re.MULTILINE)
    match_translate = Pattern_search_translate_re.search(text, pos_end_trlit)
    if match_translate:
        # pos_end_trlit = match_translate.start()
        # text_transliterate = text[pos_start_trlit:pos_end_trlit]
        pos_start_translate = match_translate.end()
        # text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate, len(text))
        # Pattern_search_translate_end, style, status_trlat = choose_pattern(text, patterns_akt2_per_e)
        Pattern_search_translate_end = patterns_salim_assur_per_e["salim_end_per"]
        Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        if match_translate_end:
            pos_end_translate = match_translate_end.start()
            text_translate = text[pos_start_translate:pos_end_translate]
        else:
            #     pos_end_translate = len(text)
            text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate,
                                                                                       len(text))
            if text_translate == "":
                pos_end_translate = len(text)
        # text_translate = text[pos_start_translate:pos_end_translate]
    else:
        text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_end_trlit,
                                                                                        len(text))
        if text_translate == "":
            pos_end_translate = len(text)
        # pos_end_translate = len(text)
        # if Unfin_Data["perevod"] != "":
        #     text_translate = Unfin_Data["perevod"]
        #     Unfin_Data["perevod"] = ""
        #     transl_from_past = True
    end_pos = pos_end_translate

    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not is_clean_akkadian_translation(text_translate):
            text_translate = ""
            end_pos = len(text)
        # else:
        #     if pos_end_translate == len(text):
        #         Unfin_Data["perevod"] = text_translate
    if text_transliterate != "":
        if not trlit_from_past and not trlit_to_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            if text_transliterate == {}:
                end_pos = len(text)
    if text_translate != "" and text_transliterate != {}:
        flag_vyp = True
    # if text_translate != "" and transl_from_past and text_transliterate == "":
    #     flag_vyp = False
    #     text_translate = ""
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, end_pos

def after_extract_salim_assur(text_dict_tr: tuple, pos_s: int):
    flag_vyp = False
    list_trl_transl = None
    if all(text_dict_tr):
        list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
        if all(list_trl_transl):
            flag_vyp = True
    # # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, flag_vyp, pos_s

def extract_sebahat(text: str, start_pos: int, pattern: str, ind: int):
    text_translate = ""
    flag_vyp = False
    transl_from_past = False
    trlit_from_past = False
    trlit_to_past = False
    text_translate_prev = ""
    global Pattern_search_trlit, Pattern_search_trlit_re, Pattern_search_translate, Pattern_search_translate_re, Pattern_search_translate_end, Pattern_search_translate_end_re
    global patterns_sebahattin_s, patterns_sebahattin_per_s, patterns_sebahattin_per_e, number_text, prov_prev_transl
    if start_pos < 0 or start_pos >= len(text):
        return (text_translate, ""), flag_vyp, start_pos
    end_pos = len(text)
    if start_pos == 0:
        pattern = re.compile(pattern)
        match = pattern.search(text, start_pos)
        if not match:
            return (text_translate, ""), flag_vyp, end_pos
        print(f"Найден поисковый якорь: {match.group()}")
        start_pos = text.find("\n", match.end())
        # start_pos = get_next_line(text, start_pos)[1]
        # text = text[match.end():]
        # start_pos = match.end()
    text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, start_pos, len(text))
    if text_transliterate != "":
        # --------------------------------------------------------------------------------------------------
        # проверка наличия перевода в начале текста для зарезервированной транслитерации
        if pos_start_trlit > 200 and start_pos < 75 and Unfin_Data['trlit'] != "" and prov_prev_transl != number_text:
            prov_prev_transl = number_text
            Pattern_search_translate = patterns_sebahattin_per_s["sebahat_start_per"]
            Pattern_search_translate_re = re.compile(Pattern_search_translate)
            match_translate = Pattern_search_translate_re.search(text, start_pos)
            if match_translate:
                pos_start_translate = match_translate.start()
                if pos_start_translate < pos_start_trlit:
                    Pattern_search_translate_end = patterns_sebahattin_per_e["sebahat_end_per"]
                    Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
                    match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
                    if match_translate_end:
                        pos_end_translate = match_translate_end.start()
                        text_translate = text[pos_start_translate:pos_end_translate]
                    else:
                        text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text,
                                                                                                        pos_start_translate,
                                                                                                        len(text))
                    if text_translate != "" and pos_end_translate <= pos_start_trlit:
                        text_translate_prev = text_translate
            if text_translate_prev != "" and Unfin_Data['trlit'] != "":
                text_transliterate_prev = Unfin_Data['trlit']
                flag_vyp = True
                Unfin_Data['trlit'] = ""
                return (text_translate_prev, text_transliterate_prev), flag_vyp, pos_end_trlit
        # ----------------------------------------------------------------------------------------------
        if pos_end_trlit == len(text) or pos_end_trlit == -1 or get_next_line(text, pos_end_trlit)[1] ==len(text):
            pos_end_trlit = len(text)
            end_pos = pos_end_trlit
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            Unfin_Data['trlit'] = text_transliterate
            trlit_to_past = True
    if Unfin_Data['trlit'] != "" and not trlit_to_past:
        text_transliterate_prev = Unfin_Data['trlit']
        trlit_from_past = True
        if text_transliterate != "":
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
        else:
            text_transliterate = text_transliterate_prev
        Unfin_Data['trlit'] = ""
    if text_transliterate == "" or text_transliterate == {}:
        return (text_translate, text_transliterate), flag_vyp, end_pos
    Pattern_search_translate = patterns_sebahattin_per_s["sebahat_start_per"]
    Pattern_search_translate_re = re.compile(Pattern_search_translate, re.MULTILINE)
    match_translate = Pattern_search_translate_re.search(text, pos_end_trlit)
    if match_translate:
        pos_start_translate = match_translate.start()
        Pattern_search_translate_end = patterns_sebahattin_per_e["sebahat_end_per"]
        Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        if match_translate_end:
            pos_end_translate = match_translate_end.start()
            text_translate = text[pos_start_translate:pos_end_translate]
        else:
            text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate,
                                                                                            len(text))
            if text_translate == "":
                pos_end_translate = len(text)
    else:
        pos_end_translate = len(text)
    end_pos = pos_end_translate

    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not is_clean_akkadian_translation(text_translate):
            text_translate = ""
            end_pos = len(text)
    if text_transliterate != "":
        if not trlit_from_past and not trlit_to_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            if text_transliterate == {}:
                end_pos = len(text)
    if text_translate != "" and text_transliterate != {}:
        flag_vyp = True
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, end_pos


def after_extract_sebahat(text_dict_tr: tuple, pos_s: int):
    flag_vyp = False
    list_trl_transl = None
    if all(text_dict_tr):
        list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
        if all(list_trl_transl):
            flag_vyp = True
    # # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, flag_vyp, pos_s


def extract_tabletVII(text: str, start_pos: int, pattern: str, ind: int):
    text_translate = ""
    flag_vyp = False
    transl_from_past = False
    trlit_from_past = False
    trlit_to_past = False
    text_translate_prev = ""
    global Pattern_search_trlit, Pattern_search_trlit_re, Pattern_search_translate, Pattern_search_translate_re, Pattern_search_translate_end, Pattern_search_translate_end_re
    global patterns_kultepe_VII_s, patterns_kultepe_VII_per_s, patterns_kultepe_VII_per_e, number_text, prov_prev_transl, patterns_byram_s, patterns_byram_per_s, patterns_byram_per_e
    if start_pos < 0 or start_pos >= len(text):
        return (text_translate, ""), flag_vyp, start_pos
    end_pos = len(text)
    if start_pos == 0:
        pattern = re.compile(pattern)
        match = pattern.search(text, start_pos)
        if not match:
            return (text_translate, ""), flag_vyp, end_pos
        print(f"Найден поисковый якорь: {match.group()}")
        start_pos = text.find("\n", match.end())
    text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, start_pos, len(text))
    if text_transliterate != "":
        # --------------------------------------------------------------------------------------------------
        # проверка наличия перевода в начале текста для зарезервированной транслитерации
        if pos_start_trlit > 200 and start_pos < 75 and Unfin_Data['trlit'] != "" and prov_prev_transl != number_text:
            prov_prev_transl = number_text
            match ind:
                case 8:
                    Pattern_search_translate = patterns_kultepe_VII_per_s["VII_start_per"]
                case 9:
                    Pattern_search_translate = patterns_byram_per_s["byram_start_per"]
            Pattern_search_translate_re = re.compile(Pattern_search_translate)
            match_translate = Pattern_search_translate_re.search(text, start_pos)

            if match_translate:
                pos_start_translate = match_translate.start()
                if pos_start_translate < pos_start_trlit:
                    match ind:
                        case 8:
                            Pattern_search_translate_end = patterns_kultepe_VII_per_e["VII_end_per"]
                        case 9:
                            Pattern_search_translate_end = patterns_byram_per_e["byram_end_per"]
                    Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
                    match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
                    if match_translate_end:
                        pos_end_translate = match_translate_end.start()
                        text_translate = text[pos_start_translate:pos_end_translate]
                    else:
                        text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text,
                                                                                                        pos_start_translate,
                                                                                                        len(text))
                    if text_translate != "" and pos_end_translate <= pos_start_trlit:
                        text_translate_prev = text_translate
            if text_translate_prev != "" and Unfin_Data['trlit'] != "":
                text_transliterate_prev = Unfin_Data['trlit']
                flag_vyp = True
                Unfin_Data['trlit'] = ""
                return (text_translate_prev, text_transliterate_prev), flag_vyp, pos_end_trlit
        # ----------------------------------------------------------------------------------------------
        if pos_end_trlit == len(text) or pos_end_trlit == -1 or get_next_line(text, pos_end_trlit)[1] ==len(text):
            pos_end_trlit = len(text)
            end_pos = pos_end_trlit
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            Unfin_Data['trlit'] = text_transliterate
            trlit_to_past = True
    if Unfin_Data['trlit'] != "" and not trlit_to_past:
        text_transliterate_prev = Unfin_Data['trlit']
        trlit_from_past = True
        if text_transliterate != "":
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
        else:
            text_transliterate = text_transliterate_prev
        Unfin_Data['trlit'] = ""
    if text_transliterate == "" or text_transliterate == {}:
        return (text_translate, text_transliterate), flag_vyp, end_pos
    match ind:
        case 8:
            Pattern_search_translate = patterns_kultepe_VII_per_s["VII_start_per"]
        case 9:
            Pattern_search_translate = patterns_byram_per_s["byram_start_per"]
    Pattern_search_translate_re = re.compile(Pattern_search_translate, re.MULTILINE)
    match_translate = Pattern_search_translate_re.search(text, pos_end_trlit)
    if match_translate:
        pos_start_translate = match_translate.start()
        match ind:
            case 8:
                Pattern_search_translate_end = patterns_kultepe_VII_per_e["VII_end_per"]
            case 9:
                Pattern_search_translate_end = patterns_byram_per_e["byram_end_per"]
        Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        if match_translate_end:
            pos_end_translate = match_translate_end.start()
            text_translate = text[pos_start_translate:pos_end_translate]
        else:
            text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate,
                                                                                            len(text))
            if text_translate == "":
                pos_end_translate = len(text)
    else:
        pos_end_translate = len(text)
    end_pos = pos_end_translate

    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not is_clean_akkadian_translation(text_translate):
            text_translate = ""
            end_pos = len(text)
    if text_transliterate != "":
        if not trlit_from_past and not trlit_to_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            if text_transliterate == {}:
                end_pos = len(text)
    if text_translate != "" and text_transliterate != {}:
        flag_vyp = True
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, end_pos


def after_extract_tabletVII(text_dict_tr: tuple, pos_s: int):
    flag_vyp = False
    list_trl_transl = None
    if all(text_dict_tr):
        list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
        if all(list_trl_transl):
            flag_vyp = True
    # # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, flag_vyp, pos_s


def extract_babylon(text: str, start_pos: int, pattern: str, ind: int):
    text_translate = ""
    text_transliterate = ""
    pos_start_trlit = 0
    pos_end_trlit = 0
    flag_vyp = False
    transl_from_past = False
    trlit_from_past = False
    trlit_to_past = False
    text_translate_prev = ""
    global Pattern_search_trlit, Pattern_search_trlit_re, Pattern_search_translate, Pattern_search_translate_re, Pattern_search_translate_end, Pattern_search_translate_end_re
    global patterns_babylon_s, patterns_babylon_per_s, patterns_babylon_per_e, number_text, prov_prev_transl
    if start_pos < 0 or start_pos >= len(text):
        return (text_translate, ""), flag_vyp, start_pos
    end_pos = len(text)
    if start_pos == 0:
        pattern = re.compile(pattern)
        match = pattern.search(text, start_pos)
        if not match:
            return (text_translate, ""), flag_vyp, end_pos
        print(f"Найден поисковый якорь: {match.group()}")
        start_pos = text.find("\n", match.end())
        # text = text[match.end():]
        # start_pos = match.end()
    Pattern_search_trlit = patterns_babylon_s["babylon_start_trl"]
    Pattern_search_trlit_re = re.compile(Pattern_search_trlit, re.MULTILINE)
    match_trlit = Pattern_search_trlit_re.search(text, start_pos)
    if match_trlit:
        val_end = match_trlit.group("start_per")
        patterns_babylon_per_s = rf'^{re.escape(val_end)}'
        pos_start_trlit = text.find("\n", match_trlit.end())
        # pos_start_trlit = match_trlit.end()
        match_trlit_end = Pattern_search_trlit_re.search(text, pos_start_trlit)
        if match_trlit_end:
            pos_end_trlit = match_trlit_end.start()
            text_transliterate = text[pos_start_trlit:pos_end_trlit]
        else:
            text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, pos_start_trlit, len(text))
        # --------------------------------------------------------------------------------------------------
        # # проверка наличия перевода в начале текста для зарезервированной транслитерации
        # if pos_start_trlit > 200 and start_pos < 75 and Unfin_Data['trlit'] != "" and prov_prev_transl != number_text:
        #     prov_prev_transl = number_text
        #     Pattern_search_translate = patterns_salim_assur_per_s["salim_start_per"]
        #     Pattern_search_translate_re = re.compile(Pattern_search_translate)
        #     match_translate = Pattern_search_translate_re.search(text, start_pos)
        #     if match_translate:
        #         pos_start_translate = match_translate.start()
        #         if pos_start_translate < pos_start_trlit:
        #             Pattern_search_translate_end = patterns_salim_assur_per_e["salim_end_per"]
        #             Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        #             match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        #             if match_translate_end:
        #                 pos_end_translate = match_translate_end.start()
        #                 text_translate = text[pos_start_translate:pos_end_translate]
        #             else:
        #                 text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text,
        #                                                                                                 pos_start_translate,
        #                                                                                                 len(text))
        #             if text_translate != "" and pos_end_translate <= pos_start_trlit:
        #                 text_translate_prev = text_translate
        #     if text_translate_prev != "" and Unfin_Data['trlit'] != "":
        #         text_transliterate_prev = Unfin_Data['trlit']
        #         flag_vyp = True
        #         Unfin_Data['trlit'] = ""
        #         return (text_translate_prev, text_transliterate_prev), flag_vyp, pos_end_trlit
        # # ----------------------------------------------------------------------------------------------
        if pos_end_trlit == len(text) or pos_end_trlit == -1 or get_next_line(text, pos_end_trlit)[1] ==len(text):
            pos_end_trlit = len(text)
            end_pos = pos_end_trlit
            # if not trlit_from_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            Unfin_Data['trlit'] = text_transliterate
            trlit_to_past = True
    if Unfin_Data['trlit'] != "" and not trlit_to_past:
        # if pos_start_trlit == pos_start:
        text_transliterate_prev = Unfin_Data['trlit']
        trlit_from_past = True
        if text_transliterate != "":
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
        else:
            text_transliterate = text_transliterate_prev
        Unfin_Data['trlit'] = ""
    if text_transliterate == "" or text_transliterate == {}:
        return (text_translate, text_transliterate), flag_vyp, end_pos

    # Pattern_search_translate = patterns_salim_assur_per_s["salim_start_per"]
    Pattern_search_translate = patterns_babylon_per_s
    Pattern_search_translate_re = re.compile(Pattern_search_translate, re.MULTILINE)
    match_translate = Pattern_search_translate_re.search(text, pos_end_trlit)
    if match_translate:
        # pos_end_trlit = match_translate.start()
        # text_transliterate = text[pos_start_trlit:pos_end_trlit]
        pos_start_translate = text.find("\n", match_translate.end())
        # pos_start_translate = match_translate.end()
        # text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate, len(text))
        # Pattern_search_translate_end, style, status_trlat = choose_pattern(text, patterns_akt2_per_e)
        Pattern_search_translate_end = patterns_babylon_per_e["babylon_end_per"]
        # Pattern_search_translate_end = r'^\d{3,4}\.\s*YBC\s*\d{4}\.'
        Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        if match_translate_end:
            pos_end_translate = match_translate_end.start()
            text_translate = text[pos_start_translate:pos_end_translate]
        else:
            #     pos_end_translate = len(text)
            text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate,
                                                                                            len(text))
            if text_translate == "":
                pos_end_translate = len(text)
        # text_translate = text[pos_start_translate:pos_end_translate]
    else:
        pos_end_translate = len(text)
    end_pos = pos_end_translate

    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not is_clean_akkadian_translation(text_translate):
            text_translate = ""
            end_pos = len(text)
        # else:
        #     if pos_end_translate == len(text):
        #         Unfin_Data["perevod"] = text_translate
    if text_transliterate != "":
        if not trlit_from_past and not trlit_to_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            if text_transliterate == {}:
                end_pos = len(text)
    if text_translate != "" and text_transliterate != {}:
        flag_vyp = True
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, end_pos

def after_extract_babylon(text_dict_tr: tuple, pos_s: int):
    flag_vyp = False
    list_trl_transl = None
    if all(text_dict_tr):
        list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
        if all(list_trl_transl):
            flag_vyp = True
    # # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, flag_vyp, pos_s

def extract_ninurta(text: str, start_pos: int, pattern: str, ind: int):
    text_translate = ""
    flag_vyp = False
    transl_from_past = False
    trlit_from_past = False
    trlit_to_past = False
    text_translate_prev = ""
    global Pattern_search_trlit, Pattern_search_trlit_re, Pattern_search_translate, Pattern_search_translate_re, Pattern_search_translate_end, Pattern_search_translate_end_re
    global patterns_ninurta_s, patterns_ninurta_per_s, patterns_ninurta_per_e, number_text, prov_prev_transl, patterns_byram_s, patterns_byram_per_s, patterns_byram_per_e
    if start_pos < 0 or start_pos >= len(text):
        return (text_translate, ""), flag_vyp, start_pos
    end_pos = len(text)
    if start_pos == 0:
        pattern = re.compile(pattern)
        match = pattern.search(text, start_pos)
        if not match:
            return (text_translate, ""), flag_vyp, end_pos
        print(f"Найден поисковый якорь : {match.group()}")
        start_pos = text.find("\n", match.end())
    Pattern_search_trlit = patterns_ninurta_s["ninurta_start_trl"]
    Pattern_search_trlit_re = re.compile(Pattern_search_trlit, re.MULTILINE)
    match_trlit = Pattern_search_trlit_re.search(text, start_pos)
    if match_trlit:
        # val_end = match_trlit.group("start_per")
        # patterns_babylon_per_s = rf'^{re.escape(val_end)}'
        # pos_start_trlit = text.find("\n", match_trlit.end())
        pos_start_trlit = match_trlit.end()
        text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, pos_start_trlit, len(text))
    else:
        text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, start_pos, len(text))
    Pattern_search_translate = patterns_ninurta_per_s["ninurta_start_per"]
    Pattern_search_translate_re = re.compile(Pattern_search_translate, re.MULTILINE)
    match_translate = Pattern_search_translate_re.search(text, pos_start_trlit)
    if match_translate:
        # pos_end_trlit = match_translate.start()
        # text_transliterate = text[pos_start_trlit:pos_end_trlit]
        pos_start_translate = match_translate.start()
        # text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate, len(text))
        # Pattern_search_translate_end, style, status_trlat = choose_pattern(text, patterns_akt2_per_e)
        Pattern_search_translate_end = patterns_ninurta_per_e["ninurta_end_per"]
        Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        if match_translate_end:
            # нужно вернуться на 1 строку
            pos_end_translate = match_translate_end.start()
            text_translate = text[pos_start_translate:pos_end_translate]
        else:
            #     pos_end_translate = len(text)
            text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate,
                                                                                            len(text))
            # if text_translate == "":
            #     pos_end_translate = len(text)
        # text_translate = text[pos_start_translate:pos_end_translate]
    else:
        text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_trlit,
                                                                                        len(text))
    if text_translate == "":
        pos_end_translate = len(text)
    end_pos = pos_end_translate
    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not is_clean_akkadian_translation(text_translate):
            text_translate = ""
            end_pos = len(text)
        # else:
        #     if pos_end_translate == len(text):
        #         Unfin_Data["perevod"] = text_translate
    if text_transliterate != "":
        if not trlit_from_past and not trlit_to_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            if text_transliterate == {}:
                end_pos = len(text)
    else:
        text_transliterate = {}
        end_pos = len(text)
    if text_translate != "" and text_transliterate != {}:
        flag_vyp = True
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, end_pos

def after_ninurta(text_dict_tr: tuple, pos_s: int):
    flag_vyp = False
    list_trl_transl = None
    if all(text_dict_tr):
        list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
        if all(list_trl_transl):
            flag_vyp = True
    # # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, flag_vyp, pos_s

def extract_nabu(text: str, start_pos: int, pattern: str, ind: int):
    text_translate = ""
    flag_vyp = False
    transl_from_past = False
    trlit_from_past = False
    trlit_to_past = False
    text_translate_prev = ""
    global Pattern_search_trlit, Pattern_search_trlit_re, Pattern_search_translate, Pattern_search_translate_re, Pattern_search_translate_end, Pattern_search_translate_end_re
    global patterns_nabu_s, patterns_nabu_per_s, patterns_nabu_per_e, number_text, prov_prev_transl
    if start_pos < 0 or start_pos >= len(text):
        return (text_translate, ""), flag_vyp, start_pos
    end_pos = len(text)
    if start_pos == 0:
        pattern = re.compile(pattern)
        match = pattern.search(text, start_pos)
        if not match:
            return (text_translate, ""), flag_vyp, end_pos
        print(f"Найден поисковый якорь: {match.group()}")
        start_pos = text.find("\n", match.end())
    text_transliterate, pos_end_trlit, pos_start_trlit = find_translit_by_rows(text, start_pos, len(text))
    # if text_transliterate != "":
        # # --------------------------------------------------------------------------------------------------
        # # проверка наличия перевода в начале текста для зарезервированной транслитерации
        # if pos_start_trlit > 200 and start_pos < 75 and Unfin_Data['trlit'] != "" and prov_prev_transl != number_text:
        #     prov_prev_transl = number_text
        #     Pattern_search_translate = patterns_sebahattin_per_s["sebahat_start_per"]
        #     Pattern_search_translate_re = re.compile(Pattern_search_translate)
        #     match_translate = Pattern_search_translate_re.search(text, start_pos)
        #     if match_translate:
        #         pos_start_translate = match_translate.start()
        #         if pos_start_translate < pos_start_trlit:
        #             Pattern_search_translate_end = patterns_sebahattin_per_e["sebahat_end_per"]
        #             Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        #             match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        #             if match_translate_end:
        #                 pos_end_translate = match_translate_end.start()
        #                 text_translate = text[pos_start_translate:pos_end_translate]
        #             else:
        #                 text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text,
        #                                                                                                 pos_start_translate,
        #                                                                                                 len(text))
        #             if text_translate != "" and pos_end_translate <= pos_start_trlit:
        #                 text_translate_prev = text_translate
        #     if text_translate_prev != "" and Unfin_Data['trlit'] != "":
        #         text_transliterate_prev = Unfin_Data['trlit']
        #         flag_vyp = True
        #         Unfin_Data['trlit'] = ""
        #         return (text_translate_prev, text_transliterate_prev), flag_vyp, pos_end_trlit
    #     # ----------------------------------------------------------------------------------------------
    #     if pos_end_trlit == len(text) or pos_end_trlit == -1 or get_next_line(text, pos_end_trlit)[1] ==len(text):
    #         pos_end_trlit = len(text)
    #         end_pos = pos_end_trlit
    #         # очистка от мусора транслитерации
    #         text_transliterate = process_text(text_transliterate)
    #         # словарь с ключами номерами и строками транслитерации
    #         text_transliterate = renumber_trust_source(text_transliterate)
    #         # text_transliterate = parse_numbered_fragments(text_transliterate)
    #         Unfin_Data['trlit'] = text_transliterate
    #         trlit_to_past = True
    # if Unfin_Data['trlit'] != "" and not trlit_to_past:
    #     text_transliterate_prev = Unfin_Data['trlit']
    #     trlit_from_past = True
    #     if text_transliterate != "":
    #         # очистка от мусора транслитерации
    #         text_transliterate = process_text(text_transliterate)
    #         # словарь с ключами номерами и строками транслитерации
    #         text_transliterate = renumber_trust_source(text_transliterate)
    #         # text_transliterate = parse_numbered_fragments(text_transliterate)
    #         text_transliterate = merge_if_consecutive(text_transliterate_prev, text_transliterate)
    #     else:
    #         text_transliterate = text_transliterate_prev
    #     Unfin_Data['trlit'] = ""
    if text_transliterate == "" or text_transliterate == {}:
        return (text_translate, text_transliterate), flag_vyp, end_pos
    Pattern_search_translate = patterns_nabu_per_s["nabu_start_per"]
    Pattern_search_translate_re = re.compile(Pattern_search_translate, re.MULTILINE)
    match_translate = Pattern_search_translate_re.search(text, pos_end_trlit)
    if match_translate:
        pos_start_translate = match_translate.end()
        Pattern_search_translate_end = patterns_nabu_per_e["nabu_end_per"]
        Pattern_search_translate_end_re = re.compile(Pattern_search_translate_end, re.MULTILINE)
        match_translate_end = Pattern_search_translate_end_re.search(text, pos_start_translate)
        if match_translate_end:
            pos_end_translate = match_translate_end.start()
            text_translate = text[pos_start_translate:pos_end_translate]
        else:
            text_translate, pos_end_translate, pos_start_translate = find_translate_by_rows(text, pos_start_translate,
                                                                                            len(text))
            if text_translate == "":
                pos_end_translate = len(text)
    else:
        pos_end_translate = len(text)
    end_pos = pos_end_translate

    if text_translate != "":
        # очистка мусора перевода
        text_translate = process_text(text_translate, False)
        if not is_clean_akkadian_translation(text_translate):
            text_translate = ""
            end_pos = len(text)
    if text_transliterate != "":
        if not trlit_from_past and not trlit_to_past:
            # очистка от мусора транслитерации
            text_transliterate = process_text(text_transliterate)
            # словарь с ключами номерами и строками транслитерации
            text_transliterate = renumber_trust_source(text_transliterate)
            # text_transliterate = parse_numbered_fragments(text_transliterate)
            if text_transliterate == {}:
                end_pos = len(text)
    if text_translate != "" and text_transliterate != {}:
        flag_vyp = True
    #     # очищенный от мусора текст и словарь транслитерации,
    #     # флаг выполнения, позиция конца перевода
    return (text_translate, text_transliterate), flag_vyp, end_pos

def after_nabu(text_dict_tr: tuple, pos_s: int):
    flag_vyp = False
    list_trl_transl = None
    if all(text_dict_tr):
        list_trl_transl = process_text_last(text_dict_tr[0], text_dict_tr[1])
        if all(list_trl_transl):
            flag_vyp = True
    # # кортеж списков транслитерации и перевода, флаг, конец перевода
    return list_trl_transl, flag_vyp, pos_s



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
    chars_to_remove = "!?/:.<>™‰˹˺[]⅁ᲟᲠᲢ¥#"
    table = str.maketrans("", "", chars_to_remove)
    # удаление ненужных символов
    a = a.translate(table)
    a = normalize_gaps(a)
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

# import math
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
def looks_like_real_translation(text, min_len=3):
    """Проверка: текст реально перевод, а не транслитерация/номер/каталог"""
    text = text.strip()
    if len(text) < min_len:
        return False

    digit_ratio = sum(c.isdigit() for c in text) / len(text)
    # if digit_ratio > 0.15:
    if digit_ratio > 0.5:
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

# def translate_to_english(text):
#     """
#     Переводит текст на английский, если язык не английский.
#     """
#     lang = detect_language(text)
#
#     if not lang:
#         # Если язык не определён, возвращаем оригинальный текст
#         return text
#
#     if lang != 'en':
#         try:
#             translated_text = GoogleTranslator(source=lang, target='en').translate(text)
#             return translated_text
#         except Exception as e:
#             print(f"Ошибка перевода: {e}")
#             return text
#     else:
#         # Если текст уже на английском
#         return text
def translate_to_english(text):
    try:
        tr = translator.translate(text)
        return tr if tr else text
    except:
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
    text = re.sub(r'TABLETLER[İI] u', 'TABLETLERİ II', text)
    text = re.sub(r'TABLETLER[!ÏIi]', 'TABLETLERİ', text)
    # text = re.sub(r'^\d+\.\s*§alim-A$$ur’s death', 'Salim-Assur’s death', text)
    # text = re.sub(r'^\d+\.\s*[§S]alim-A\$\$ur’s death', 'Salim-Assur’s death', text, flags=re.MULTILINE)
    # text = re.sub(r'^\d+\.\s*KÜLTEPE TABLETLERİ VII', 'KÜLTEPE TABLETLERİ VII', text, flags=re.MULTILINE)
    # text = re.sub(r'^\d+\.\s*SEBAHATTİN BAYRAM', 'SEBAHATTİN BAYRAM', text, flags=re.MULTILINE)
    # списки шаблонов поиска якорей для разных вариантов пар первого и второго блоков
    # перевод, транслитерация
    pattern1 = r'^(?:INDIVIDUAL AND FAMILY IN\s*[O0]\s*LD ASSYRIAN SOCIETY)\n'
    # pattern1 = r'\d{2,}:\s*(?:\d+[-–—]\d+\s*[:,)]\,?\s*[\s\S]{0,80}?)?\s*"'
    # pattern1 = r'\d{2,}:\s*(?:\d+[-–—]\d+[:,)]\s*[^"]{0,80}?)?\s*"'
    pattern2 = r'TABLETLERİ II\n'
    pattern3 = r'TABLETLERİ\n'
    # транслитерация, перевод и наоборот
    pattern4 = r'^'
    pattern5 = r'^'
    pattern6 = r'^'
    pattern7 = r"^(?:Salim-Assur[’']s|Sadaya[’']s son)"
    pattern8 = r'^SEBAHATTİN BAYRAM'
    pattern9 = r'^KÜLTEPE TABLETLERİ VII'
    pattern10 = r'^S. BAYRAM-R'
    pattern11 = r'^ALTBABYLONISCHE BRIEFE'
    pattern12 = r'^(?:\d+\s*)?Tukultï-Ninurta'
    pattern13 = r'^NABU'
    # и то и другое а потом выбирать

    # список списков шаблонов поиска первого блока
    all_patterns = [pattern1, pattern2, pattern3, pattern4, pattern5, pattern6, pattern7, pattern8, pattern9, pattern10, pattern11, pattern12, pattern13]
    len_arr = len(all_patterns)
    # len_arr = 1
    # список функций поиска первого блока соответствует списку списков шаблонов
    # extract_function_1 = [extract_quoted_substring, extract_salim_assur]
    extract_function_1 = [extract_quoted_substring, extract_ankara, extract_ankara_next, extract_letter_space_digit_colon_space, extract_numbs_and_diapasons, extract_numbs_and_diapasons, extract_salim_assur, extract_sebahat, extract_tabletVII, extract_tabletVII, extract_babylon, extract_ninurta, extract_nabu]
    # список функций поиска второго блока соответствует списку функций поиска первого блока
    # extract_function_2 = [extract_parenthesized_substring, after_extract_salim_assur]
    extract_function_2 = [extract_parenthesized_substring, extract_after_ankara, extract_after_ankara_next, extract_after_letter_space_digit_colon_space, after_extract_numbs_and_diapasones, after_extract_numbs_and_diapasones, after_extract_salim_assur, after_extract_sebahat, after_extract_tabletVII, after_extract_tabletVII, after_extract_babylon, after_ninurta, after_nabu]
    str_txt = [""] * len_arr
    str_txt_1 = [""] * len_arr
    # предварительная очистка
    text = cleaning_from_ocr_prelim(text)
    text = re.sub(r'^(?:\d+\.)?\s*[§S]alim-A$$ur’s', 'Salim-Assur’s', text)
    text = re.sub(r'^(?:\d+\.)?\s*[§S]alim-A\$\$ur’s', 'Salim-Assur’s', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s*K[Üh]LTEPE TABLETLERİ VII', 'KÜLTEPE TABLETLERİ VII', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\.\s*SEBAHATTİN BAYRAM', 'SEBAHATTİN BAYRAM', text, flags=re.MULTILINE)
    text = re.sub(r'^\d+\s*S. BAYRAM-R', 'S. BAYRAM-R', text, flags=re.MULTILINE)
    text = re.sub(r'^N\.?A\.?B\.?U\.?', 'NABU', text, flags=re.MULTILINE)

    i = 0
    csv_rows = []
    start_pos = 0

    while i < len_arr:
        pattern = all_patterns[i]
        print(f"Работаем с {i + 1} группой шаблонов")
        work = True
        while work:
            str_txt[i % len_arr], flag, next_pos = extract_function_1[i % len_arr](text, start_pos, pattern, i)

            if flag:
                print("Найден 1 блок")
                if isinstance(str_txt[i % len_arr], tuple):
                    text_tuple = str_txt[i % len_arr]
                    str_txt_1[i % len_arr], flag2, close_pos = extract_function_2[i % len_arr](text_tuple, next_pos)
                else:
                    str_txt_1[i % len_arr], flag2, close_pos = extract_function_2[i % len_arr](text, next_pos)
                if flag2:
                    print("Найден 2 блок")
                    # translate_str_arr = []
                    # accad_str_arr = []
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
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 3:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][1]
                                translate_str_arr = str_txt_1[i % len_arr][0]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 4:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 5:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 6:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 7:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 8:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 9:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]

                        case 10:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 11:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]
                        case 12:
                            if isinstance(str_txt_1[i % len_arr], tuple):
                                accad_str_arr = str_txt_1[i % len_arr][0]
                                translate_str_arr = str_txt_1[i % len_arr][1]
                            else:
                                translate_str_arr = str_txt_1[i % len_arr]
                                accad_str_arr = str_txt[i % len_arr]

                    # if isinstance(translate_str_arr, str):
                    #     translate_str_arr = [translate_str_arr]
                    # if isinstance(accad_str_arr, str):
                    #     accad_str_arr = [accad_str_arr]
                    num_i = 1
                    for translate_str, accad_str in zip(translate_str_arr, accad_str_arr):
                        # 1. Очистка перевода
                        t = translate_str.replace("\n", " ")

                        # 2. Очистка аккадского
                        a = accad_str.replace("\n", " ")
                        a = normalize_for_mt(a)

                        # --------------------------------------------------------------
                        # # 3. Токенизация перевода
                        t_sentences = sent_tokenize(t)
                        # t_sentences = [sent for sent in t_sentences if looks_like_real_translation(sent)]
                        # # определение языка и перевод на английский, если перевод не английский\n",
                        # t_sentences = [translate_to_english(sent) if detect_language(sent) != 'en' else sent for sent in t_sentences]
                        # t_sentences = [
                        #     sent.strip()
                        #     for sent in t_sentences
                        #     if looks_like_real_translation(sent)
                        # ]
                        #
                        # t_sentences = [
                        #     translate_to_english(sent)
                        #     for sent in t_sentences
                        # ]
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
                    # print("Ищем следующий 1 блок")
                else:
                    print("Не найден 2 блок")
                    # не найден 2 блок,
                    if close_pos < len(text):
                        # ищем следующие первые
                        # print("Ищем следующий 1 блок")
                        # меняем шаблон
                        start_pos = close_pos + 1
                    else:
                        # print("Прошли текст, меняем шаблон")
                        # прошли текст, меняем шаблон
                        work = False
                        start_pos = 0
            else:
                print("Не найден 1 блок")
                # не найден первый блок
                if next_pos < len(text):
                # if len(text) - next_pos < 6:
                    # print("Продолжаем по тексту поиск 1 блока")
                    # продолжаем идти по тексту
                    start_pos = next_pos + 1
                else:
                    # print("Прошли текст, меняем шаблон")
                    # прошли текст, меняем шаблон
                    work = False
                    start_pos = 0
        # меняем очерёдность поиска блоков
        i += 1
        # меняем шаблон
        # if i < len_arr:
        #     print(f"Переходим на {i + 1} группу шаблонов")
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
# import csv
# from io import StringIO

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
                "oare_id": global_id,
                "transliteration": accad,
                "translation": trans
            })
            global_id += 1

    df =  pd.DataFrame(rows, columns=["oare_id", "transliteration", "translation"])
    # сохранение файла
    df.to_csv("train_accad.csv", index=False, encoding="utf-8")

    return df


# def print_file_head(path, n=5, encoding="utf-8"):
#     with open(path, "r", encoding=encoding) as f:
#         for i, line in enumerate(f):
#             if i >= n:
#                 break
#             print(f"{i}: {line.rstrip()}")


# Завантаження даних з CSV-файлу
thiscompteca = "D:/Projects/Python/Конкурсы/Old_accad_translate"
# thiscompteca = "G:/Visual Studio 2010/Projects/Python/Old_accad_translate/"
csv_file_path = thiscompteca+'/data/publications.csv'
df_trnl = pd.read_csv(csv_file_path)
# ----------------------------------------
df_trnl = df_trnl.drop_duplicates()
# ------------------------------------------------
# csv_file_path = thiscompteca+'/data/train.csv'
# df_trnl = pd.read_csv(csv_file_path)
# print(df_trnl.shape[0])
# csv_file_path = thiscompteca+'/train_accad.csv'
# df_trnl = pd.read_csv(csv_file_path)
# print(df_trnl.shape[0])
# # csv_file_path = thiscompteca+'/train_combined.csv'
# # df_trnl = pd.read_csv(csv_file_path)
# # print(df_trnl.shape[0])
# csv_file_path = thiscompteca+'/data/test.csv'
# df_trnl = pd.read_csv(csv_file_path)
# print(df_trnl.shape[0])
# df_trnl.to_csv("train.csv", index=False)
# # -------------------------------------------
# csv_file_path = thiscompteca+'/data/test.csv'
# df_txt = pd.read_csv(csv_file_path)
# num_row = 0
# for num_row in range(df_txt.shape[0]):
#     if num_row > 5:
#         break
#     for num_col in range(df_txt.shape[1]):
#         print(df_txt.iat[num_row, num_col])
#     print('-' * 50)
# ----------------------------------------
# df_trnl = df_trnl.drop_duplicates()
# print(df_trnl[df_trnl].head(10))  # Перші 5 строк даних
# print(df_trnl.shape)  # Dataset Shape
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
    number_text = i
    # print(f"{num_i + 1} текст начинаем искать")
    # print(f"{num + 1} пару блоков начинаем искать.\n")
    print(f"Index = {i}\n")
    # if i == 74880:
    if i == 69686:        #206345:  #17542
    #не печатает переводы
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
            # print(row)
            # print(len(row))
            # trl, transl = row
            # for line in row:
            # trl, transl = parse_csv_line(str(row))
            # print(f"\nТранслитерация{num_i + 1}-{num_i+1}\n {trl}")
            # print(f"\nПеревод{num_i + 1}-{num_i+1}\n {transl}")
            # print("-" * 50)
            all_rows.append(row)
            # print(f"{num + 1} пара блоков найдена.\n")
            # print(row)
            num += 1
    # print(f"{num_i + 1} текст прошли")
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