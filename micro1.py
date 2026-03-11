import re

BIBLIO_RE = re.compile(
    r"""
    No\.?\s*\d+           # No. 309
    |Nr\.?\s*\d+          # Nr. 309
    |\b\d+(?:/?[a-z]+)?\s+\d+\b    # 88/k 595 или 88k 595
    |\d+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s*\d{4}   # Фамилия и год, напр. 49 Çeçen 1995
    """,
    re.VERBOSE
)

txt = """Pí-lá-ah-Ištar / a-na Ma-nu<-um>-ba-lúm-A-šur
ú Šu-zu-zu / iṣ-ba-at-ni-a-tí-ma
um-ma Pí-lá-ah-Ištar-ma a-na Ma-nu-um-ba-lu-um-A-šur
ú Šu-zu-zu-ma / a-ha-at-ni
5.ir-té-be / ba-a-nim / KÙ.BABBAR ma-la
i-ga-mu-ru 3 né-nu a-na
ki-iš-da-tí-ni / lu ni-iš-ta-pá-ak-ma
ú-la É DAM.GÀR-ri-im / KÙ.BABBAR
a-na ṣí-ib-tim lu ni-il5-qé-ma / gam-ra-am
10.lu ni-ig-mu-ur / a-ha-at-ni / a-na
mu-tim / lu ni-dí-in-ší / um-ma Ma-nu-um[-ba-lu-um-A-šu]r-ma
ú Šu-zu-zu-ma / a-na Pí-lá-ah-Ištar
KÙ.BABBAR ú-lá ni-šu / a-li-ik
i-na pí-ni / KÙ.BABBAR ma-lá / ta-ga-mu-ru
15.É DAM.GÀR-ri-im a-na ṣí-ib-tim
le-qé-ma / gam-ra-am / gu5-mu-ur-ma
a-ha-at-ni a-na mu-tim / dí-ší-ma
KÙ.BABBAR ù ṣí-ba-sú / ša i-na
É DAM.GÀR-ri-im ta-la-qé-ú-ma
20.ta-ga-mu-ru gu5-mu-ur-ma
"""
# pattern = re.compile(r"\b\d+(?:/?[a-z]+)?\s+\d+\b", re.VERBOSE)
if BIBLIO_RE.search(txt):
    print("BIBLIO")