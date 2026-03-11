import re

BIBLIO_RE = re.compile(
    r"""
    No\.?\s*\d+           # No. 309
    |Nr\.?\s*\d+          # Nr. 309
    |\b\d+(?:/?[a-z]+)\s+\d+\b    # 88/k 595 или 88k 595
    |\d+\s+[A-ZÇĞİÖŞÜ][a-zçğıöşü]+\s*\d{4}   # Фамилия и год, напр. 49 Çeçen 1995
    """,
    re.VERBOSE
)

txt = """ú-lá 1 2/3 ma-na / ú-lá e-li-iš"""
# pattern = re.compile(r"\b\d+(?:/?[a-z]+)?\s+\d+\b", re.VERBOSE)
if BIBLIO_RE.search(txt):
    print("BIBLIO")