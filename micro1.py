import re

text = " 1-12: "
print(re.search(r"\s\d{1,2}-\d{1,2}:\s", text))
