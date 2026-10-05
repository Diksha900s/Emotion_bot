import re
CONTRACTIONS = {"can't": "cannot", "won't": "will not", "i'm": "i am", "don't": "do not",
                "didn't": "did not", "n't": " not", "'re": " are", "'ve": " have", "'ll": " will"}
STOP = {"the","a","an","is","was","to","of","and","in","on","at","it","my","me","i","that","this","so"}
def clean(t: str) -> str:
    t = t.lower()
    t = re.sub(r"http\S+|<.*?>|#\w+", " ", t)
    for k, v in CONTRACTIONS.items(): t = t.replace(k, v)
    t = re.sub(r"[^a-z\s]", " ", t)          # drop punctuation/digits
    return " ".join(w for w in t.split() if w not in STOP)
