import re

def tokenize(text: str) -> list[str]:
    """Convert text into normalized lexical tokens for BM25."""
    tokens: list[str] = []
    for word in re.findall(r"\w+", text):
        if not word.strip("_"):
            continue
        separated = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", word)
        separated = re.sub(r"([A-Z])([A-Z][a-z])", r"\1 \2", separated)
        parts = [
            part.lower()
            for part in re.split(r"[_\s]+", separated)
            if part
        ]
        normalized = word.lower()
        tokens.append(normalized)
        if parts != [normalized]:
            tokens.extend(parts)
    return tokens
