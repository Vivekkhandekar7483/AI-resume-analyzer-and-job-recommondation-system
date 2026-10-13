import re

# Protected personal details are removed so they can never influence scoring.
PROTECTED = r"\b(gender|sex|age|date of birth|dob|marital status|religion|nationality)\b\s*[:\-]\s*\w+"


def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(PROTECTED, " ", text)
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"\S+@\S+", " ", text)                    # emails
    text = re.sub(r"\+?\d[\d\s\-()]{8,}\d", " ", text)      # phone numbers
    text = re.sub(r"[^a-z0-9+#./\s-]", " ", text)           # keep + # . / - for C++, C#, .NET
    return re.sub(r"\s+", " ", text).strip()
