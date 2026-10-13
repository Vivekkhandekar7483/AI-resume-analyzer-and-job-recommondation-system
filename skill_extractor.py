import re
import pandas as pd

# Optional spaCy integration for advanced entity/noun-chunk extraction
try:
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except Exception:
        nlp = None
except ImportError:
    nlp = None


def load_skills(path="data/skill_dictionary.csv") -> dict:
    """Load skill dictionary with alias mapping and topics."""
    df = pd.read_csv(path).fillna("")
    skills = {}
    for _, r in df.iterrows():
        skill_name = str(r["skill"]).strip()
        if not skill_name or skill_name.startswith("#"):
            continue
        aliases_str = str(r.get("aliases", "")).strip()
        aliases = [skill_name.lower()]
        if aliases_str:
            aliases.extend([a.strip().lower() for a in aliases_str.split("|") if a.strip()])
        skills[skill_name] = {
            "category": str(r.get("category", "General")).strip(),
            "topic": str(r.get("topic", skill_name)).strip(),
            "aliases": list(dict.fromkeys(aliases)),  # deduplicate preserving order
        }
    return skills


def extract_skills(clean_resume: str, skills: dict) -> dict:
    """
    Return a dictionary of {skill_name: category} for all skills detected in clean_resume.
    Uses precise boundary regex to match skills and aliases (e.g. C++, C#, .NET, React, SQL).
    """
    found = {}
    if not clean_resume or not skills:
        return found

    text = f" {clean_resume} "

    for name, info in skills.items():
        for alias in info["aliases"]:
            # Handle boundary check safely for symbols like ++, #, .net
            escaped = re.escape(alias)
            pattern = r"(?<![a-z0-9+#.])" + escaped + r"(?![a-z0-9+#])"
            if re.search(pattern, text, flags=re.IGNORECASE):
                found[name] = info["category"]
                break

    # Advanced improvement: spaCy noun chunks matching if spaCy model loaded
    if nlp and len(found) < 3:
        doc = nlp(clean_resume)
        extracted_chunks = {chunk.text.lower() for chunk in doc.noun_chunks}
        for name, info in skills.items():
            if name not in found:
                for alias in info["aliases"]:
                    if alias in extracted_chunks:
                        found[name] = info["category"]
                        break

    return found
