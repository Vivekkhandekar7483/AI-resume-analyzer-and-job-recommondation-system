"""
gemini_analyzer.py - Structured resume analysis with Gemini.

Why this exists: the TF-IDF / keyword pipeline only sees exact words. Gemini reads the resume
like a recruiter (projects, experience, context), so it can recognise skills that are described
but not named in the dictionary, and judge how well the candidate fits EVERY role.

The output is forced into a JSON schema (no free text to parse) and then validated against the
resume text, so invented skills are removed before anything is shown to the student.

Only the CLEANED resume text is sent (emails, phones, URLs and protected attributes are already
stripped by text_cleaner.clean_text).
"""
import json
import os
from typing import List

from pydantic import BaseModel
from google import genai
from google.genai import types

MODELS = ["gemini-2.5-flash", "gemini-2.0-flash"]
MAX_CHARS = 8000  # enough for a normal 2-3 page resume; less text = faster answer
TIMEOUT_MS = 45000  # give up on a stuck call after 45 s and try the next model


class RoleFit(BaseModel):
    role: str
    fit_score: int  # 0-100
    reason: str


class GeminiAnalysis(BaseModel):
    predicted_role: str
    confidence: int  # 0-100
    experience_level: str  # Fresher / Junior / Mid / Senior
    summary: str
    role_fits: List[RoleFit]
    verified_skills: List[str]
    extra_skills: List[str]
    missing_skills: List[str]
    strengths: List[str]
    weaknesses: List[str]
    resume_improvements: List[str]


SYSTEM_INSTRUCTION = (
    "You are an expert technical recruiter and career mentor evaluating a student's resume. "
    "Judge ONLY job-related evidence: skills, projects, education, internships and experience. "
    "Never consider or mention gender, age, religion, nationality, marital status, disability or photo. "
    "Never invent skills: a skill counts only if the resume text supports it. "
    "Scores are estimates for learning guidance, not hiring decisions. "
    "Be concrete and short; every bullet must be actionable."
)


def get_api_key() -> str:
    return (os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or "").strip()


def is_configured() -> bool:
    return bool(get_api_key())


def _build_prompt(resume_text: str, target_role: str, roles: List[dict], detected: List[str]) -> str:
    role_lines = "\n".join(f"- {r['role']}: needs {r['required_skills']}" for r in roles)
    return (
        f"TARGET ROLE: {target_role}\n\n"
        f"ALL ROLES (score the candidate against EVERY one, use these exact names):\n{role_lines}\n\n"
        f"SKILLS ALREADY DETECTED BY KEYWORD MATCHING: {', '.join(detected) or 'none'}\n\n"
        "TASK:\n"
        "1. role_fits: one entry per role above, fit_score 0-100 based on real evidence "
        "(projects and experience weigh more than a keyword in a skills list).\n"
        "2. predicted_role: the single best-fitting role from the list; confidence 0-100.\n"
        "3. verified_skills: required skills of the TARGET ROLE that the resume truly shows.\n"
        "4. missing_skills: required skills of the TARGET ROLE with no evidence.\n"
        "5. extra_skills: other relevant technical skills found in the resume that are not "
        "in the detected list.\n"
        "6. strengths, weaknesses, resume_improvements: 3-5 short bullets each.\n"
        "7. experience_level: Fresher, Junior, Mid or Senior.\n"
        "8. summary: 2 sentences.\n\n"
        f"RESUME TEXT (lowercased, cleaned):\n\"\"\"\n{resume_text[:MAX_CHARS]}\n\"\"\""
    )


def _clamp(v, lo=0, hi=100) -> int:
    try:
        return max(lo, min(hi, int(round(float(v)))))
    except (TypeError, ValueError):
        return lo


def _validate(raw: dict, resume_text: str, role_names: List[str], detected: List[str]) -> dict:
    """Clamp numbers, force role names to the known list, drop skills the resume does not contain."""
    text = resume_text.lower()
    by_lower = {r.lower(): r for r in role_names}

    fits = {}
    for item in raw.get("role_fits", []):
        name = by_lower.get(str(item.get("role", "")).strip().lower())
        if name:
            fits[name] = {"role": name, "fit_score": _clamp(item.get("fit_score")),
                          "reason": str(item.get("reason", ""))[:300]}

    predicted = by_lower.get(str(raw.get("predicted_role", "")).strip().lower())
    if not predicted and fits:
        predicted = max(fits.values(), key=lambda x: x["fit_score"])["role"]

    detected_lower = {d.lower() for d in detected}

    def grounded(skills):
        out = []
        for s in skills or []:
            s = str(s).strip()
            if s and (s.lower() in detected_lower or s.lower() in text):
                out.append(s)
        return list(dict.fromkeys(out))

    def clean_list(items, n=6):
        return [str(x).strip() for x in (items or []) if str(x).strip()][:n]

    return {
        "predicted_role": predicted or "",
        "confidence": _clamp(raw.get("confidence")),
        "experience_level": str(raw.get("experience_level", "")).strip() or "Fresher",
        "summary": str(raw.get("summary", "")).strip(),
        "role_fits": list(fits.values()),
        "verified_skills": grounded(raw.get("verified_skills")),
        "extra_skills": grounded(raw.get("extra_skills")),
        "missing_skills": clean_list(raw.get("missing_skills"), 12),
        "strengths": clean_list(raw.get("strengths")),
        "weaknesses": clean_list(raw.get("weaknesses")),
        "resume_improvements": clean_list(raw.get("resume_improvements")),
    }


def analyze(resume_text: str, target_role: str, roles: List[dict], detected: List[str]) -> dict:
    """Return a validated analysis dict. Raises RuntimeError with a readable message on failure."""
    key = get_api_key()
    if not key:
        raise RuntimeError("No Gemini API key found. Add GEMINI_API_KEY to your .env file.")

    try:
        client = genai.Client(api_key=key, http_options=types.HttpOptions(timeout=TIMEOUT_MS))
    except AttributeError:  # older google-genai without HttpOptions
        client = genai.Client(api_key=key)
    prompt = _build_prompt(resume_text, target_role, roles, detected)

    def make_config(model: str):
        kwargs = dict(
            system_instruction=SYSTEM_INSTRUCTION,
            temperature=0.2,  # low = consistent scores for the same resume
            response_mime_type="application/json",
            response_schema=GeminiAnalysis,
        )
        if model.startswith("gemini-2.5"):
            # 2.5 models "think" before answering, which adds 10-40 s; this task does not need it
            try:
                kwargs["thinking_config"] = types.ThinkingConfig(thinking_budget=0)
            except AttributeError:
                pass
        return types.GenerateContentConfig(**kwargs)

    last_error = None
    for model in MODELS:
        try:
            resp = client.models.generate_content(model=model, contents=prompt, config=make_config(model))
            data = None
            if getattr(resp, "parsed", None) is not None:
                parsed = resp.parsed
                data = parsed.model_dump() if hasattr(parsed, "model_dump") else dict(parsed)
            elif resp.text:
                data = json.loads(resp.text)
            if data:
                result = _validate(data, resume_text, [r["role"] for r in roles], detected)
                result["model_used"] = model
                return result
        except Exception as e:  # try the next model
            last_error = e

    raise RuntimeError(
        f"Gemini analysis failed ({type(last_error).__name__}: {last_error}). "
        "Check your GEMINI_API_KEY and internet connection."
    )