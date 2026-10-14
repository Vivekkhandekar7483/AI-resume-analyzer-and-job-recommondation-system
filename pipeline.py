"""
pipeline.py - ONE place that runs the whole analysis.
Both the Streamlit app (local mode) and the FastAPI backend call analyze_resume(),
so the two always give identical results.

Final score per role = 40% rule-based hybrid score + 60% Gemini fit score (when Gemini is on).
If Gemini is off or fails, the app silently falls back to the hybrid score only.
"""
import json

import pandas as pd

from resume_parser import extract_text
from text_cleaner import clean_text
from skill_extractor import extract_skills
from job_matcher import score_roles, predict_ml_role
from roadmap_generator import make_roadmap
import gemini_analyzer

W_RULES = 0.40
W_GEMINI = 0.60


def analyze_resume(file_bytes: bytes, filename: str, target_role: str,
                   skills_dict: dict, roles_df: pd.DataFrame, use_gemini: bool = True) -> dict:
    raw_text = extract_text(file_bytes, filename)
    clean = clean_text(raw_text)
    if len(clean) < 40:
        raise ValueError("Insufficient text extracted. Scanned image-only PDFs are not supported.")

    skills_found = extract_skills(clean, skills_dict)
    scores = score_roles(clean, skills_found, roles_df)
    if target_role not in set(scores["Role"]):
        raise ValueError(f"Unknown target role: {target_role}")

    # Optional Gemini layer
    gemini, gemini_error = None, None
    if use_gemini:
        if not gemini_analyzer.is_configured():
            gemini_error = "Gemini API key not configured."
        else:
            try:
                roles = roles_df[["role", "required_skills"]].to_dict("records")
                gemini = gemini_analyzer.analyze(clean, target_role, roles, list(skills_found))
            except Exception as e:
                gemini_error = str(e)

    # Blend scores
    scores["HybridScore"] = scores["Score"]
    if gemini:
        fit = {f["role"]: f["fit_score"] for f in gemini["role_fits"]}
        scores["GeminiScore"] = scores["Role"].map(fit)
        blended = W_RULES * scores["HybridScore"] + W_GEMINI * scores["GeminiScore"]
        scores["Score"] = blended.where(scores["GeminiScore"].notna(), scores["HybridScore"]).round(1)
        scores = scores.sort_values("Score", ascending=False).reset_index(drop=True)

    target_row = scores[scores["Role"] == target_role].iloc[0]
    roadmap = make_roadmap(list(target_row["Missing"]), skills_dict)

    ml = predict_ml_role(clean)
    if ml.get("available"):
        ml["probabilities"] = ml["probabilities"].to_dict("records")

    return {
        "file_name": filename,
        "target_role": target_role,
        "char_count": len(clean),
        "skills_detected": {k: v for k, v in skills_found.items()},
        # to_json turns NaN into null; a plain to_dict leaves NaN, which FastAPI cannot send as JSON
        "role_scores": json.loads(scores.to_json(orient="records")),
        "ml_prediction": ml,
        "gemini": gemini,
        "gemini_error": gemini_error,
        "roadmap": roadmap,
        "score_method": "40% rules + 60% Gemini" if gemini else "rules only (skill coverage + TF-IDF)",
    }