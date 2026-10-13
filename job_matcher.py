import os
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

W_COVERAGE = 0.70  # 70% weight on explicit skill coverage
W_TFIDF = 0.30     # 30% weight on TF-IDF textual cosine similarity
TOKEN_PATTERN = r"[a-z0-9]+(?:[.\-][a-z0-9]+)*[+#]*"
MODEL_PATH = "data/ml_resume_classifier.joblib"


def predict_ml_role(clean_resume: str) -> dict:
    """
    Predict job role class using the Supervised ML model trained on dataset resumes.
    Returns dict with top_role, confidence_score, and probabilities.
    """
    if not os.path.exists(MODEL_PATH):
        return {"available": False, "reason": "ML Model not trained yet."}

    try:
        pipeline = joblib.load(MODEL_PATH)
        probas = pipeline.predict_proba([clean_resume.lower()])[0]
        classes = pipeline.classes_

        prob_df = pd.DataFrame({
            "Role": classes,
            "Probability": [round(float(p) * 100, 1) for p in probas]
        }).sort_values("Probability", ascending=False).reset_index(drop=True)

        top_row = prob_df.iloc[0]
        return {
            "available": True,
            "top_role": top_row["Role"],
            "confidence": top_row["Probability"],
            "probabilities": prob_df,
        }
    except Exception as e:
        return {"available": False, "reason": str(e)}


def score_roles(clean_resume: str, found_skills: dict, roles_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compare clean resume text and detected skills against all job roles in roles_df.
    Returns a DataFrame sorted by match Score descending.
    """
    if roles_df.empty:
        return pd.DataFrame()

    # Construct role text corpus for TF-IDF
    corpus_docs = []
    for _, row in roles_df.iterrows():
        req_text = str(row["required_skills"]).replace(";", " ")
        desc_text = str(row.get("description", ""))
        corpus_docs.append(f"{req_text} {desc_text}".lower())

    # Calculate TF-IDF Cosine Similarity
    vectorizer = TfidfVectorizer(token_pattern=TOKEN_PATTERN)
    tfidf_matrix = vectorizer.fit_transform(corpus_docs + [clean_resume.lower()])
    resume_vector = tfidf_matrix[-1]
    role_vectors = tfidf_matrix[:-1]

    cos_sims = cosine_similarity(resume_vector, role_vectors).flatten()

    results = []
    found_set = {s.lower() for s in found_skills.keys()}

    for (idx, row), sim in zip(roles_df.iterrows(), cos_sims):
        role_name = str(row["role"]).strip()
        req_skills_raw = [s.strip() for s in str(row["required_skills"]).split(";") if s.strip()]

        matched = []
        missing = []
        for req in req_skills_raw:
            if req.lower() in found_set:
                matched.append(req)
            else:
                missing.append(req)

        req_count = max(len(req_skills_raw), 1)
        coverage_ratio = len(matched) / req_count

        # Compute hybrid score
        raw_score = 100.0 * (W_COVERAGE * coverage_ratio + W_TFIDF * max(float(sim), 0.0))
        final_score = min(round(raw_score, 1), 100.0)

        results.append({
            "Role": role_name,
            "Score": final_score,
            "Coverage": round(100.0 * coverage_ratio, 1),
            "Similarity": round(float(sim), 3),
            "Matched": matched,
            "Missing": missing,
            "Description": str(row.get("description", "")),
        })

    results_df = pd.DataFrame(results).sort_values("Score", ascending=False).reset_index(drop=True)
    return results_df
