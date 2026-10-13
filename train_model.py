"""
train_model.py - Supervised ML Trainer for AI Resume Analyzer

Reads all 227+ real resumes in data/Resumes/, auto-labels them based on domain signatures,
trains a Scikit-Learn TF-IDF + LogisticRegression / NaiveBayes model, evaluates performance,
and saves the trained classifier artifact to data/ml_resume_classifier.joblib.

Usage:
    python train_model.py
"""
import os
import re
import json
import joblib
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import classification_report, accuracy_score

from resume_parser import extract_text
from text_cleaner import clean_text

DATASET_DIR = Path("data/Resumes")
MODEL_OUT = Path("data/ml_resume_classifier.joblib")
META_OUT = Path("data/ml_model_metadata.json")


def auto_label_resume(filename: str, clean_text_str: str) -> str:
    """Label resume into standardized job roles based on filename and header content."""
    f_lower = filename.lower()
    t_header = clean_text_str[:1500].lower()
    full_text = (f_lower + " " + clean_text_str).lower()

    # High precision filename rules
    if re.search(r"\b(java|j2ee)\b", f_lower):
        return "Java Developer"
    if re.search(r"\b(hadoop|big data)\b", f_lower):
        return "Hadoop / Big Data Engineer"
    if re.search(r"\b(qa|testing|test)\b", f_lower):
        return "QA / Test Engineer"
    if re.search(r"\b(pm|project manager|scrum master|pmp|agile program manager)\b", f_lower):
        return "Project Manager"
    if re.search(r"\b(bsa|ba|business analyst)\b", f_lower):
        return "Business Analyst"
    if re.search(r"\b(php)\b", f_lower):
        return "PHP Developer"

    # Header / Title matching
    if re.search(r"\b(java developer|j2ee developer|full stack java)\b", t_header):
        return "Java Developer"
    if re.search(r"\b(scrum master|project manager|program manager)\b", t_header):
        return "Project Manager"
    if re.search(r"\b(business analyst|business systems analyst)\b", t_header):
        return "Business Analyst"
    if re.search(r"\b(hadoop developer|big data engineer)\b", t_header):
        return "Hadoop / Big Data Engineer"
    if re.search(r"\b(qa engineer|test automation|software test)\b", t_header):
        return "QA / Test Engineer"
    if re.search(r"\b(php developer|web developer)\b", t_header):
        return "Web Developer"
    if re.search(r"\b(data analyst|business intelligence)\b", t_header):
        return "Data Analyst"
    if re.search(r"\b(machine learning|data scientist)\b", t_header):
        return "Machine Learning Engineer"

    # Score-based fallback
    scores = {
        "Java Developer": len(re.findall(r"\b(java|j2ee|spring|hibernate|maven)\b", full_text)),
        "Business Analyst": len(re.findall(r"\b(business analyst|brd|prd|gap analysis|use cases)\b", full_text)),
        "Project Manager": len(re.findall(r"\b(project manager|scrum|agile|pmp|stakeholder|sprint)\b", full_text)),
        "Hadoop / Big Data Engineer": len(re.findall(r"\b(hadoop|hdfs|hive|pig|spark|mapreduce)\b", full_text)),
        "QA / Test Engineer": len(re.findall(r"\b(qa|selenium|test cases|qtp|testing)\b", full_text)),
        "Web Developer": len(re.findall(r"\b(php|javascript|html|css|wordpress)\b", full_text)),
    }
    best_role = max(scores, key=scores.get)
    if scores[best_role] >= 3:
        return best_role
    return "Software Engineer"


def build_and_train():
    if not DATASET_DIR.is_dir():
        print(f"Error: Dataset directory {DATASET_DIR} not found.")
        return

    files = sorted([f for f in DATASET_DIR.iterdir() if f.suffix.lower() in (".docx", ".pdf")])
    if not files:
        print(f"No .docx or .pdf files found in {DATASET_DIR}")
        return

    print(f"Found {len(files)} resumes in {DATASET_DIR}. Processing & extracting text...")

    records = []
    for i, path in enumerate(files, 1):
        try:
            raw_text = extract_text(path.read_bytes(), path.name)
            cleaned = clean_text(raw_text)
            if len(cleaned) < 50:
                continue
            label = auto_label_resume(path.name, cleaned)
            records.append({"filename": path.name, "text": cleaned, "label": label})
        except Exception as e:
            print(f"Skipping {path.name} due to extraction error: {e}")

    df = pd.DataFrame(records)
    print(f"\nClean dataset built with {len(df)} samples.")
    print("Class label distribution:")
    print(df["label"].value_counts().to_string())

    # Filter out rare classes with < 2 samples for stratified cross-validation
    class_counts = df["label"].value_counts()
    valid_classes = class_counts[class_counts >= 2].index
    df = df[df["label"].isin(valid_classes)].reset_index(drop=True)

    X = df["text"]
    y = df["label"]

    # Construct Scikit-Learn Pipeline
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            token_pattern=r"[a-z0-9]+(?:[.\-][a-z0-9]+)*[+#]*",
            ngram_range=(1, 2),
            max_features=3000,
            sublinear_tf=True
        )),
        ("classifier", LogisticRegression(C=1.5, max_iter=1000, class_weight="balanced"))
    ])

    # Perform 5-Fold Stratified Cross Validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")
    mean_acc = round(float(scores.mean() * 100), 2)
    print(f"\n5-Fold Cross-Validation Accuracy: {mean_acc}% (+/- {round(float(scores.std() * 100), 2)}%)")

    # Fit pipeline on full dataset
    pipeline.fit(X, y)

    # Save trained model artifact
    MODEL_OUT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_OUT)
    print(f"Trained model saved to: {MODEL_OUT}")

    # Export model metadata
    classes = list(pipeline.classes_)
    metadata = {
        "total_resumes": len(df),
        "classes": classes,
        "cv_accuracy": mean_acc,
        "class_distribution": df["label"].value_counts().to_dict(),
    }
    with open(META_OUT, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {META_OUT}")


if __name__ == "__main__":
    build_and_train()
