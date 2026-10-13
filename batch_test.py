"""
batch_test.py - run the resume pipeline over a whole folder of real resumes
and log the result of every file to a CSV, so you can see where text
extraction or skill extraction is going wrong without clicking through
the Streamlit app one file at a time.

Usage:
    python batch_test.py "C:\\Users\\Asus\\Downloads\\archive (4)\\Resumes"
    python batch_test.py path/to/resumes --out reports/batch_results.csv

Only reads files. Never modifies or deletes anything in the input folder.
"""
import argparse
import sys
from pathlib import Path

import pandas as pd

from resume_parser import extract_text
from text_cleaner import clean_text
from skill_extractor import load_skills, extract_skills
from job_matcher import score_roles

MIN_CHARS = 50  # below this, extraction almost certainly failed


def process_folder(folder: Path, skills: dict, roles: pd.DataFrame) -> pd.DataFrame:
    files = sorted(f for f in folder.iterdir() if f.suffix.lower() in (".pdf", ".docx"))
    if not files:
        print(f"No .pdf or .docx files found in {folder}")
        return pd.DataFrame()

    rows = []
    for i, path in enumerate(files, 1):
        print(f"[{i}/{len(files)}] {path.name}")
        row = {"filename": path.name, "status": "ok", "char_count": 0,
               "num_skills": 0, "skills_found": "", "top_role": "",
               "top_score": "", "notes": ""}
        try:
            raw_text = extract_text(path.read_bytes(), path.name)
            clean = clean_text(raw_text)
            row["char_count"] = len(clean)

            if len(clean) < MIN_CHARS:
                row["status"] = "warning"
                row["notes"] = "Very little text extracted - likely a scanned/image resume or a parsing failure."
                rows.append(row)
                continue

            found = extract_skills(clean, skills)
            row["num_skills"] = len(found)
            row["skills_found"] = "; ".join(sorted(found))

            top = score_roles(clean, found, roles).iloc[0]
            row["top_role"] = top["Role"]
            row["top_score"] = top["Score"]

            if not found:
                row["status"] = "warning"
                row["notes"] = "Text extracted fine but zero skills matched - dictionary may not cover this field."
        except Exception as e:
            row["status"] = "error"
            row["notes"] = f"{type(e).__name__}: {e}"

        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(description="Batch-test the resume pipeline over a folder of resumes.")
    ap.add_argument("folder", help="Path to a folder of .pdf/.docx resumes")
    ap.add_argument("--out", default="reports/batch_results.csv", help="Output CSV path")
    args = ap.parse_args()

    folder = Path(args.folder)
    if not folder.is_dir():
        print(f"Not a folder: {folder}")
        sys.exit(1)

    skills = load_skills()
    roles = pd.read_csv("data/job_roles.csv")
    df = process_folder(folder, skills, roles)
    if df.empty:
        return

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print("\n--- Summary ---")
    print(df["status"].value_counts().to_string())
    print(f"\nMedian skills found per resume: {df['num_skills'].median()}")
    print(f"Files with zero skills matched: {(df['num_skills'] == 0).sum()} / {len(df)}")
    print(f"Files that errored out: {(df['status'] == 'error').sum()} / {len(df)}")
    print(f"\nFull results saved to: {out_path}")


if __name__ == "__main__":
    main()