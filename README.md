# 🧭 AI Resume Analyzer & Job Recommendation System

An NLP + Gemini-powered application that helps students understand how well their resume matches a target job role. It extracts skills from a PDF/DOCX resume, scores the resume against job roles, lists missing skills, builds a week-by-week learning roadmap, and includes an AI career advisor chatbot.

> Match scores are estimates for learning guidance, **not** hiring decisions.

---

## ✨ Features

- **PDF & DOCX upload** – parsed in memory, nothing is stored on disk.
- **Text cleaning** – lowercases, removes emails, phone numbers and links, keeps `C++`, `C#`, `.NET`; protected details (gender, age, religion, nationality, marital status) are stripped so they can never influence scoring.
- **Skill extraction** – controlled dictionary of ~70 skills in categories (Programming, ML, Data, Databases, Backend, Web, Cloud & DevOps, Tools, Testing, Business Analysis, …) with alias matching.
- **Hybrid keyword scoring** – 70% skill coverage + 30% TF-IDF cosine similarity against each job role in `data/job_roles.csv`.
- **Gemini deep analysis** – reads the resume like a recruiter and returns, as validated structured JSON: best-fit role, experience level, a summary, strengths, weaknesses, resume-improvement tips, and extra skills. Skills not present in the resume text are discarded to prevent hallucinations.
- **Blended final score** – `40% keyword score + 60% Gemini fit score` (falls back to keyword-only if Gemini is unavailable).
- **Top 3 recommended roles** with match % and a chart of all roles.
- **Skill-gap analysis and learning roadmap**, downloadable text report.
- **Gemini career advisor chatbot** with conversation memory.
- **Optional FastAPI backend** (`api.py`) exposing the same pipeline as a REST API.

## 🖥️ App layout

| Tab | Contents |
|---|---|
| 📋 Overview | Match score cards, Gemini summary, top 3 recommended roles with %, chart of all roles |
| 🧩 Skills & Gaps | Detected skills, matched/missing skills, strengths, weaknesses, how to improve the resume |
| 🗺️ Roadmap & Report | Week-by-week learning plan and report download |
| 💬 AI Advisor | Chat with the Gemini career mentor |

## 🔄 How it works

```
Upload resume → extract text → clean text → keyword skill extraction
      → hybrid role scoring ──┐
      → Gemini analysis ──────┴→ blended score → top roles, gaps, roadmap, report
```

## 📁 Project structure

```
ai_resume_analyzer/
├── app.py                  # Streamlit dashboard
├── pipeline.py             # Runs the full analysis (used by app and API)
├── gemini_analyzer.py      # Structured Gemini resume analysis + validation
├── ai_chat.py              # Gemini career-advisor chatbot
├── api.py                  # Optional FastAPI backend
├── resume_parser.py        # PDF / DOCX text extraction
├── text_cleaner.py         # Cleaning and protected-info removal
├── skill_extractor.py      # Alias-aware skill extraction
├── job_matcher.py          # Skill coverage + TF-IDF scoring
├── roadmap_generator.py    # Learning roadmap
├── report_generator.py     # Text report
├── relabel_with_gemini.py  # (optional) label a resume dataset with Gemini
├── train_model.py          # (optional) train/evaluate a role classifier
├── batch_test.py           # Run the pipeline over a folder of resumes
├── requirements.txt
├── .env.example            # Template for your API key
├── data/
│   ├── job_roles.csv
│   ├── skill_dictionary.csv
│   └── Resumes/            # (not in the repo) your private resume dataset
├── sample_resumes/         # Anonymised sample resumes
└── reports/                # Generated output (not in the repo)
```

## 🛠️ Setup

**Requirements:** Python 3.10+ and a free Gemini API key.

```bash
git clone https://github.com/<your-username>/ai_resume_analyzer.git
cd ai_resume_analyzer

python -m venv .venv
# Windows:      .venv\Scripts\activate
# macOS/Linux:  source .venv/bin/activate

pip install -r requirements.txt
```

### Get your own Gemini API key

1. Open [Google AI Studio](https://aistudio.google.com/app/apikey) and create an API key.
2. Copy `.env.example` to `.env` (Windows: `copy .env.example .env`).
3. Put your key in `.env`:

```env
GEMINI_API_KEY=your_key_here
```

The `.env` file is private and is never committed. Without a key the app still works using keyword scoring only; the Gemini analysis, strengths/weaknesses and chatbot need a key.

## 🚀 Run

```bash
streamlit run app.py
```

Open http://localhost:8501, upload a resume (or pick a sample), choose a target role.

### Optional: FastAPI backend

```bash
uvicorn api:app --reload --port 8000
```

Interactive docs at http://localhost:8000/docs. Endpoints: `GET /health`, `GET /roles`, `POST /analyze`, `POST /chat`.

### Optional: batch test and classifier training

These need a folder of resumes at `data/Resumes/` (not included in this repository for privacy).

```bash
python batch_test.py data/Resumes        # run the pipeline over many resumes
python relabel_with_gemini.py            # label resumes with roles from job_roles.csv
python train_model.py                    # train and evaluate a role classifier
```

`train_model.py` reports the majority-class baseline, accuracy, macro-F1, per-class precision/recall and a confusion matrix. The classifier is optional and is not used by the dashboard.

## ⚖️ Responsible AI

- Scores are estimates for guidance, not automated hiring or rejection.
- Protected personal information is removed before scoring and before text is sent to Gemini.
- Only **cleaned** resume text (no emails, phone numbers or links) is sent to the Gemini API.
- Resumes are processed in memory and not stored by the app.
- Missing keywords do not always mean missing ability.

## ⚠️ Limitations

- Keyword matching can miss skills described in different words; Gemini helps but can still be wrong.
- Scanned image-only PDFs are not supported (no OCR).
- Gemini scores can vary slightly between runs.
- The sample dataset is dominated by Java, Project Manager and Business Analyst resumes, so a classifier trained on it is weak for ML/NLP/CV roles.

## 🔑 API key & cost

Every person who runs the project locally uses **their own** Gemini API key. Your key is never part of the repository. Check [Google's current pricing](https://ai.google.dev/pricing) for free-tier limits.

## 🧰 Tech stack

Python · Streamlit · Plotly · pandas · scikit-learn · pypdf · python-docx · Google Gemini API (`google-genai`) · FastAPI · Pydantic

## 📄 License

Add a license of your choice (for example MIT) before publishing.