# 🧭 AI Resume Analyzer & Job Recommendation System

An NLP-based educational application that evaluates student resumes against target job roles, computes a hybrid match score, identifies missing skills, generates a week-by-week learning roadmap, exports analysis reports, and includes an interactive **Gemini AI Career Advisor Chatbot**.

---

## 🌟 Key Features

1. **Multi-Format Resume Upload**: Supports PDF (`.pdf`) and Word (`.docx`) resumes parsed completely in memory without permanent file storage.
2. **Text Cleaning & Symbol Preservation**: Normalizes unstructured text while preserving critical technical symbols (`C++`, `C#`, `.NET`, `HTML/CSS`, `Next.js`). Strips protected personal information (gender, age, marital status) to eliminate bias.
3. **Controlled Skill Dictionary**: 50+ technical skills organized into categories (*Programming, ML, Data, Databases, Backend, Web, Cloud & DevOps, Tools, Computer Vision, CS Fundamentals*) with alias mapping.
4. **Hybrid Role Matching**: Combines **Skill Coverage (70%)** and **TF-IDF Cosine Similarity (30%)** to rank job roles accurately.
5. **Skill-Gap Analysis & Roadmap**: Automatically identifies matched vs. missing skills and builds a personalized week-by-week learning roadmap.
6. **Gemini AI Career Advisor**: Built-in chatbot powered by Google Gemini API (`GEMINI_API_KEY`) for interactive Q&A, interview advice, and resume optimization tips.
7. **Downloadable Analysis Reports**: Export comprehensive text (`.txt`) match reports.
8. **Interactive Streamlit Dashboard**: Clean visual UI with custom CSS, Plotly match score charts, quick sample resume loaders, and interactive tabs.

---

## 📁 Suggested Folder Structure

```
ai_resume_analyzer/
├── app.py                  # Streamlit Dashboard & Web Interface
├── resume_parser.py        # PDF & DOCX text extraction
├── text_cleaner.py         # Text cleaning, normalization & PII stripping
├── skill_extractor.py      # Regex alias-aware skill extraction
├── job_matcher.py          # TF-IDF + Skill Coverage hybrid matching engine
├── roadmap_generator.py    # Learning roadmap generator
├── report_generator.py     # Text report exporter
├── ai_chat.py              # Gemini API AI Chatbot integration
├── batch_test.py           # Bulk evaluation CLI tool
├── requirements.txt        # Python package dependencies
├── .env.example            # Environment variables template
├── .gitignore              # Git ignore rules
├── README.md               # Project documentation
│
├── data/
│   ├── job_roles.csv       # Job role dataset & skill requirements
│   └── skill_dictionary.csv# Skill dictionary & category/alias mappings
│
├── sample_resumes/         # Pre-built sample resumes for testing
│   ├── Resume_Data_Analyst.docx
│   ├── Resume_ML_Engineer.docx
│   └── Resume_NLP_Engineer.docx
│
├── reports/                # Output analysis reports & batch test logs
└── tests/                  # Automated unit test suite
    ├── test_cases.csv      # Evaluation benchmark matrix
    └── test_pipeline.py    # Unittest pipeline tests
```

---

## 🛠️ Installation & Setup

### 1. Clone or Open the Repository
```bash
cd ai_resume_analyzer
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Gemini API Key (Optional for Chatbot)
Copy `.env.example` to `.env` and paste your Gemini API Key:
```bash
cp .env.example .env
```
Inside `.env`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```
*(Get a free key from [Google AI Studio](https://aistudio.google.com/app/apikey))*

---

## 🚀 How to Run

### Run the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Run Automated Unit Tests
```bash
python -m unittest discover tests
```

### Run Batch Test on Folder of Resumes
```bash
python batch_test.py sample_resumes
```

---

## 📊 Minimum Features Checklist

- [x] Upload PDF or DOCX resume
- [x] Extract and clean unstructured text
- [x] Identify 50+ technical skills
- [x] Compare resume against 8 target job roles
- [x] Calculate hybrid match score (Skill Coverage + TF-IDF)
- [x] Recommend top 3 suitable roles
- [x] Identify matched vs. missing skills
- [x] Generate week-by-week learning roadmap
- [x] Downloadable analysis report
- [x] Gemini AI Chatbot integration
- [x] Streamlit web dashboard
- [x] Automated test suite & documentation

---

## ⚖️ Responsible AI Guidelines

- **Guidance Only**: Match scores are estimates for educational guidance, not automated recruiter decisions.
- **Fairness & Privacy**: Protected personal information (gender, age, marital status, religion) is stripped and ignored during evaluation.
- **Data Protection**: Resumes are processed in memory and never stored permanently on external servers.
