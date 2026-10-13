import os
import os 
import json
import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from resume_parser import extract_text
from text_cleaner import clean_text
from skill_extractor import load_skills, extract_skills
from job_matcher import score_roles, predict_ml_role
from roadmap_generator import make_roadmap
from ai_chat import is_configured, ask
from report_generator import generate_text_report

load_dotenv()

MAX_MB = 5
ROLE_PLACEHOLDER = "-- Select Target Job Role --"

# Page Configuration
st.set_page_config(
    page_title="AI Resume Analyzer & Job Recommender",
    page_icon="🧭",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-Aesthetic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .block-container {
        padding-top: 1.8rem;
        padding-bottom: 3rem;
        max-width: 1150px;
    }
    
    .header-title {
        background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
        font-size: 2.3rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 1.5rem;
    }
    
    .score-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.25rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1.5rem;
    }
    .score-high { background: linear-gradient(135deg, #10b981 0%, #059669 100%); }
    .score-mid  { background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%); }
    .score-low  { background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%); }
    
    .skill-pill {
        display: inline-block;
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        margin: 0.2rem 0.3rem 0.35rem 0;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .missing-pill {
        display: inline-block;
        background-color: #fef2f2;
        color: #991b1b;
        border: 1px solid #fecaca;
        padding: 0.35rem 0.85rem;
        border-radius: 9999px;
        margin: 0.2rem 0.3rem 0.35rem 0;
        font-size: 0.85rem;
        font-weight: 500;
    }
    
    .ml-card {
        background: linear-gradient(135deg, #f8fafc 0%, #f1f5f9 100%);
        border: 1px solid #cbd5e1;
        border-radius: 12px;
        padding: 1.2rem 1.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    
    .ai-badge {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .ai-active { background-color: #dcfce7; color: #166534; }
    .ai-inactive { background-color: #f3f4f6; color: #4b5563; }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def load_app_data():
    skills = load_skills("data/skill_dictionary.csv")
    roles = pd.read_csv("data/job_roles.csv")
    ml_meta = {}
    if os.path.exists("data/ml_model_metadata.json"):
        with open("data/ml_model_metadata.json", "r") as f:
            ml_meta = json.load(f)
    return skills, roles, ml_meta


def render_pills(items, css_class: str) -> str:
    if not items:
        return "<i style='color:#94a3b8;'>None</i>"
    return "".join(f"<span class='{css_class}'>{s}</span>" for s in items)


def get_score_class(score: float) -> str:
    if score >= 70:
        return "score-high"
    if score >= 40:
        return "score-mid"
    return "score-low"


skills_dict, roles_df, ml_metadata = load_app_data()

# Sidebar Setup
with st.sidebar:
    st.image("https://img.icons8.com/isometric/96/resume.png", width=64)
    st.markdown("### 🧭 Resume Control Center")
    
    ai_ready = is_configured()
    if ai_ready:
        st.markdown("<span class='ai-badge ai-active'>🟢 Gemini AI Advisor Active</span>", unsafe_allow_html=True)
    else:
        st.markdown("<span class='ai-badge ai-inactive'>⚪ Gemini API Key missing</span>", unsafe_allow_html=True)
    
    
    st.markdown("---")
    st.subheader("1. Upload Resume")
    uploaded_file = st.file_uploader(f"Upload PDF or DOCX (Max {MAX_MB}MB)", type=["pdf", "docx"])
    
    st.markdown("**OR Quick-Load Sample Resume:**")
    sample_choice = st.selectbox(
        "Select a pre-built sample resume",
        ["-- None --", "Resume_Data_Analyst.docx", "Resume_ML_Engineer.docx", "Resume_NLP_Engineer.docx"]
    )
    
    st.markdown("---")
    st.subheader("2. Target Job Role")
    target_role = st.selectbox("Select target role", [ROLE_PLACEHOLDER] + list(roles_df["role"]))
    
    #st.markdown("---")
    #st.caption("ℹ️ **Responsible AI Notice**")
    # st.caption("Match scores are automated estimates based on skill coverage, text similarity, and ML classification. Personal information is excluded from scoring.")

# Main Application Banner
st.markdown("<h1 class='header-title'>AI Resume Analyzer & Job Recommendation System</h1>", unsafe_allow_html=True)
#st.markdown("<p class='subtitle'>NLP skill extraction, TF-IDF cosine matching,  and Gemini AI career advisory.</p>", unsafe_allow_html=True)

# File Extraction Logic
file_bytes = None
file_name = None

if uploaded_file is not None:
    file_bytes = uploaded_file.getvalue()
    file_name = uploaded_file.name
elif sample_choice != "-- None --":
    sample_path = os.path.join("sample_resumes", sample_choice)
    if os.path.exists(sample_path):
        with open(sample_path, "rb") as f:
            file_bytes = f.read()
        file_name = sample_choice

if not file_bytes:
    st.info("👈 **Get started by uploading your resume in the sidebar of PDF/DOCX format or choosing a Quick-Load Sample Resume.**")
    st.stop()

if target_role == ROLE_PLACEHOLDER:
    st.warning("👈 **Please select a Target Job Role in the sidebar to complete the matching analysis.**")
    st.stop()

if len(file_bytes) > MAX_MB * 1024 * 1024:
    st.error(f"File size exceeds the maximum limit of {MAX_MB}MB.")
    st.stop()

# Parse & Process
try:
    raw_text = extract_text(file_bytes, file_name)
    cleaned_resume = clean_text(raw_text)
except Exception as e:
    st.error(f"Error parsing resume file: {e}")
    st.stop()

if len(cleaned_resume) < 40:
    st.error("Insufficient text extracted. Scanned image-only PDFs or empty files are not supported.")
    st.stop()

extracted_skills = extract_skills(cleaned_resume, skills_dict)
match_results = score_roles(cleaned_resume, extracted_skills, roles_df)
ml_prediction = predict_ml_role(cleaned_resume)

target_row = match_results[match_results["Role"] == target_role].iloc[0]
roadmap_steps = make_roadmap(target_row["Missing"], skills_dict)

# Header Metrics
st.success(f"📄 Active Resume: **{file_name}**")

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Target Match Score", f"{target_row['Score']}%")
m2.metric("Skills Detected", len(extracted_skills))
m3.metric("Missing Role Skills", len(target_row["Missing"]))
top_recommendation = match_results.iloc[0]["Role"]
m4.metric("Top Hybrid Role", top_recommendation)
if ml_prediction.get("available"):
    m5.metric("ML Classifier Role", ml_prediction["top_role"], delta=f"{ml_prediction['confidence']}% conf")

st.markdown(
    f"<div class='score-badge {get_score_class(target_row['Score'])}'>"
    f"Target Role: {target_role} — {target_row['Score']}% Match</div>",
    unsafe_allow_html=True
)

# Tabs
tab_overview, tab_skills, tab_roadmap, tab_chat = st.tabs([
    "📊 Role Matching & ML Classification",
    "🧩 Skill Breakdown & Gaps",
    "🗺️ Learning Roadmap & Report",
    "💬 Gemini AI Advisor"
])

with tab_overview:
    # Supervised ML Classification Section
    if ml_prediction.get("available"):
        st.markdown("<div class='ml-card'>", unsafe_allow_html=True)
        st.markdown(f"### Machine Learning Classifier Prediction")
        st.markdown(
            f" the ML model predicts this candidate is best suited for: "
            f"**<span style='color:#1d4ed8;'>{ml_prediction['top_role']}</span>** "
            f"with **{ml_prediction['confidence']}% confidence**.",
            unsafe_allow_html=True
        )
        
        # Probabilities bar chart
        prob_df = ml_prediction["probabilities"].head(6)
        fig_ml = px.bar(
            prob_df,
            x="Probability",
            y="Role",
            orientation="h",
            range_x=[0, 100],
            text="Probability",
            color="Probability",
            color_continuous_scale=["#cbd5e1", "#3b82f6", "#1d4ed8"],
            labels={"Probability": "Probability (%)", "Role": "Predicted Class"},
        )
        fig_ml.update_layout(
            showlegend=False,
            coloraxis_showscale=False,
            height=260,
            margin=dict(l=20, r=20, t=10, b=10),
            xaxis_title="Probability (%)",
            yaxis_title=""
        )
        fig_ml.update_traces(texttemplate='%{text}%', textposition='outside')
        st.plotly_chart(fig_ml, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
    st.subheader("Hybrid Role Match Score Comparison (Skill Coverage 70% + TF-IDF Cosine 30%)")
    chart_df = match_results.sort_values("Score", ascending=True)
    
    fig = px.bar(
        chart_df,
        x="Score",
        y="Role",
        orientation="h",
        range_x=[0, 100],
        text="Score",
        color="Score",
        color_continuous_scale=["#ef4444", "#f59e0b", "#10b981"],
        labels={"Score": "Match Percentage (%)", "Role": "Job Role"},
    )
    fig.update_layout(
        showlegend=False,
        coloraxis_showscale=False,
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis_title="Match Score (%)",
        yaxis_title=""
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Top 3 Recommended Roles")
    top_3 = match_results.head(3)
    cols = st.columns(3)
    for idx, (c, (_, r)) in enumerate(zip(cols, top_3.iterrows()), 1):
        with c:
            st.markdown(f"### {idx}. {r['Role']}")
            st.markdown(f"**Score:** `{r['Score']}%` | **Skill Coverage:** `{r['Coverage']}%`")
            st.progress(min(int(r['Score']), 100))
            st.caption(r['Description'])

with tab_skills:
    st.subheader("Detected Resume Skills (Categorized)")
    by_category = {}
    for skill_name, cat in extracted_skills.items():
        by_category.setdefault(cat, []).append(skill_name)
    
    if by_category:
        for category, s_list in sorted(by_category.items()):
            st.markdown(f"**{category}** ({len(s_list)})")
            st.markdown(render_pills(s_list, "skill-pill"), unsafe_allow_html=True)
    else:
        st.info("No predefined dictionary skills were matched in the resume text.")
    
    st.markdown("---")
    st.subheader(f"Skill Gap Analysis for '{target_role}'")
    col_matched, col_missing = st.columns(2)
    
    with col_matched:
        st.markdown("#### ✅ Skills Matched")
        st.markdown(render_pills(target_row["Matched"], "skill-pill"), unsafe_allow_html=True)
        
    with col_missing:
        st.markdown("#### ❌ Skills Missing or Weak")
        st.markdown(render_pills(target_row["Missing"], "missing-pill"), unsafe_allow_html=True)

with tab_roadmap:
    st.subheader(f"Personalized Learning Roadmap for '{target_role}'")
    for step in roadmap_steps:
        st.markdown(f"- {step}")
        
    st.markdown("---")
    st.subheader("⬇️ Download Full Analysis Report")
    report_content = generate_text_report(
        target_role=target_role,
        score=target_row["Score"],
        matched=target_row["Matched"],
        missing=target_row["Missing"],
        top_roles_df=match_results,
        roadmap=roadmap_steps
    )
    
    st.download_button(
        label="📥 Download Analysis Report (.txt)",
        data=report_content,
        file_name=f"resume_analysis_{target_role.lower().replace(' ', '_')}.txt",
        mime="text/plain"
    )

with tab_chat:
    st.subheader("💬 Gemini AI Career Advisor")
    st.caption("Ask questions about your match score, missing skills, interview preparation, or resume improvement.")
    
    if not is_configured():
        st.info(
            "🔑 **Gemini API Key is not set in `.env`.**\n\n"
            "To enable the AI Chatbot feature:\n"
            "1. Get a free API key from [Google AI Studio](https://aistudio.google.com/app/apikey).\n"
            "2. Add your key to your `.env` file:\n"
            "```env\n"
            "GEMINI_API_KEY=your_actual_api_key_here\n"
            "```\n"
            "3. Restart the Streamlit application."
        )
    else:
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []
            
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])
                
        user_input = st.chat_input("e.g. How can I quickly learn FastAPI? What project should I build?")
        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            with st.chat_message("user"):
                st.write(user_input)
                
            with st.chat_message("assistant"):
                with st.spinner("Gemini is generating response..."):
                    try:
                        ai_response = ask(
                            question=user_input,
                            target_role=target_role,
                            score=target_row["Score"],
                            matched=target_row["Matched"],
                            missing=target_row["Missing"]
                        )
                    except Exception as err:
                        ai_response = f"⚠️ {err}"
                st.write(ai_response)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})