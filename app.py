
import os
from html import escape

import pandas as pd
import plotly.express as px
import streamlit as st
from dotenv import load_dotenv

from skill_extractor import load_skills
from pipeline import analyze_resume
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

# Styling (works in light and dark theme)
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

    .stat-card {
        background: rgba(148, 163, 184, 0.12);
        border: 1px solid rgba(148, 163, 184, 0.30);
        border-radius: 12px;
        padding: 0.9rem 1.1rem;
        height: 100%;
    }
    .stat-label { font-size: 0.8rem; color: #94a3b8; font-weight: 500; margin-bottom: 0.2rem; }
    .stat-value { font-size: 1.35rem; font-weight: 700; line-height: 1.25; word-break: normal; overflow-wrap: anywhere; }
    .stat-sub   { font-size: 0.8rem; color: #94a3b8; margin-top: 0.15rem; }

    .score-badge {
        display: inline-flex;
        align-items: center;
        padding: 0.5rem 1.25rem;
        border-radius: 9999px;
        font-weight: 700;
        font-size: 1.1rem;
        color: white;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin: 1rem 0 1.2rem 0;
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
    return skills, roles


def render_pills(items, css_class: str) -> str:
    if not items:
        return "<i style='color:#94a3b8;'>None</i>"
    return "".join(f"<span class='{css_class}'>{escape(str(s))}</span>" for s in items)


def stat_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f"<div class='stat-sub'>{escape(sub)}</div>" if sub else ""
    return (f"<div class='stat-card'><div class='stat-label'>{escape(label)}</div>"
            f"<div class='stat-value'>{escape(str(value))}</div>{sub_html}</div>")


def get_score_class(score: float) -> str:
    if score >= 70:
        return "score-high"
    if score >= 40:
        return "score-mid"
    return "score-low"


skills_dict, roles_df = load_app_data()

# Sidebar
with st.sidebar:
    #st.image("https://img.icons8.com/isometric/96/resume.png", width=64)
    st.markdown("### Resume Control Center")

    ai_ready = is_configured()
    if ai_ready:
        st.markdown("<span class='ai-badge ai-active'>🟢  AI model is Active</span>", unsafe_allow_html=True)
        st.caption("Only cleaned resume text (no email, phone or links) is sent to AI.")
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

# Banner
st.markdown("<h1 class='header-title'>AI Resume Analyzer & Job Recommendation System</h1>", unsafe_allow_html=True)

# File input
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


# Analysis (cached: Streamlit reruns this script on every click, and we must not call Gemini each time)
@st.cache_data(show_spinner=False, ttl=3600, max_entries=20)
def run_analysis(file_bytes, file_name, target_role, use_gemini):
    return analyze_resume(file_bytes, file_name, target_role, skills_dict, roles_df, use_gemini)


try:
    with st.spinner("Analyzing resume (the first call can take 10-20 seconds)..."):
        result = run_analysis(file_bytes, file_name, target_role, ai_ready)
except Exception as e:
    st.error(f"Analysis error: {e}")
    st.stop()

extracted_skills = result["skills_detected"]
match_results = pd.DataFrame(result["role_scores"])
gemini = result.get("gemini")
roadmap_steps = result["roadmap"]
target_row = match_results[match_results["Role"] == target_role].iloc[0]

if ai_ready and result.get("gemini_error"):
    st.warning(f"Gemini analysis unavailable, showing keyword-based results only. ({result['gemini_error']})")
    run_analysis.clear()  # do not cache a failed Gemini call, so the next rerun retries it

# Header
st.success(f"📄 Active Resume: **{file_name}**")

if gemini:
    best_role, best_sub = gemini["predicted_role"], f"{gemini['confidence']}% confidence"
    level = gemini["experience_level"]
else:
    best_role, best_sub = match_results.iloc[0]["Role"], "keyword-based"
    level = None

# Tabs
tab_overview, tab_skills, tab_roadmap, tab_chat = st.tabs([
    "📋 Overview",
    "🧩 Skills & Gaps",
    "🗺️ Roadmap & Report",
    "💬 AI Advisor"
])

# ---------------- Overview---------------
with tab_overview:
    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(stat_card("Target Match Score", f"{target_row['Score']}%", target_role), unsafe_allow_html=True)
    c2.markdown(stat_card("Best-Fit Role", best_role, best_sub), unsafe_allow_html=True)
    if level:
        c3.markdown(stat_card("Experience Level", level), unsafe_allow_html=True)
    else:
        c3.markdown(stat_card("Missing Role Skills", str(len(target_row["Missing"])), target_role), unsafe_allow_html=True)
    c4.markdown(stat_card("Skills Detected", str(len(extracted_skills)), "from dictionary"), unsafe_allow_html=True)

    st.markdown(
        f"<div class='score-badge {get_score_class(target_row['Score'])}'>"
        f"Target Role: {escape(target_role)} — {target_row['Score']}% Match</div>",
        unsafe_allow_html=True
    )

    if gemini:
        st.info(gemini["summary"])

    st.subheader("Top 3 Recommended Roles")
    top_3 = match_results.head(3)
    cols = st.columns(3)
    for idx, (c, (_, r)) in enumerate(zip(cols, top_3.iterrows()), 1):
        with c:
            st.markdown(f"### {idx}. {r['Role']}")
            st.markdown(f"**Match:** `{r['Score']}%` | **Skill Coverage:** `{r['Coverage']}%`")
            st.progress(min(int(r['Score']), 100))
            st.caption(r['Description'])

    st.markdown("---")
    st.subheader("Match Score for All Roles")
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
        height=420,
        margin=dict(l=20, r=20, t=10, b=20),
        xaxis_title="Match Score (%)",
        yaxis_title=""
    )
    fig.update_traces(texttemplate='%{text}%', textposition='outside')
    st.plotly_chart(fig, use_container_width="stretch")
    st.caption("AI estimates for learning guidance only, not hiring decisions. "
               "Missing keywords do not always mean missing ability.")

# ---------------- Skills & Gaps ----------------
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

    if gemini and gemini["extra_skills"]:
        st.markdown("**Other skills Gemini found in the resume**")
        st.markdown(render_pills(gemini["extra_skills"], "skill-pill"), unsafe_allow_html=True)

    st.markdown("---")
    st.subheader(f"Skill Gap Analysis for '{target_role}'")
    col_matched, col_missing = st.columns(2)

    with col_matched:
        st.markdown("#### ✅ Skills Matched")
        st.markdown(render_pills(target_row["Matched"], "skill-pill"), unsafe_allow_html=True)

    with col_missing:
        st.markdown("#### ❌ Skills Missing or Weak")
        st.markdown(render_pills(target_row["Missing"], "missing-pill"), unsafe_allow_html=True)

with tab_skills:
    st.markdown("---")
    if gemini:
        col_s, col_w = st.columns(2)
        with col_s:
            st.markdown("#### 💪 Strengths")
            for s in gemini["strengths"]:
                st.markdown(f"- {s}")
        with col_w:
            st.markdown("#### ⚠️ Weaknesses")
            for s in gemini["weaknesses"]:
                st.markdown(f"- {s}")

        st.markdown("#### 🛠️ How to improve this resume")
        for s in gemini["resume_improvements"]:
            st.markdown(f"- {s}")
    else:
        st.info("Add GEMINI_API_KEY to your .env file to see strengths, weaknesses and resume improvement tips.")

# ---------------- Roadmap & Report ----------------
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
        file_name=f"resume_analysis_{target_role.lower().replace(' ', '_').replace('/', '')}.txt",
        mime="text/plain"
    )

# ---------------- AI Advisor ----------------
with tab_chat:
    st.subheader("💬 Gemini AI Career Advisor")
    st.caption("Ask questions about your match score, missing skills, interview preparation, or resume improvement.")

    if not ai_ready:
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
                            missing=target_row["Missing"],
                            history=st.session_state.chat_history[:-1][-6:]
                        )
                    except Exception as err:
                        ai_response = f"⚠️ {err}"
                st.write(ai_response)
            st.session_state.chat_history.append({"role": "assistant", "content": ai_response})