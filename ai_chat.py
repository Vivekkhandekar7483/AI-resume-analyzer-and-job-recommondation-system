"""
ai_chat.py - AI Career Advisor Chatbot using Google's Gemini API.
Integrates securely via .env (checks GEMINI_API_KEY or GOOGLE_API_KEY).
Only processed summary context (score, target role, skills) is sent to the LLM.
"""
import os
from google import genai
from google.genai import types

PREFERRED_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.0-flash",
    "gemini-1.5-flash",
    "gemini-flash-latest",
]

SYSTEM_INSTRUCTION = (
    "You are a friendly, encouraging, and highly knowledgeable AI Career Mentor for students. "
    "Your goal is to help students analyze their resume match scores, understand missing skills, "
    "and give practical advice for interviewing, project building, and career growth. "
    "Keep your answers concise, practical, and structured with clear markdown formatting. "
    "Do not invent skills that were not detected in the resume context."
)


def get_api_key() -> str:
    """Retrieve Gemini API key from environment variables."""
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    return str(key).strip()


def is_configured() -> bool:
    """Check whether a Gemini API key is configured."""
    return bool(get_api_key())


def ask(question: str, target_role: str, score: float, matched: list, missing: list, history: list = None) -> str:
    """
    Send analysis context and student question to Gemini API.
    Returns response string. Raises RuntimeError with user-friendly error on failure.
    """
    api_key = get_api_key()
    if not api_key:
        raise RuntimeError("No Gemini API key found. Please add GEMINI_API_KEY to your .env file.")

    context = (
        f"--- CANDIDATE RESUME ANALYSIS CONTEXT ---\n"
        f"Target Role: {target_role}\n"
        f"Match Score: {score}%\n"
        f"Matched Skills: {', '.join(matched) if matched else 'None'}\n"
        f"Missing Skills: {', '.join(missing) if missing else 'None'}\n"
        f"-----------------------------------------\n\n"
        f"Student Question: {question}"
    )

    client = genai.Client(api_key=api_key)

    last_error = None
    for model_name in PREFERRED_MODELS:
        try:
            response = client.models.generate_content(
                model=model_name,
                contents=context,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=0.7,
                ),
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            last_error = e
            continue

    raise RuntimeError(f"Could not reach Gemini API ({type(last_error).__name__} if any). Please check your GEMINI_API_KEY in .env.")
