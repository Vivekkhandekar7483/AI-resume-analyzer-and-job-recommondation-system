from datetime import datetime
import pandas as pd


def generate_text_report(target_role: str, score: float, matched: list, missing: list,
                         top_roles_df: pd.DataFrame, roadmap: list) -> str:
    """Generate a clean, structured text report for downloading."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "============================================================",
        "          AI RESUME ANALYZER & CAREER MATCH REPORT          ",
        "============================================================",
        f"Generated At: {timestamp}",
        f"Target Role: {target_role}",
        f"Resume Match Score: {score}%",
        "------------------------------------------------------------",
        "",
        "[SKILLS FOUND IN RESUME]",
        *([f"  • {s}" for s in matched] if matched else ["  • None detected"]),
        "",
        f"[MISSING SKILLS FOR {target_role.upper()}]",
        *([f"  • {s}" for s in missing] if missing else ["  • None (Full Match)"]),
        "",
        "[TOP RECOMMENDED ROLES]",
    ]

    if not top_roles_df.empty:
        for i, (_, row) in enumerate(top_roles_df.head(3).iterrows(), 1):
            role_name = row.get("Role", row.get("role", "Unknown Role"))
            role_score = row.get("Score", row.get("score", "N/A"))
            coverage = row.get("Coverage", row.get("coverage", "N/A"))
            lines.append(f"  {i}. {role_name} - {role_score}% Match (Coverage: {coverage}%)")

    lines.extend([
        "",
        "[RECOMMENDED LEARNING ROADMAP]",
        *([f"  {item}" for item in roadmap]),
        "",
        "------------------------------------------------------------",
        "RESPONSIBLE AI NOTICE:",
        "Match scores are automated estimates based on skill keyword and textual similarity.",
        "They are designed for learning and self-improvement guidance, not hiring decisions.",
        "============================================================",
    ])

    return "\n".join(lines)
