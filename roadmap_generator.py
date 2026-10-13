def make_roadmap(missing_skills: list, skills_dict: dict) -> list:
    """
    Generate a week-by-week actionable learning roadmap based on missing skills.
    Returns a list of structured roadmap steps.
    """
    if not missing_skills:
        return [
            "🎉 Excellent! You have matched all key required skills for this target role.",
            "Suggested Next Step: Focus on advanced projects, portfolio optimization, and system architecture design."
        ]

    roadmap = []
    for i, skill in enumerate(missing_skills, 1):
        topic = skills_dict.get(skill, {}).get("topic", f"Fundamentals of {skill}")
        category = skills_dict.get(skill, {}).get("category", "Technical Skill")
        step = f"Week {i} [{category}]: Study {skill} — Focus topic: '{topic}'. Build a mini-project or complete hands-on exercises."
        roadmap.append(step)

    roadmap.append("Week " + str(len(missing_skills) + 1) + " [Integration]: Build an end-to-end portfolio project combining your new and existing skills.")
    return roadmap
