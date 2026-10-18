"""Profile Analysis Agent.

Parses the student's resume / academic record text and produces a
structured StudentProfile. This profile anchors everything downstream: it
is the first thing embedded into long-term vector memory so every later
agent can recall it.
"""
from __future__ import annotations

import re
from typing import Dict, List

from ..llm import generate_json
from ..schemas import AgentState, StudentProfile
from ..vectorstore import upsert_student_memory

SYSTEM_PROMPT = (
    "You are the Profile Analysis Agent inside CareerPilot AI, a multi-agent career "
    "mentorship system. Read the student's resume/academic text and extract a clean "
    "structured profile. Respond ONLY with JSON matching this shape: "
    '{"name": str, "education": str, "skills": [str], "interests": [str], '
    '"experience_summary": str}. Keep skills to concrete, resume-style tokens '
    '(e.g. "Python", "React", "SQL"), 4-10 items.'
)

_KNOWN_SKILLS = [
    "python", "java", "javascript", "typescript", "react", "node.js", "node",
    "spring boot", "sql", "mysql", "postgresql", "mongodb", "aws", "azure",
    "gcp", "docker", "kubernetes", "git", "html", "css", "tailwind",
    "machine learning", "deep learning", "pytorch", "tensorflow", "pandas",
    "numpy", "data analysis", "rest api", "graphql", "linux", "c++", "c",
    "android", "flutter", "django", "flask", "excel", "power bi", "tableau",
    "cybersecurity", "networking", "terraform", "ci/cd",
]


def _fallback(resume_text: str) -> StudentProfile:
    text_lower = resume_text.lower()
    found_skills = [s.title() if s.islower() else s for s in _KNOWN_SKILLS if s in text_lower]
    found_skills = [s if s not in ("Sql", "Aws", "Gcp", "Ci/Cd") else s.upper() for s in found_skills]

    name_match = re.search(r"name[:\-]\s*([^\n.,]+)", resume_text, re.IGNORECASE)
    name = name_match.group(1).strip() if name_match else "Student"

    edu_match = re.search(
        r"(b\.?tech|bachelor|m\.?tech|master|b\.?sc|m\.?sc|computer science|information technology)[^\n]{0,80}",
        resume_text, re.IGNORECASE,
    )
    education = edu_match.group(0).split(". ")[0].strip() if edu_match else "Undergraduate, Computer Science (assumed)"

    return StudentProfile(
        name=name,
        education=education,
        skills=found_skills[:10] or ["Python", "SQL", "Communication"],
        interests=["Software Development", "Artificial Intelligence"],
        experience_summary=(
            "Resume parsed in offline mode: summary generated from keyword extraction "
            "rather than full LLM comprehension."
        ),
    )


def run(state: AgentState) -> Dict:
    resume_text = state.get("resume_text", "") or "No resume text provided."

    profile = generate_json(
        SYSTEM_PROMPT,
        f"Resume / academic record:\n\n{resume_text}",
        fallback=lambda: _fallback(resume_text),
        agent_name="ProfileAnalysisAgent",
    )

    # normalize shape defensively in case the model omits a field
    profile = {
        "name": profile.get("name", "Student"),
        "education": profile.get("education", "Not specified"),
        "skills": profile.get("skills", []) or [],
        "interests": profile.get("interests", []) or [],
        "experience_summary": profile.get("experience_summary", ""),
    }

    upsert_student_memory(
        state["student_id"],
        summary_text=(
            f"Name: {profile['name']}. Education: {profile['education']}. "
            f"Skills: {', '.join(profile['skills'])}. Interests: {', '.join(profile['interests'])}."
        ),
        metadata={"stage": "profile_analysis"},
    )

    log = list(state.get("agent_log", []))
    log.append({
        "agent": "Profile Analysis Agent",
        "summary": f"Extracted {len(profile['skills'])} skills and built the base student profile.",
    })

    return {"profile": profile, "agent_log": log}
