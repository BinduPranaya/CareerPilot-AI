"""Skill Gap Agent.

Compares the student's current skills against the RAG-retrieved
expectations for their primary target role, and prioritizes what is
missing. Downstream, the Learning Roadmap Agent consumes this list
directly.
"""
from __future__ import annotations

import re
from typing import Dict, List

from ..llm import generate_json
from ..schemas import AgentState, SkillGap
from ..vectorstore import query_role_requirements

SYSTEM_PROMPT = (
    "You are the Skill Gap Agent inside CareerPilot AI. Compare the student's current "
    "skills against the role requirement text retrieved for their target role. Identify "
    "the 4-6 most important missing or weak skills. Respond ONLY with a JSON array: "
    '[{"skill": str, "priority": "Critical"|"Important"|"Nice to have", '
    '"current_level": "None"|"Beginner"|"Intermediate", "target_level": "Intermediate"|"Advanced"}]. '
    "Order by priority, Critical first."
)


def _extract_required_skills(requirement_text: str) -> List[str]:
    # the seed corpus lists skills as a comma-separated clause after "expects"
    match = re.search(r"expects (.+?)\.\s*Market demand", requirement_text)
    clause = match.group(1) if match else requirement_text
    parts = re.split(r",| and ", clause)
    return [p.strip().rstrip(".") for p in parts if p.strip()]


def _fallback(skills: List[str], role: str) -> List[SkillGap]:
    reqs = query_role_requirements(role, n_results=1)
    req_text = reqs[0] if reqs else ""
    required = _extract_required_skills(req_text) or ["System Design", "Communication", "Testing"]
    have_lower = {s.lower() for s in skills}

    gaps: List[SkillGap] = []
    for i, req_skill in enumerate(required):
        if any(req_skill.lower() in h or h in req_skill.lower() for h in have_lower):
            continue
        priority = "Critical" if i < 2 else ("Important" if i < 4 else "Nice to have")
        gaps.append(SkillGap(
            skill=req_skill.title(),
            priority=priority,
            current_level="None",
            target_level="Intermediate" if priority != "Critical" else "Advanced",
        ))
    return gaps[:6] or [SkillGap(skill="Portfolio Projects", priority="Important",
                                  current_level="Beginner", target_level="Intermediate")]


def run(state: AgentState) -> Dict:
    profile = state.get("profile", {})
    skills = profile.get("skills", [])
    role = state.get("primary_target_role", "AI Engineer")

    reqs = query_role_requirements(role, n_results=2)
    grounding = "\n".join(reqs)

    user_prompt = (
        f"Target role: {role}\n"
        f"Role requirement grounding (retrieved via RAG):\n{grounding}\n\n"
        f"Student's current skills: {skills}"
    )

    gaps = generate_json(
        SYSTEM_PROMPT,
        user_prompt,
        fallback=lambda: _fallback(skills, role),
        agent_name="SkillGapAgent",
    )

    if not isinstance(gaps, list) or not gaps:
        gaps = _fallback(skills, role)

    log = list(state.get("agent_log", []))
    critical = sum(1 for g in gaps if g.get("priority") == "Critical")
    log.append({
        "agent": "Skill Gap Agent",
        "summary": f"Identified {len(gaps)} skill gaps for {role} ({critical} critical).",
    })

    return {"skill_gaps": gaps, "agent_log": log}
