"""Learning Roadmap Agent.

Turns the prioritized skill gaps into a concrete, week-by-week study plan.
Re-invoked automatically by the orchestrator's feedback loop whenever the
Progress Monitoring Agent decides the student's readiness trajectory needs
correction (see graph.py).
"""
from __future__ import annotations

from typing import Dict, List

from ..llm import generate_json
from ..schemas import AgentState, RoadmapItem, SkillGap

SYSTEM_PROMPT = (
    "You are the Learning Roadmap Agent inside CareerPilot AI. Turn the student's "
    "prioritized skill gaps into a week-by-week learning plan covering 8 weeks. "
    'Respond ONLY with a JSON array: [{"week": int, "focus_skill": str, '
    '"activity": str (concrete task), "resource": str (course/platform/project idea)}]. '
    "Spend more weeks on Critical-priority skills."
)

_RESOURCE_BANK = [
    "Self-paced course + official documentation",
    "Hands-on guided project on GitHub",
    "Interactive practice platform (e.g. LeetCode / HackerRank style)",
    "Capstone mini-project combining 2-3 skills",
    "Official certification prep track",
]


def _fallback(gaps: List[SkillGap]) -> List[RoadmapItem]:
    if not gaps:
        gaps = [SkillGap(skill="Portfolio Projects", priority="Important",
                          current_level="Beginner", target_level="Intermediate")]

    weighted: List[str] = []
    for g in gaps:
        weight = 3 if g.get("priority") == "Critical" else (2 if g.get("priority") == "Important" else 1)
        weighted.extend([g["skill"]] * weight)

    plan: List[RoadmapItem] = []
    for week in range(1, 9):
        skill = weighted[(week - 1) % len(weighted)]
        resource = _RESOURCE_BANK[(week - 1) % len(_RESOURCE_BANK)]
        plan.append(RoadmapItem(
            week=week,
            focus_skill=skill,
            activity=f"Build working knowledge of {skill} through a small applied exercise.",
            resource=resource,
        ))
    return plan


def run(state: AgentState) -> Dict:
    gaps = state.get("skill_gaps", [])

    roadmap = generate_json(
        SYSTEM_PROMPT,
        f"Prioritized skill gaps: {gaps}",
        fallback=lambda: _fallback(gaps),
        agent_name="LearningRoadmapAgent",
    )

    if not isinstance(roadmap, list) or not roadmap:
        roadmap = _fallback(gaps)

    log = list(state.get("agent_log", []))
    replanned = state.get("iterations", 0) > 0
    log.append({
        "agent": "Learning Roadmap Agent",
        "summary": (
            f"{'Re-generated' if replanned else 'Generated'} an {len(roadmap)}-week personalized "
            "study plan."
        ),
    })

    return {"roadmap": roadmap, "agent_log": log}
