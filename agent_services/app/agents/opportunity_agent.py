"""Internship & Opportunity Agent.

Surfaces internships, hackathons, competitions, and scholarships matched to
the student's target role and current skill gaps. In production this would
poll live job boards / university ERP feeds (see Scalability & Future
Scope) -- here it generates realistic, clearly-labeled sample opportunities
so the rest of the pipeline and UI can be demoed end-to-end.
"""
from __future__ import annotations

from typing import Dict, List

from ..llm import generate_json
from ..schemas import AgentState, Opportunity

SYSTEM_PROMPT = (
    "You are the Internship & Opportunity Agent inside CareerPilot AI. Suggest 4 "
    "realistic, clearly-labeled sample opportunities (internships, hackathons, "
    "scholarships, or competitions) relevant to the student's target role and skill "
    "gaps. Respond ONLY with a JSON array: "
    '[{"title": str, "type": "Internship"|"Hackathon"|"Scholarship"|"Competition", '
    '"organization": str, "deadline": str (relative, e.g. "Rolling" or "3 weeks"), '
    '"match_reason": str (1 sentence)}]. Clearly invent organization names; do not '
    "imply these are live postings."
)


def _fallback(role: str, gaps: List[Dict]) -> List[Opportunity]:
    top_gap = gaps[0]["skill"] if gaps else "core technical skills"
    return [
        Opportunity(
            title=f"{role} Summer Internship Program",
            type="Internship",
            organization="Sample Tech Collective",
            deadline="3 weeks",
            match_reason=f"Aligns with the {role} track the Career Recommendation Agent identified.",
        ),
        Opportunity(
            title=f"Build with {top_gap} — 48hr Challenge",
            type="Hackathon",
            organization="Campus Innovation Hub",
            deadline="Rolling",
            match_reason=f"Hands-on practice for the {top_gap} gap flagged by the Skill Gap Agent.",
        ),
        Opportunity(
            title="Future Talent Merit Scholarship",
            type="Scholarship",
            organization="National STEM Foundation",
            deadline="6 weeks",
            match_reason="Open to students actively pursuing a structured upskilling roadmap.",
        ),
        Opportunity(
            title=f"{role} Case Competition",
            type="Competition",
            organization="Industry Skills Council",
            deadline="4 weeks",
            match_reason=f"Portfolio-building opportunity directly tied to the {role} career path.",
        ),
    ]


def run(state: AgentState) -> Dict:
    role = state.get("primary_target_role", "AI Engineer")
    gaps = state.get("skill_gaps", [])

    opportunities = generate_json(
        SYSTEM_PROMPT,
        f"Target role: {role}\nSkill gaps: {gaps}",
        fallback=lambda: _fallback(role, gaps),
        agent_name="OpportunityAgent",
    )

    if not isinstance(opportunities, list) or not opportunities:
        opportunities = _fallback(role, gaps)

    log = list(state.get("agent_log", []))
    log.append({
        "agent": "Internship & Opportunity Agent",
        "summary": f"Surfaced {len(opportunities)} matched opportunities for {role}.",
    })

    return {"opportunities": opportunities, "agent_log": log}
