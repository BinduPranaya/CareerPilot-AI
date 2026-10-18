"""Career Recommendation Agent.

Matches the student's profile against a small set of target roles, using
RAG retrieval over the industry skill knowledge base (ChromaDB) to ground
each recommendation's market-demand signal in real reference text rather
than the model's unverified opinion.
"""
from __future__ import annotations

from typing import Dict, List

from ..llm import generate_json
from ..schemas import AgentState, CareerRecommendation
from ..vectorstore import query_role_requirements

CANDIDATE_ROLES = [
    "AI Engineer",
    "Data Scientist",
    "Backend Developer",
    "Cloud Engineer",
    "Cybersecurity Analyst",
]

SYSTEM_PROMPT = (
    "You are the Career Recommendation Agent inside CareerPilot AI. Given a student's "
    "profile and grounding context about what each candidate role expects, rank the "
    "candidate roles for this student. Respond ONLY with a JSON array of objects: "
    '[{"role": str, "fit_score": int (0-100), "rationale": str (1 sentence), '
    '"market_demand": "High"|"Medium-High"|"Medium"}]. Sort by fit_score descending.'
)


def _overlap_score(skills: List[str], requirement_text: str) -> int:
    if not skills:
        return 40
    text = requirement_text.lower()
    hits = sum(1 for s in skills if s.lower() in text)
    base = 35 + hits * 12
    return max(20, min(96, base))


def _demand_from_text(text: str) -> str:
    text_l = text.lower()
    if "high" in text_l and "medium" not in text_l.split("market demand")[-1]:
        return "High"
    if "medium-high" in text_l:
        return "Medium-High"
    return "Medium"


def _fallback(skills: List[str], target_role: str | None) -> List[CareerRecommendation]:
    recs: List[CareerRecommendation] = []
    for role in CANDIDATE_ROLES:
        reqs = query_role_requirements(role, n_results=1)
        req_text = reqs[0] if reqs else f"{role} requires standard industry skills."
        score = _overlap_score(skills, req_text)
        if target_role and target_role.lower() in role.lower():
            score = min(98, score + 15)
        recs.append(CareerRecommendation(
            role=role,
            fit_score=score,
            rationale=f"Estimated from overlap between the student's skills and {role} expectations.",
            market_demand=_demand_from_text(req_text),
        ))
    recs.sort(key=lambda r: r["fit_score"], reverse=True)
    return recs


def run(state: AgentState) -> Dict:
    profile = state.get("profile", {})
    skills = profile.get("skills", [])
    target_role = state.get("target_role")

    grounding = "\n".join(
        f"- {role}: {query_role_requirements(role, 1)[0] if query_role_requirements(role, 1) else ''}"
        for role in CANDIDATE_ROLES
    )

    user_prompt = (
        f"Student profile: {profile}\n\n"
        f"Student's stated target role (may be empty): {target_role}\n\n"
        f"Role requirement grounding (retrieved via RAG):\n{grounding}\n\n"
        f"Candidate roles to rank exactly: {CANDIDATE_ROLES}"
    )

    recommendations = generate_json(
        SYSTEM_PROMPT,
        user_prompt,
        fallback=lambda: _fallback(skills, target_role),
        agent_name="CareerRecommendationAgent",
    )

    if not isinstance(recommendations, list) or not recommendations:
        recommendations = _fallback(skills, target_role)

    recommendations = sorted(recommendations, key=lambda r: r.get("fit_score", 0), reverse=True)
    primary_role = recommendations[0]["role"] if recommendations else (target_role or "AI Engineer")

    log = list(state.get("agent_log", []))
    log.append({
        "agent": "Career Recommendation Agent",
        "summary": f"Ranked {len(recommendations)} roles; top match is {primary_role}.",
    })

    return {
        "career_recommendations": recommendations,
        "primary_target_role": primary_role,
        "agent_log": log,
    }
