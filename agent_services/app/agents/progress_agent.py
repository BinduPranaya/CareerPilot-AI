"""Progress Monitoring Agent.

Computes a single placement-readiness score from everything the other
agents have produced so far, and makes an autonomous decision: if the
score is below threshold and the orchestrator hasn't already retried, it
flags `replan_triggered=True`, which graph.py uses as a conditional edge
back to the Learning Roadmap Agent. This is the system's core "autonomous
decision making" loop -- no human has to notice the student is off-track
and manually ask for a new plan.
"""
from __future__ import annotations

from typing import Dict

from ..schemas import AgentState

REPLAN_THRESHOLD = 55
MAX_ITERATIONS = 2


def _label_for(score: int) -> str:
    if score >= 90:
        return "Placement Ready"
    if score >= 75:
        return "Placement Ready Soon"
    if score >= 55:
        return "Building Momentum"
    return "Early Stage"


def run(state: AgentState) -> Dict:
    skill_gaps = state.get("skill_gaps", [])
    roadmap = state.get("roadmap", [])
    interview_feedback = state.get("interview_feedback") or {}
    iterations = state.get("iterations", 0)

    critical_unresolved = sum(1 for g in skill_gaps if g.get("priority") == "Critical")

    score = 42
    score += min(len(roadmap), 8) * 2          # credit for a concrete plan existing
    score -= critical_unresolved * 6           # penalize unresolved critical gaps
    if interview_feedback.get("overall_score"):
        score = round(score * 0.7 + interview_feedback["overall_score"] * 0.3)
    else:
        score += 10  # neutral baseline boost before any interview practice has happened
    score += 5 * iterations                    # credit for having already re-planned once
    score = max(5, min(98, score))

    replan_triggered = score < REPLAN_THRESHOLD and iterations < MAX_ITERATIONS

    log = list(state.get("agent_log", []))
    log.append({
        "agent": "Progress Monitoring Agent",
        "summary": (
            f"Computed placement-readiness score of {score} "
            f"({'triggering an autonomous re-plan' if replan_triggered else 'no re-plan needed'})."
        ),
    })

    return {
        "readiness_score": score,
        "readiness_label": _label_for(score),
        "replan_triggered": replan_triggered,
        "iterations": iterations + (1 if replan_triggered else 0),
        "agent_log": log,
    }
