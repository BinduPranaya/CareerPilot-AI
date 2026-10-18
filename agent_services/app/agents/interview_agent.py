"""Mock Interview Agent.

Generates a tailored set of technical + HR interview questions based on the
target role and flagged skill gaps. A separate `evaluate_answer()` entry
point (called from its own API route, not the main graph) scores a
student's typed response against a strong-answer rubric -- this is what
the frontend's Mock Interview page calls each time the student answers a
question, producing the structured feedback report referenced in the
architecture.
"""
from __future__ import annotations

from typing import Dict, List

from ..llm import generate_json
from ..schemas import AgentState, InterviewQA

QUESTION_SYSTEM_PROMPT = (
    "You are the Mock Interview Agent inside CareerPilot AI. Generate 4 interview "
    "questions for the student's target role: 3 Technical (grounded in their flagged "
    "skill gaps) and 1 HR/behavioral. Respond ONLY with a JSON array: "
    '[{"question": str, "category": "Technical"|"HR", "sample_strong_answer": str '
    "(2-3 sentences describing what a strong answer covers)}]."
)

EVAL_SYSTEM_PROMPT = (
    "You are the Mock Interview Agent inside CareerPilot AI, evaluating a student's "
    "interview answer. Compare it against the sample strong answer and give constructive "
    'feedback. Respond ONLY with JSON: {"score": int (0-100), "strengths": [str], '
    '"improvements": [str]}.'
)


def _fallback_questions(role: str, gaps: List[Dict]) -> List[InterviewQA]:
    gap_skills = [g["skill"] for g in gaps[:3]] or ["core fundamentals"]
    questions: List[InterviewQA] = [
        InterviewQA(
            question=f"Walk me through how you would apply {skill} in a project for a {role} role.",
            category="Technical",
            sample_strong_answer=(
                f"A strong answer names a concrete scenario, explains the reasoning behind using "
                f"{skill}, and mentions trade-offs or alternatives considered."
            ),
        )
        for skill in gap_skills
    ]
    questions.append(InterviewQA(
        question="Tell me about a time you had to learn something new under a tight deadline.",
        category="HR",
        sample_strong_answer=(
            "A strong answer uses a specific situation, the action taken to learn quickly, and a "
            "measurable, positive outcome (STAR structure)."
        ),
    ))
    return questions


def run(state: AgentState) -> Dict:
    role = state.get("primary_target_role", "AI Engineer")
    gaps = state.get("skill_gaps", [])

    questions = generate_json(
        QUESTION_SYSTEM_PROMPT,
        f"Target role: {role}\nSkill gaps: {gaps}",
        fallback=lambda: _fallback_questions(role, gaps),
        agent_name="MockInterviewAgent",
    )

    if not isinstance(questions, list) or not questions:
        questions = _fallback_questions(role, gaps)

    log = list(state.get("agent_log", []))
    log.append({
        "agent": "Mock Interview Agent",
        "summary": f"Prepared {len(questions)} simulated interview questions for {role}.",
    })

    return {"interview_questions": questions, "agent_log": log}


def evaluate_answer(question: str, sample_strong_answer: str, student_answer: str) -> Dict:
    """Used by the standalone /interview/evaluate endpoint, not the main graph."""

    def _fallback() -> Dict:
        student_len = len(student_answer.split())
        ref_words = set(sample_strong_answer.lower().split())
        ans_words = set(student_answer.lower().split())
        overlap = len(ref_words & ans_words)
        score = max(35, min(92, 40 + overlap * 4 + min(student_len, 40)))
        strengths = ["Answer addresses the question directly."] if student_len > 15 else []
        improvements = []
        if student_len < 25:
            improvements.append("Add a concrete example or project to ground the answer.")
        if overlap < 3:
            improvements.append("Reference more of the specific concepts the question is probing for.")
        if not strengths:
            strengths.append("Attempted a relevant response.")
        if not improvements:
            improvements.append("Tighten the answer using a clear structure (situation, action, result).")
        return {"score": score, "strengths": strengths, "improvements": improvements}

    result = generate_json(
        EVAL_SYSTEM_PROMPT,
        f"Question: {question}\nSample strong answer: {sample_strong_answer}\n"
        f"Student's answer: {student_answer}",
        fallback=_fallback,
        agent_name="MockInterviewAgent.evaluate",
    )
    if not isinstance(result, dict) or "score" not in result:
        result = _fallback()
    return result
