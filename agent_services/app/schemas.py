"""
Shared data contracts for the CareerPilot AI agent graph.

`AgentState` is the single object that flows through every node of the
LangGraph StateGraph. Every agent reads the fields it needs and writes back
the fields it owns -- this shared, growing context is what lets later agents
(e.g. the Mock Interview Agent) act on what earlier agents (e.g. the Skill
Gap Agent) already discovered, without re-deriving it.
"""
from __future__ import annotations

from typing import List, Optional, TypedDict


class StudentProfile(TypedDict, total=False):
    name: str
    education: str
    skills: List[str]
    interests: List[str]
    experience_summary: str


class CareerRecommendation(TypedDict, total=False):
    role: str
    fit_score: int          # 0-100
    rationale: str
    market_demand: str      # "High" | "Medium" | "Low"


class SkillGap(TypedDict, total=False):
    skill: str
    priority: str            # "Critical" | "Important" | "Nice to have"
    current_level: str       # "None" | "Beginner" | "Intermediate" | "Advanced"
    target_level: str


class RoadmapItem(TypedDict, total=False):
    week: int
    focus_skill: str
    activity: str
    resource: str


class Opportunity(TypedDict, total=False):
    title: str
    type: str                # "Internship" | "Hackathon" | "Scholarship" | "Competition"
    organization: str
    deadline: str
    match_reason: str


class InterviewQA(TypedDict, total=False):
    question: str
    category: str            # "Technical" | "HR"
    sample_strong_answer: str


class InterviewFeedback(TypedDict, total=False):
    overall_score: int       # 0-100
    strengths: List[str]
    improvements: List[str]


class AgentLogEntry(TypedDict, total=False):
    agent: str
    summary: str


class AgentState(TypedDict, total=False):
    # --- input ---
    student_id: str
    resume_text: str
    target_role: Optional[str]

    # --- written by Profile Analysis Agent ---
    profile: StudentProfile

    # --- written by Career Recommendation Agent ---
    career_recommendations: List[CareerRecommendation]
    primary_target_role: str

    # --- written by Skill Gap Agent ---
    skill_gaps: List[SkillGap]

    # --- written by Learning Roadmap Agent ---
    roadmap: List[RoadmapItem]

    # --- written by Internship & Opportunity Agent ---
    opportunities: List[Opportunity]

    # --- written by Mock Interview Agent ---
    interview_questions: List[InterviewQA]
    interview_feedback: InterviewFeedback

    # --- written by Progress Monitoring Agent ---
    readiness_score: int
    readiness_label: str
    replan_triggered: bool

    # --- orchestration bookkeeping ---
    agent_log: List[AgentLogEntry]
    iterations: int
