"""
FastAPI surface for the CareerPilot AI agent-service.

This microservice owns all "thinking" (LangGraph orchestration, Gemini
calls, ChromaDB/RAG). The Spring Boot backend is the system of record for
auth, persistence, and the API the frontend talks to directly -- it calls
this service over HTTP and stores the structured results in MySQL.
"""
from __future__ import annotations

import logging
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from .agents import interview_agent
from .graph import careerpilot_graph
from .llm import is_live
from .schemas import AgentState
from .vectorstore import ensure_seeded, get_student_memory

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("careerpilot.api")

app = FastAPI(title="CareerPilot AI - Agent Service", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tightened at the Spring Boot / gateway layer in production
    allow_methods=["*"],
    allow_headers=["*"],
)

# in-memory cache of each student's last run, keyed by student_id.
# the backend is the durable system of record (MySQL); this is just a
# fast-path cache so the frontend can re-fetch a result without re-running
# the whole graph.
_last_run_cache: dict[str, AgentState] = {}


@app.on_event("startup")
def _startup() -> None:
    ensure_seeded()
    logger.info("CareerPilot agent-service ready. Gemini live mode: %s", is_live())


class RunRequest(BaseModel):
    student_id: str
    resume_text: str
    target_role: Optional[str] = None


class InterviewEvalRequest(BaseModel):
    student_id: str
    question: str
    sample_strong_answer: str
    student_answer: str


@app.get("/health")
def health():
    return {"status": "ok", "gemini_live": is_live()}


@app.post("/api/agents/run")
def run_pipeline(req: RunRequest):
    if not req.resume_text or not req.resume_text.strip():
        raise HTTPException(status_code=400, detail="resume_text is required")

    initial_state: AgentState = {
        "student_id": req.student_id,
        "resume_text": req.resume_text,
        "target_role": req.target_role,
        "agent_log": [],
        "iterations": 0,
    }

    final_state = careerpilot_graph.invoke(initial_state)
    _last_run_cache[req.student_id] = final_state
    return final_state


@app.get("/api/agents/state/{student_id}")
def get_state(student_id: str):
    state = _last_run_cache.get(student_id)
    if state is None:
        raise HTTPException(status_code=404, detail="No pipeline run found for this student yet")
    return state


@app.get("/api/agents/memory/{student_id}")
def get_memory(student_id: str):
    return {"student_id": student_id, "long_term_context": get_student_memory(student_id)}


@app.post("/api/interview/evaluate")
def evaluate_interview_answer(req: InterviewEvalRequest):
    result = interview_agent.evaluate_answer(
        req.question, req.sample_strong_answer, req.student_answer
    )

    # fold this answer's score into the cached state's running interview feedback
    state = _last_run_cache.get(req.student_id)
    if state is not None:
        existing = state.get("interview_feedback") or {"overall_score": 0, "strengths": [], "improvements": []}
        existing["overall_score"] = round((existing.get("overall_score", 0) + result["score"]) / 2) \
            if existing.get("overall_score") else result["score"]
        existing["strengths"] = list(dict.fromkeys(existing.get("strengths", []) + result.get("strengths", [])))[:5]
        existing["improvements"] = list(dict.fromkeys(existing.get("improvements", []) + result.get("improvements", [])))[:5]
        state["interview_feedback"] = existing

    return result
