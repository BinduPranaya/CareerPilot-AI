"""
Vector memory layer for CareerPilot AI, backed by ChromaDB.

Two collections live here:

1. ``industry_skill_kb`` -- a small seeded knowledge base describing what
   each target role expects (skills, tools, seniority signals). The Career
   Recommendation Agent and Skill Gap Agent retrieve from this collection
   via similarity search -- this is the RAG step referenced in the tech
   stack ("Retrieval-Augmented Generation (RAG)" + "ChromaDB Vector
   Database").

2. ``student_memory`` -- one growing entry per student holding a running
   summary of everything the agents have learned about them. This is what
   gives CareerPilot AI *long-term* context: any agent can recall a
   student's history across sessions instead of only the current request.

Embeddings: when GEMINI_API_KEY is configured we embed with Gemini's
``text-embedding-004`` model for real semantic retrieval. Otherwise we fall
back to a small deterministic hashing embedding so the whole system -
including RAG - still runs fully offline for local development and judging.
"""
from __future__ import annotations

import math
import os
import re
from typing import List, Sequence

import chromadb

from .llm import GEMINI_API_KEY, is_live

CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_data")
CHROMA_HOST = os.getenv("CHROMA_HOST", "").strip()
CHROMA_PORT = int(os.getenv("CHROMA_PORT", "8000"))

EMBED_DIM = 256


class _HashingEmbeddingFunction:
    """Deterministic, dependency-free embedding used when no API key is set."""

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        return [self._embed(text) for text in input]

    @staticmethod
    def _embed(text: str) -> List[float]:
        vec = [0.0] * EMBED_DIM
        for token in re.findall(r"[a-zA-Z0-9+]+", text.lower()):
            idx = hash(token) % EMBED_DIM
            vec[idx] += 1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]


class _GeminiEmbeddingFunction:
    """Real semantic embeddings via the Gemini embedding model."""

    def __call__(self, input: Sequence[str]) -> List[List[float]]:
        import google.generativeai as genai

        out: List[List[float]] = []
        for text in input:
            try:
                result = genai.embed_content(model="models/text-embedding-004", content=text)
                out.append(result["embedding"])
            except Exception:
                out.append(_HashingEmbeddingFunction._embed(text))
        return out


def _make_client():
    if CHROMA_HOST:
        return chromadb.HttpClient(host=CHROMA_HOST, port=CHROMA_PORT)
    return chromadb.PersistentClient(path=CHROMA_PATH)


_embedding_fn = _GeminiEmbeddingFunction() if (GEMINI_API_KEY and is_live()) else _HashingEmbeddingFunction()
_client = _make_client()

skill_kb = _client.get_or_create_collection(name="industry_skill_kb", embedding_function=_embedding_fn)
student_memory = _client.get_or_create_collection(name="student_memory", embedding_function=_embedding_fn)


# ---------------------------------------------------------------------------
# Seed data: this is the retrieval corpus the agents ground their
# recommendations in. In production this would be refreshed from live job
# postings / labor-market feeds (see Scalability & Future Scope).
# ---------------------------------------------------------------------------
_SEED_ROLES = {
    "AI Engineer": (
        "AI Engineer expects strong Python, PyTorch or TensorFlow, prompt engineering, "
        "RAG pipelines, vector databases, LLM fine-tuning basics, REST API design, and "
        "familiarity with MLOps/Docker. Market demand: High."
    ),
    "Data Scientist": (
        "Data Scientist expects Python or R, SQL, statistics and probability, pandas, "
        "scikit-learn, data visualization, A/B testing, and clear stakeholder communication. "
        "Market demand: High."
    ),
    "Backend Developer": (
        "Backend Developer expects Java or Node.js, Spring Boot or Express, REST/GraphQL "
        "API design, relational databases, system design fundamentals, Git, and testing "
        "practices. Market demand: High."
    ),
    "Cloud Engineer": (
        "Cloud Engineer expects AWS or Azure or GCP, Docker and Kubernetes, Terraform or "
        "CloudFormation, CI/CD pipelines, networking basics, and cost/security best "
        "practices. Market demand: Medium-High."
    ),
    "Cybersecurity Analyst": (
        "Cybersecurity Analyst expects network security fundamentals, SIEM tools, threat "
        "modeling, scripting (Python/Bash), Linux, vulnerability assessment, and security "
        "certifications such as CompTIA Security+. Market demand: Medium-High."
    ),
}


def ensure_seeded() -> None:
    """Seed the industry knowledge base once, if empty."""
    if skill_kb.count() > 0:
        return
    ids = list(_SEED_ROLES.keys())
    skill_kb.add(
        ids=ids,
        documents=list(_SEED_ROLES.values()),
        metadatas=[{"role": role} for role in ids],
    )


def query_role_requirements(role: str, n_results: int = 2) -> List[str]:
    """RAG retrieval step: fetch the most relevant role-expectation documents."""
    ensure_seeded()
    results = skill_kb.query(query_texts=[role], n_results=min(n_results, skill_kb.count() or 1))
    docs = results.get("documents", [[]])
    return docs[0] if docs else []


def upsert_student_memory(student_id: str, summary_text: str, metadata: dict) -> None:
    """Persist/refresh the long-term context blob for a student."""
    student_memory.upsert(
        ids=[student_id],
        documents=[summary_text],
        metadatas=[metadata],
    )


def get_student_memory(student_id: str) -> str:
    result = student_memory.get(ids=[student_id])
    docs = result.get("documents") or []
    return docs[0] if docs else ""
