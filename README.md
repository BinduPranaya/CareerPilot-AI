# CareerPilot AI

A multi-agent career navigation system for personalized placement success.
Built for **Agentic Arena 2026 — Round 1: Agent Ideathon** (Education & Learning
track, Case Study 2: Career & Skill Mentor).

CareerPilot AI runs seven specialized AI agents — Profile Analysis, Career
Recommendation, Skill Gap, Learning Roadmap, Internship & Opportunity, Mock
Interview, and Progress Monitoring — orchestrated as a LangGraph state graph,
backed by Gemini + ChromaDB (RAG) for grounded recommendations, with a Spring
Boot API/persistence layer and a React dashboard.

## Architecture

```
┌──────────────┐      REST/JWT      ┌──────────────────┐      REST       ┌────────────────────────┐
│   React +    │ ─────────────────▶ │   Spring Boot     │ ──────────────▶ │   Python agent-service  │
│   Tailwind   │ ◀───────────────── │   (auth, MySQL,    │ ◀────────────── │   FastAPI + LangGraph   │
│   frontend   │                    │   gateway)         │                 │   + Gemini + ChromaDB   │
└──────────────┘                    └──────────────────┘                 └────────────────────────┘
                                              │                                       │
                                              ▼                                       ▼
                                          ┌────────┐                            ┌───────────┐
                                          │ MySQL  │                            │ ChromaDB  │
                                          │(records)│                           │ (vectors) │
                                          └────────┘                            └───────────┘
```

The **agent-service** owns all "thinking": it runs the LangGraph multi-agent
pipeline, calls Gemini, and retrieves grounding context from ChromaDB (RAG).
The **backend** is the system of record: it authenticates students, persists
every agent's output into MySQL, and is the only API the frontend talks to
directly. This separation means the agent orchestration logic, LLM, and
vector store can evolve (or be swapped) independently of the product API.

### Agent graph

```
Profile Analysis → Career Recommendation → Skill Gap → Learning Roadmap
                                                              │
                            ┌─────────────────────────────────┘
                            ▼
              Internship & Opportunity → Mock Interview → Progress Monitoring
                                                              │
                                  (if readiness score is low) │
                            ◀──────────────────────────────────┘
                     autonomous re-plan back to Learning Roadmap
```

The Progress Monitoring Agent can autonomously route execution back to the
Learning Roadmap Agent (capped at 2 retries) — this is what makes the system
agentic rather than a fixed script.

## Repository layout

```
careerpilot-ai/
├── agent-service/    Python · FastAPI · LangGraph · Gemini · ChromaDB
├── backend/          Java 17 · Spring Boot 3 · MySQL · JWT auth
├── frontend/         React 18 · Vite · Tailwind CSS
└── docker-compose.yml
```

## Quick start (Docker Compose)

```bash
cp .env.example .env       # optionally add a GEMINI_API_KEY
docker compose up --build
```

- Frontend: http://localhost:5173
- Backend API: http://localhost:8080
- Agent service: http://localhost:8001 (internal; the backend calls it)
- ChromaDB: http://localhost:8000 (internal)

Leave `GEMINI_API_KEY` blank to run the whole system in **offline/mock
mode** — every agent has a deterministic fallback, so the full pipeline
(profile parsing → recommendations → skill gaps → roadmap → opportunities →
interview questions → readiness scoring, including the autonomous re-plan
loop) works end-to-end with no API key and no cost. Add a real key to switch
every agent to live Gemini calls and to embed RAG documents with Gemini's
`text-embedding-004` instead of the offline hashing fallback.

## Running locally without Docker

### 1. agent-service (Python 3.11+)

```bash
cd agent-service
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env        # optionally set GEMINI_API_KEY
uvicorn app.api:app --reload --port 8001
```

### 2. backend (Java 17 + Maven)

Requires MySQL running locally (or point `MYSQL_URL` at any reachable
instance).

```bash
cd backend
cp .env.example .env
mvn spring-boot:run
```

> This project was assembled in a sandboxed environment without access to
> Maven Central, so the Java code could not be compiled there. It follows
> standard Spring Boot 3 / Spring Security 6 conventions — run `mvn clean
> install` locally (with normal internet access) to build it.

### 3. frontend (Node 18+)

```bash
cd frontend
npm install
cp .env.example .env        # VITE_API_BASE_URL
npm run dev
```

Open http://localhost:5173, register an account, and paste a resume on the
onboarding screen (a "Use a sample resume" button is provided) to trigger
the full agent pipeline.

## API overview (backend, port 8080)

| Method | Path | Auth | Description |
|---|---|---|---|
| POST | `/api/auth/register` | – | Create a student account, returns a JWT |
| POST | `/api/auth/login` | – | Log in, returns a JWT |
| GET | `/api/students/me` | ✅ | Current student's basic info |
| POST | `/api/pipeline/run` | ✅ | Run the full agent pipeline on a resume; persists + returns the dashboard |
| GET | `/api/dashboard` | ✅ | Re-fetch the latest persisted dashboard without re-running agents |
| POST | `/api/interview/{questionId}/answer` | ✅ | Submit an answer; Mock Interview Agent scores it |

Send `Authorization: Bearer <token>` on authenticated routes.

## agent-service endpoints (port 8001, called by the backend)

| Method | Path | Description |
|---|---|---|
| GET | `/health` | Liveness + whether Gemini is live or mock |
| POST | `/api/agents/run` | Runs the full LangGraph pipeline for a student |
| GET | `/api/agents/state/{student_id}` | Last cached run for a student |
| GET | `/api/agents/memory/{student_id}` | Long-term context stored in ChromaDB |
| POST | `/api/interview/evaluate` | Scores a single interview answer |

## What's simplified for this Ideathon round

- **Opportunities** are realistic, clearly-labeled sample listings (an
  internship/hackathon board integration is listed under Scalability &
  Future Scope rather than implemented here).
- **Offline embeddings**: without a Gemini key, RAG retrieval uses a small
  deterministic hashing embedding instead of a real semantic model, so the
  whole system still runs with zero external dependencies for local dev and
  judging.
- **Relational mapping**: child tables (skill gaps, roadmap items, etc.)
  reference `studentId` as a plain foreign-key column rather than a managed
  JPA `@ManyToOne` relationship, to keep the schema simple for this scope.

## Tech stack

Frontend: React, Tailwind CSS · Backend: Spring Boot, REST APIs, JWT · Agent
framework: LangGraph · AI: Gemini API, LLMs, Agent-to-Agent communication ·
Databases: MySQL, ChromaDB · AI features: RAG, resume parsing, skill
matching, personalized learning analytics · Deployment: Docker.
