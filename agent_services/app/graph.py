"""
The Agent Orchestrator.

Builds the LangGraph ``StateGraph`` that wires together all seven
CareerPilot AI agents. The graph is intentionally not a flat pipeline: the
Progress Monitoring Agent can route execution back to the Learning Roadmap
Agent (a conditional edge), which is what makes this an *agentic* system
rather than a fixed script -- the orchestrator changes the plan based on
what it autonomously decides during execution.

    Profile -> Career -> SkillGap -> Roadmap -> Opportunity -> Interview -> Progress
                            ^                                                  |
                            +------------------ replan_triggered --------------+
"""
from __future__ import annotations

from langgraph.graph import END, StateGraph

from .agents import (
    career_agent,
    interview_agent,
    opportunity_agent,
    profile_agent,
    progress_agent,
    roadmap_agent,
    skill_gap_agent,
)
from .schemas import AgentState


def _route_after_progress(state: AgentState) -> str:
    return "learning_roadmap" if state.get("replan_triggered") else END


def build_graph():
    graph = StateGraph(AgentState)

    graph.add_node("profile_analysis", profile_agent.run)
    graph.add_node("career_recommendation", career_agent.run)
    graph.add_node("skill_gap_analysis", skill_gap_agent.run)
    graph.add_node("learning_roadmap", roadmap_agent.run)
    graph.add_node("opportunity_matching", opportunity_agent.run)
    graph.add_node("mock_interview", interview_agent.run)
    graph.add_node("progress_monitoring", progress_agent.run)

    graph.set_entry_point("profile_analysis")
    graph.add_edge("profile_analysis", "career_recommendation")
    graph.add_edge("career_recommendation", "skill_gap_analysis")
    graph.add_edge("skill_gap_analysis", "learning_roadmap")
    graph.add_edge("learning_roadmap", "opportunity_matching")
    graph.add_edge("opportunity_matching", "mock_interview")
    graph.add_edge("mock_interview", "progress_monitoring")
    graph.add_conditional_edges(
        "progress_monitoring",
        _route_after_progress,
        {"learning_roadmap": "learning_roadmap", END: END},
    )

    return graph.compile()


# Compiled once at import time and reused across requests.
careerpilot_graph = build_graph()
