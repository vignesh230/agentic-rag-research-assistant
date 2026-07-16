"""Build and compile the agentic RAG LangGraph.

Call build_graph() once at startup and cache the result — compilation is
expensive (~100ms) and the graph is stateless between invocations.
"""

from __future__ import annotations

import structlog
from langgraph.graph import END, START, StateGraph

from rag_agent.db.client import DBClient
from rag_agent.graph.nodes import make_critic, make_planner, make_retrieve, make_synthesizer
from rag_agent.graph.state import AgentState
from rag_agent.ingestion.embedder import Embedder
from rag_agent.llm import get_llm
from rag_agent.settings import Settings

log = structlog.get_logger(__name__)


def build_graph(settings: Settings, db: DBClient, embedder: Embedder, top_k: int):
    """Compile the agentic RAG graph with injected dependencies.

    Args:
        settings: Application settings (model, max_critic_loops, etc.)
        db: Initialised DBClient for vector retrieval.
        embedder: Initialised Embedder for query embedding.
        top_k: Number of chunks to retrieve per sub-question.

    Returns:
        Compiled LangGraph runnable.
    """
    llm = get_llm(settings)

    planner = make_planner(settings, llm)
    retrieve = make_retrieve(settings, db, embedder, top_k)
    synthesizer = make_synthesizer(settings, llm)
    critic = make_critic(settings, llm)

    def _route_after_critic(state: AgentState) -> str:
        verdict = (state.get("critic_verdict") or "supported").lower()
        loops = state.get("critic_loops", 0)
        # If critic wants a rewrite AND we haven't hit the cap, loop back.
        if verdict.startswith("rewrite:") and loops < settings.max_critic_loops:
            return "retrieve"
        return END

    g: StateGraph = StateGraph(AgentState)
    g.add_node("planner", planner)
    g.add_node("retrieve", retrieve)
    g.add_node("synthesizer", synthesizer)
    g.add_node("critic", critic)

    g.add_edge(START, "planner")
    g.add_edge("planner", "retrieve")
    g.add_edge("retrieve", "synthesizer")
    g.add_edge("synthesizer", "critic")
    g.add_conditional_edges("critic", _route_after_critic, {"retrieve": "retrieve", END: END})

    log.debug("agentic.graph.compiled", max_critic_loops=settings.max_critic_loops)
    return g.compile()
