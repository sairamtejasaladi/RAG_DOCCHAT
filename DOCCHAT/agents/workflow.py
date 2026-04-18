"""
Agent Workflow — Multi-agent RAG pipeline using LangGraph.
Orchestrates: RelevanceChecker -> ResearchAgent -> VerificationAgent
with self-correction loop on verification failure.
"""
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Dict
from langchain_core.documents import Document

from langchain_classic.retrievers import EnsembleRetriever

from docchat.agents.research_agent import ResearchAgent
from docchat.agents.verification_agent import VerificationAgent
from docchat.agents.relevance_checker import RelevanceChecker
from docchat.utils.logging import logger


class AgentState(TypedDict):
    question: str
    documents: List[Document]
    draft_answer: str
    verification_report: str
    is_relevant: bool
    retriever: EnsembleRetriever
    iteration_count: int
    relevance_classification: str


class AgentWorkflow:
    def __init__(self):
        self.researcher = ResearchAgent()
        self.verifier = VerificationAgent()
        self.relevance_checker = RelevanceChecker()
        self.compiled_workflow = self.build_workflow()

    def build_workflow(self):
        """Create and compile the multi-agent workflow graph."""
        workflow = StateGraph(AgentState)

        # Add nodes
        workflow.add_node("check_relevance", self._check_relevance_step)
        workflow.add_node("research", self._research_step)
        workflow.add_node("verify", self._verification_step)

        # Define edges
        workflow.set_entry_point("check_relevance")
        workflow.add_conditional_edges(
            "check_relevance",
            self._decide_after_relevance_check,
            {"relevant": "research", "irrelevant": END},
        )
        workflow.add_edge("research", "verify")
        workflow.add_conditional_edges(
            "verify",
            self._decide_next_step,
            {"re_research": "research", "end": END},
        )
        return workflow.compile()

    def _check_relevance_step(self, state: AgentState) -> Dict:
        """Check if the question is relevant to the uploaded documents."""
        retriever = state["retriever"]
        classification = self.relevance_checker.check(
            question=state["question"],
            retriever=retriever,
            k=20,
        )

        if classification == "CAN_ANSWER":
            return {"is_relevant": True, "relevance_classification": "CAN_ANSWER"}
        elif classification == "PARTIAL":
            return {"is_relevant": True, "relevance_classification": "PARTIAL"}
        else:  # NO_MATCH
            return {
                "is_relevant": False,
                "relevance_classification": "NO_MATCH",
                "draft_answer": (
                    "This question isn't related (or there's no data) for your query. "
                    "Please ask another question relevant to the uploaded document(s)."
                ),
            }

    def _decide_after_relevance_check(self, state: AgentState) -> str:
        decision = "relevant" if state["is_relevant"] else "irrelevant"
        logger.info(f"Relevance decision: {decision}")
        return decision

    def _research_step(self, state: AgentState) -> Dict:
        """Run the research agent to generate a draft answer."""
        result = self.researcher.generate(state["question"], state["documents"])
        logger.info("Research agent completed draft answer.")
        new_count = state.get("iteration_count", 0) + 1
        return {"draft_answer": result["draft_answer"], "iteration_count": new_count}

    def _verification_step(self, state: AgentState) -> Dict:
        """Run the verification agent to fact-check the draft answer."""
        # Handoff (research_agent -> verification_agent) is auto-tracked
        result = self.verifier.check(state["draft_answer"], state["documents"])
        logger.info("Verification agent completed report.")
        return {"verification_report": result["verification_report"]}

    def _decide_next_step(self, state: AgentState) -> str:
        """Decide whether to re-research or end based on verification."""
        current_count = state.get("iteration_count", 0) + 1
        verification_report = state["verification_report"]
        if (
            "Supported: NO" in verification_report
            or "Relevant: NO" in verification_report
        ):
            if current_count >= 3:
                logger.info(f"Max iterations ({current_count}) reached — ending workflow.")
                return "end"
            logger.info(f"Verification failed (iteration {current_count}) — triggering re-research loop.")
            return "re_research"
        else:
            logger.info(f"Verification successful (iteration {current_count}) — ending workflow.")
            return "end"

    def full_pipeline(self, question: str, retriever: EnsembleRetriever):
        """Execute the full multi-agent RAG pipeline."""
        try:
            logger.info(f"Starting full pipeline with question='{question}'")

            documents = retriever.invoke(question)
            logger.info(f"Retrieved {len(documents)} relevant documents.")

            initial_state = AgentState(
                question=question,
                documents=documents,
                draft_answer="",
                verification_report="",
                is_relevant=False,
                retriever=retriever,
                iteration_count=0,
                relevance_classification="",
            )

            final_state = self.compiled_workflow.invoke(initial_state)

            return {
                "draft_answer": final_state["draft_answer"],
                "verification_report": final_state["verification_report"],
                "documents": final_state["documents"],
                "relevance_classification": final_state.get("relevance_classification", ""),
                "iteration_count": final_state.get("iteration_count", 0),
            }

        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            raise
