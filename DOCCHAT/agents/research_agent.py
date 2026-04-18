"""
Research Agent — generates answers from retrieved document context.
Uses local Ollama (Llama 3.1) via LangChain.
"""
from typing import Dict, List
from langchain_core.documents import Document

from docchat.config.settings import settings
from docchat.utils.logging import logger
from docchat.utils.llm_factory import get_llm

class ResearchAgent:
    def __init__(self):
        """Initialize the research agent."""
        logger.info(f"Initializing ResearchAgent (provider={settings.LLM_PROVIDER})")
        self.llm = get_llm(temperature=0.3, num_ctx=4096)
        logger.info("ResearchAgent initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        """Sanitize the LLM's response by stripping unnecessary whitespace."""
        return response_text.strip()

    def generate_prompt(self, question: str, context: str) -> str:
        """Generate a structured prompt for the LLM to generate a precise answer."""
        return f"""You are an AI assistant designed to provide precise and factual answers based on the given context.

**Instructions:**
- Answer the following question using only the provided context.
- Be clear, concise, and factual.
- Return as much information as you can get from the context.
- If the context does not contain enough information, say so.

**Question:** {question}

**Context:**
{context}

**Provide your answer below:**"""

    def generate(self, question: str, documents: List[Document]) -> Dict:
        """Generate an initial answer using the provided documents."""
        logger.info(
            f"ResearchAgent.generate called with question='{question}' "
            f"and {len(documents)} documents."
        )

        # Combine document contents into one context string
        context = "\n\n".join([doc.page_content for doc in documents])
        logger.info(f"Combined context length: {len(context)} characters.")

        prompt = self.generate_prompt(question, context)

        try:
            # LangChain .invoke() handles the call to your local Ollama server
            # This replaces self._call_llm and the openai choices parsing
            response = self.llm.invoke(prompt)
            llm_response = response.content.strip()
            
            logger.info("Research LLM response received.")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise RuntimeError(
                "Failed to generate answer due to a local model error."
            ) from e

        draft_answer = (
            self.sanitize_response(llm_response)
            if llm_response
            else "I cannot answer this question based on the provided documents."
        )
        logger.info(f"Generated answer length: {len(draft_answer)} chars")

        return {"draft_answer": draft_answer, "context_used": context}