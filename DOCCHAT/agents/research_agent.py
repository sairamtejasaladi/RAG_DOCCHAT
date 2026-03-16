"""
Research Agent — generates answers from retrieved document context.
Uses Azure OpenAI (GPT-4) via the openai SDK.
"""
from openai import AzureOpenAI
from typing import Dict, List
from langchain_core.documents import Document

from docchat.config.settings import settings
from docchat.utils.logging import logger


class ResearchAgent:
    def __init__(self):
        """Initialize the research agent with Azure OpenAI."""
        logger.info("Initializing ResearchAgent with Azure OpenAI...")
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.model_name = settings.AZURE_OPENAI_MODEL_NAME
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

        context = "\n\n".join([doc.page_content for doc in documents])
        logger.info(f"Combined context length: {len(context)} characters.")

        prompt = self.generate_prompt(question, context)

        # LLM call — metrics, latency, tokens, tracing all auto-captured
        try:
            response = self._call_llm(prompt)
            llm_response = response.choices[0].message.content.strip()
            logger.info("Research LLM response received.")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise RuntimeError(
                "Failed to generate answer due to a model error."
            ) from e

        draft_answer = (
            self.sanitize_response(llm_response)
            if llm_response
            else "I cannot answer this question based on the provided documents."
        )
        logger.info(f"Generated answer length: {len(draft_answer)} chars")

        return {"draft_answer": draft_answer, "context_used": context}

    def _call_llm(self, prompt: str):
        """Call Azure OpenAI LLM."""
        return self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=800,
        )
