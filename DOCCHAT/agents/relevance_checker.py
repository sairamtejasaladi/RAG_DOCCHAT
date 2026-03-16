"""
Relevance Checker — determines if the user's question is within scope of the uploaded documents.
Uses Azure OpenAI for classification.
"""
from openai import AzureOpenAI
from docchat.config.settings import settings
from docchat.utils.logging import logger


class RelevanceChecker:
    def __init__(self):
        """Initialize the relevance checker with Azure OpenAI."""
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.model_name = settings.AZURE_OPENAI_MODEL_NAME

    def check(self, question: str, retriever, k: int = 3) -> str:
        """
        1. Retrieve the top-k document chunks.
        2. Combine them into a single text.
        3. Classify relevance with the LLM.

        Returns: "CAN_ANSWER", "PARTIAL", or "NO_MATCH".
        """
        logger.debug(
            f"RelevanceChecker.check called with question='{question}' and k={k}"
        )

        # Retrieve document chunks
        top_docs = retriever.invoke(question)
        if not top_docs:
            logger.debug("No documents returned. Classifying as NO_MATCH.")
            return "NO_MATCH"

        document_content = "\n\n".join(
            doc.page_content for doc in top_docs[:k]
        )

        prompt = f"""You are an AI relevance checker between a user's question and provided document content.

**Instructions:**
- Classify how well the document content addresses the user's question.
- Respond with only one of the following labels: CAN_ANSWER, PARTIAL, NO_MATCH.
- Do not include any additional text or explanation.

**Labels:**
1) "CAN_ANSWER": The passages contain enough explicit information to fully answer the question.
2) "PARTIAL": The passages mention or discuss the question's topic but do not provide all details.
3) "NO_MATCH": The passages do not discuss or mention the question's topic at all.

**Important:** If the passages mention or reference the topic in any way, even if incomplete, respond with "PARTIAL" instead of "NO_MATCH".

**Question:** {question}

**Passages:** {document_content}

**Respond ONLY with one of: CAN_ANSWER, PARTIAL, NO_MATCH**"""

        # LLM call — metrics, latency, tokens, tracing all auto-captured
        try:
            response = self._call_llm(prompt)
            llm_response = response.choices[0].message.content.strip().upper()
            logger.debug(f"LLM relevance response: {llm_response}")
        except Exception as e:
            logger.error(f"Error during relevance check: {e}")
            return "NO_MATCH"

        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        if llm_response not in valid_labels:
            logger.debug(f"Invalid classification '{llm_response}'. Defaulting to NO_MATCH.")
            return "NO_MATCH"

        return llm_response

    def _call_llm(self, prompt: str):
        """Call Azure OpenAI LLM."""
        return self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=10,
        )
