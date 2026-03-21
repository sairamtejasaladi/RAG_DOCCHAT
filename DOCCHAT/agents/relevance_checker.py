"""
Relevance Checker — determines if the user's question is within scope of the uploaded documents.
Uses local Ollama for classification.
"""
from langchain_ollama import ChatOllama
from docchat.config.settings import settings
from docchat.utils.logging import logger

class RelevanceChecker:
    def __init__(self):
        """Initialize the relevance checker with local Ollama."""
        # Initializing the LangChain Ollama client
        self.llm = ChatOllama(
            model=settings.LLM_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,
            # Adding a small limit to speed up classification
            num_predict=10 
        )

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

        try:
            # The .invoke() method returns an AIMessage object
            response = self.llm.invoke(prompt)
            # Access content directly from the message object
            llm_response = response.content.strip().upper()
            
            logger.debug(f"LLM relevance response: {llm_response}")
        except Exception as e:
            logger.error(f"Error during relevance check: {e}")
            return "NO_MATCH"

        valid_labels = {"CAN_ANSWER", "PARTIAL", "NO_MATCH"}
        
        # Simple check to ensure the model didn't return extra text
        for label in valid_labels:
            if label in llm_response:
                return label

        logger.debug(f"Invalid classification '{llm_response}'. Defaulting to NO_MATCH.")
        return "NO_MATCH"