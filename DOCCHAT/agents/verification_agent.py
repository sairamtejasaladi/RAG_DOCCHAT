"""
Verification Agent — fact-checks the draft answer against source documents.
Uses local Ollama (Llama 3.1) via LangChain.
"""
from typing import Dict, List
from langchain_core.documents import Document
from langchain_ollama import ChatOllama

from docchat.config.settings import settings
from docchat.utils.logging import logger

class VerificationAgent:
    def __init__(self):
        """Initialize the verification agent with local Ollama."""
        logger.info(f"Initializing VerificationAgent with local model: {settings.LLM_MODEL_NAME}")
        
        # Initialize the LangChain Ollama client
        self.llm = ChatOllama(
            model=settings.LLM_MODEL_NAME,
            base_url=settings.OLLAMA_BASE_URL,
            temperature=0,  # CRITICAL: Keep at 0 for deterministic fact-checking
            num_ctx=4096,   # Ensures enough space for answer + context
            num_predict=500 # Matches your previous max_tokens setting
        )
        logger.info("VerificationAgent initialized successfully.")

    def sanitize_response(self, response_text: str) -> str:
        return response_text.strip()

    def generate_prompt(self, answer: str, context: str) -> str:
        """Generate a prompt for verifying the answer against source context."""
        return f"""You are an AI assistant designed to verify the accuracy and relevance of answers based on provided context.

**Instructions:**
- Verify the following answer against the provided context.
- Check for:
  1. Direct/indirect factual support (YES/NO)
  2. Unsupported claims (list any if present)
  3. Contradictions (list any if present)
  4. Relevance to the question (YES/NO)
- Provide additional details or explanations where relevant.
- Respond in the exact format specified below.

**Format:**
Supported: YES/NO
Unsupported Claims: [item1, item2, ...]
Contradictions: [item1, item2, ...]
Relevant: YES/NO
Additional Details: [Any extra information or explanations]

**Answer:** {answer}

**Context:**
{context}

**Respond ONLY with the above format.**"""

    def parse_verification_response(self, response_text: str) -> Dict:
        """Parse the LLM's verification response into a structured dictionary."""
        try:
            lines = response_text.split("\n")
            verification = {}
            for line in lines:
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().capitalize()
                    value = value.strip()
                    # Mapping local model keys to expected keys
                    key_map = {
                        "Supported": "Supported",
                        "Unsupported claims": "Unsupported Claims",
                        "Contradictions": "Contradictions",
                        "Relevant": "Relevant",
                        "Additional details": "Additional Details"
                    }
                    target_key = key_map.get(key)
                    if target_key:
                        if target_key in {"Unsupported Claims", "Contradictions"}:
                            if value.startswith("[") and value.endswith("]"):
                                items = value[1:-1].split(",")
                                items = [
                                    item.strip().strip('"').strip("'")
                                    for item in items
                                    if item.strip()
                                ]
                                verification[target_key] = items
                            else:
                                verification[target_key] = []
                        elif target_key == "Additional Details":
                            verification[target_key] = value
                        else:
                            verification[target_key] = value.upper()

            # Fill missing keys to prevent UI/Workflow crashes
            for key in ["Supported", "Unsupported Claims", "Contradictions", "Relevant", "Additional Details"]:
                if key not in verification:
                    verification[key] = [] if "Claims" in key or "Contradictions" in key else ("NO" if key != "Additional Details" else "")

            return verification
        except Exception as e:
            logger.error(f"Error parsing verification response: {e}")
            return None

    def format_verification_report(self, verification: Dict) -> str:
        """Format the verification report into a readable string."""
        supported = verification.get("Supported", "NO")
        unsupported_claims = verification.get("Unsupported Claims", [])
        contradictions = verification.get("Contradictions", [])
        relevant = verification.get("Relevant", "NO")
        additional_details = verification.get("Additional Details", "")

        report = f"**Supported:** {supported}\n"
        report += f"**Unsupported Claims:** {', '.join(unsupported_claims) if unsupported_claims else 'None'}\n"
        report += f"**Contradictions:** {', '.join(contradictions) if contradictions else 'None'}\n"
        report += f"**Relevant:** {relevant}\n"
        report += f"**Additional Details:** {additional_details if additional_details else 'None'}\n"

        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """Verify the answer against the provided documents."""
        logger.info(f"VerificationAgent.check called with {len(documents)} documents.")

        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(answer, context)

        try:
            # Replaces the old _call_llm and Azure structure
            response = self.llm.invoke(prompt)
            llm_response = response.content.strip()
            logger.info("Verification LLM response received.")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise RuntimeError("Failed to verify answer due to a local model error.") from e

        # Parsing remains the same but handles the new string content
        sanitized = self.sanitize_response(llm_response)
        if not sanitized:
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the local model.",
            }
        else:
            verification_report = self.parse_verification_response(sanitized)
            if verification_report is None:
                verification_report = {
                    "Supported": "NO",
                    "Unsupported Claims": [],
                    "Contradictions": [],
                    "Relevant": "NO",
                    "Additional Details": "Failed to parse local model response.",
                }

        report_formatted = self.format_verification_report(verification_report)
        return {
            "verification_report": report_formatted,
            "context_used": context,
        }