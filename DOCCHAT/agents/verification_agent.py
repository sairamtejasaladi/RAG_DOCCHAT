"""
Verification Agent — fact-checks the draft answer against source documents.
Uses Azure OpenAI (GPT-4) via the openai SDK.
"""
from openai import AzureOpenAI
from typing import Dict, List
from langchain_core.documents import Document

from docchat.config.settings import settings
from docchat.utils.logging import logger


class VerificationAgent:
    def __init__(self):
        """Initialize the verification agent with Azure OpenAI."""
        logger.info("Initializing VerificationAgent with Azure OpenAI...")
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,
            api_version=settings.AZURE_OPENAI_API_VERSION,
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT,
        )
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT_NAME
        self.model_name = settings.AZURE_OPENAI_MODEL_NAME
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
                    if key in {
                        "Supported",
                        "Unsupported claims",
                        "Contradictions",
                        "Relevant",
                        "Additional details",
                    }:
                        if key in {"Unsupported claims", "Contradictions"}:
                            if value.startswith("[") and value.endswith("]"):
                                items = value[1:-1].split(",")
                                items = [
                                    item.strip().strip('"').strip("'")
                                    for item in items
                                    if item.strip()
                                ]
                                verification[key] = items
                            else:
                                verification[key] = []
                        elif key == "Additional details":
                            verification[key] = value
                        else:
                            verification[key] = value.upper()

            # Ensure all keys are present
            for key in [
                "Supported",
                "Unsupported Claims",
                "Contradictions",
                "Relevant",
                "Additional Details",
            ]:
                if key not in verification:
                    if key in {"Unsupported Claims", "Contradictions"}:
                        verification[key] = []
                    elif key == "Additional Details":
                        verification[key] = ""
                    else:
                        verification[key] = "NO"

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
        if unsupported_claims:
            report += f"**Unsupported Claims:** {', '.join(unsupported_claims)}\n"
        else:
            report += "**Unsupported Claims:** None\n"

        if contradictions:
            report += f"**Contradictions:** {', '.join(contradictions)}\n"
        else:
            report += "**Contradictions:** None\n"

        report += f"**Relevant:** {relevant}\n"

        if additional_details:
            report += f"**Additional Details:** {additional_details}\n"
        else:
            report += "**Additional Details:** None\n"

        return report

    def check(self, answer: str, documents: List[Document]) -> Dict:
        """Verify the answer against the provided documents."""
        logger.info(
            f"VerificationAgent.check called with {len(documents)} documents."
        )

        context = "\n\n".join([doc.page_content for doc in documents])
        prompt = self.generate_prompt(answer, context)

        # LLM call — metrics, latency, tokens, tracing all auto-captured
        try:
            response = self._call_llm(prompt)
            llm_response = response.choices[0].message.content.strip()
            logger.info("Verification LLM response received.")
        except Exception as e:
            logger.error(f"Error during model inference: {e}")
            raise RuntimeError(
                "Failed to verify answer due to a model error."
            ) from e

        # Parse the response
        sanitized = self.sanitize_response(llm_response) if llm_response else ""
        if not sanitized:
            verification_report = {
                "Supported": "NO",
                "Unsupported Claims": [],
                "Contradictions": [],
                "Relevant": "NO",
                "Additional Details": "Empty response from the model.",
            }
        else:
            verification_report = self.parse_verification_response(sanitized)
            if verification_report is None:
                verification_report = {
                    "Supported": "NO",
                    "Unsupported Claims": [],
                    "Contradictions": [],
                    "Relevant": "NO",
                    "Additional Details": "Failed to parse the model's response.",
                }

        report_formatted = self.format_verification_report(verification_report)
        logger.info(f"Verification report generated.")

        return {
            "verification_report": report_formatted,
            "context_used": context,
        }

    def _call_llm(self, prompt: str):
        """Call Azure OpenAI LLM."""
        return self.client.chat.completions.create(
            model=self.deployment,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=500,
        )
