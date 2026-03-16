"""
DocChat — AI-powered Multi-Agent RAG system.

This is the main entry point for the DocChat application.
It initializes:
  1. Azure OpenAI client configuration from .env
  2. Document processing pipeline
  3. Hybrid retrieval system (BM25 + Vector)
  4. Multi-agent workflow (Relevance → Research → Verification)
  5. Gradio web interface for document upload and Q&A

All configuration is loaded from the .env file via settings.py.
"""
import sys
import os
import hashlib
from typing import List, Dict

from dotenv import load_dotenv
load_dotenv()

import gradio as gr

from docchat.document_processor.file_handler import DocumentProcessor
from docchat.retriever.builder import RetrieverBuilder
from docchat.agents.workflow import AgentWorkflow
from docchat.config import constants
from docchat.config.settings import settings
from docchat.utils.logging import logger

# ── Sample examples (optional — users can also upload their own) ────────────
EXAMPLES = {
    "Sample: Upload your own documents": {
        "question": "Ask any question about your uploaded document(s)",
        "file_paths": [],
    },
}


def main():
    processor = DocumentProcessor()
    retriever_builder = RetrieverBuilder()
    workflow = AgentWorkflow()

    # ── Custom CSS ──────────────────────────────────────────────────────────
    css = """
    .title {
        font-size: 1.5em !important;
        text-align: center !important;
        color: #FFD700;
    }
    .subtitle {
        font-size: 1em !important;
        text-align: center !important;
        color: #FFD700;
    }
    .text { text-align: center; }
    .observability-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        text-align: center;
        font-size: 0.85em;
        margin: 10px auto;
        max-width: 500px;
    }
    """

    js = """
    function createGradioAnimation() {
        var container = document.createElement('div');
        container.id = 'gradio-animation';
        container.style.fontSize = '2em';
        container.style.fontWeight = 'bold';
        container.style.textAlign = 'center';
        container.style.marginBottom = '20px';
        container.style.color = '#eba93f';
        var text = 'Welcome to DocChat - Multi-Agent RAG!';
        for (var i = 0; i < text.length; i++) {
            (function(i){
                setTimeout(function(){
                    var letter = document.createElement('span');
                    letter.style.opacity = '0';
                    letter.style.transition = 'opacity 0.1s';
                    letter.innerText = text[i];
                    container.appendChild(letter);
                    setTimeout(function() { letter.style.opacity = '0.9'; }, 50);
                }, i * 80);
            })(i);
        }
        var gc = document.querySelector('.gradio-container');
        gc.insertBefore(container, gc.firstChild);
        return 'Animation created';
    }
    """

    with gr.Blocks(title="DocChat") as demo:
        gr.Markdown("## DocChat: Multi-Agent RAG Application", elem_classes="subtitle")
        gr.Markdown("# How it works", elem_classes="title")
        gr.Markdown(
            "Upload your document(s) (PDF, DOCX, TXT, MD), enter your query, "
            "then hit **Submit**. The multi-agent pipeline will retrieve, research, "
            "verify, and return the answer — all fully traced!",
            elem_classes="text",
        )

        # Session state
        session_state = gr.State({"file_hashes": frozenset(), "retriever": None})

        with gr.Row():
            with gr.Column():
                files = gr.Files(
                    label="Upload Documents",
                    file_types=[".pdf", ".docx", ".txt", ".md"],
                )
                question = gr.Textbox(label="Question", lines=3, placeholder="Ask a question about your documents...")
                submit_btn = gr.Button("Submit", variant="primary")

            with gr.Column():
                answer_output = gr.Textbox(label="Answer", interactive=False, lines=10)
                verification_output = gr.Textbox(label="Verification Report", interactive=False, lines=8)

        gr.Markdown(
            "---\n"
            "**DocChat** is a multi-agent RAG system powered by Azure OpenAI."
        )

        # ── Process question ────────────────────────────────────────────────
        def process_question(question_text: str, uploaded_files: List, state: Dict):
            """Handle questions with document caching and full observability."""
            try:
                if not question_text or not question_text.strip():
                    raise ValueError("Question cannot be empty")
                if not uploaded_files:
                    raise ValueError("No documents uploaded")

                current_hashes = _get_file_hashes(uploaded_files)

                if state["retriever"] is None or current_hashes != state["file_hashes"]:
                    logger.info("Processing new/changed documents...")
                    chunks = processor.process(uploaded_files)
                    if not chunks:
                        raise ValueError(
                            "No text could be extracted from the uploaded documents."
                        )
                    retriever = retriever_builder.build_hybrid_retriever(chunks)
                    state.update({"file_hashes": current_hashes, "retriever": retriever})

                result = workflow.full_pipeline(
                    question=question_text,
                    retriever=state["retriever"],
                )

                return result["draft_answer"], result["verification_report"], state

            except Exception as e:
                logger.error(f"Processing error: {str(e)}")
                return f"Error: {str(e)}", "", state

        submit_btn.click(
            fn=process_question,
            inputs=[question, files, session_state],
            outputs=[answer_output, verification_output, session_state],
        )

    demo.launch(server_name="127.0.0.1", server_port=7860, share=False, css=css, js=js)


def _get_file_hashes(uploaded_files: List) -> frozenset:
    """Generate SHA-256 hashes for uploaded files."""
    hashes = set()
    for file in uploaded_files:
        fpath = file if isinstance(file, str) else file.name
        with open(fpath, "rb") as f:
            hashes.add(hashlib.sha256(f.read()).hexdigest())
    return frozenset(hashes)


if __name__ == "__main__":
    main()
