"""
DocChat Runner — launch the DocChat RAG application.

Run from the workspace root:
    python run_docchat.py

Prerequisites:
    1. pip install -r requirements_docchat.txt
    2. Create .env file with Azure OpenAI credentials:
       - AZURE_OPENAI_API_KEY
       - AZURE_OPENAI_ENDPOINT
       - AZURE_OPENAI_DEPLOYMENT_NAME
    3. Ensure Python 3.10+ is installed
"""
import sys
import os

# Ensure the workspace root is on sys.path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from docchat.app import main

if __name__ == "__main__":
    main()
