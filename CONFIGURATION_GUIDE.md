# DocChat Configuration Guide

## Quick Reference for .env Setup

This guide explains how your Azure OpenAI credentials flow from the `.env` file to all agents in the system.

---

## 📋 Configuration Flow

```
.env file (root directory)
    │
    │ Loaded by pydantic_settings.BaseSettings
    ▼
settings.py (docchat/config/settings.py)
    │
    │ Imported as 'settings' singleton
    ▼
All Agents:
    ├─► research_agent.py
    ├─► verification_agent.py
    └─► relevance_checker.py
```

---

## 🔧 Step-by-Step Setup

### Step 1: Create .env File

Create a file named `.env` in your project root (`doc_chat/.env`):

```env
# Azure OpenAI Configuration


### Step 2: Verify settings.py

**Location**: `docchat/config/settings.py`

```python
from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    # These fields automatically load from .env
    AZURE_OPENAI_API_KEY: str = ""
    AZURE_OPENAI_ENDPOINT: str = ""
    AZURE_OPENAI_API_VERSION: str = "2024-12-01-preview"
    AZURE_OPENAI_DEPLOYMENT_NAME: str = "gpt-4"
    AZURE_OPENAI_MODEL_NAME: str = "gpt-4"
    
    class Config:
        env_file = ".env"  # 👈 This line enables .env loading
        env_file_encoding = "utf-8"

# Global singleton instance
settings = Settings()
```

**How it works:**
- `BaseSettings` from `pydantic_settings` automatically reads `.env`
- All environment variables matching field names are loaded
- Type validation ensures correctness
- Default values are used if env var not found

### Step 3: Agents Use Settings

All three agents load credentials the same way:

#### Research Agent
```python
# docchat/agents/research_agent.py
from docchat.config.settings import settings

class ResearchAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,        # ✅ From .env
            api_version=settings.AZURE_OPENAI_API_VERSION, # ✅ From .env
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT, # ✅ From .env
        )
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT_NAME
```

#### Verification Agent
```python
# docchat/agents/verification_agent.py
from docchat.config.settings import settings

class VerificationAgent:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,        # ✅ From .env
            api_version=settings.AZURE_OPENAI_API_VERSION, # ✅ From .env
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT, # ✅ From .env
        )
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT_NAME
```

#### Relevance Checker
```python
# docchat/agents/relevance_checker.py
from docchat.config.settings import settings

class RelevanceChecker:
    def __init__(self):
        self.client = AzureOpenAI(
            api_key=settings.AZURE_OPENAI_API_KEY,        # ✅ From .env
            api_version=settings.AZURE_OPENAI_API_VERSION, # ✅ From .env
            azure_endpoint=settings.AZURE_OPENAI_ENDPOINT, # ✅ From .env
        )
        self.deployment = settings.AZURE_OPENAI_DEPLOYMENT_NAME
```

---

## ✅ Verification Checklist

Before running the application, verify:

- [ ] `.env` file exists in project root (`doc_chat/.env`)
- [ ] All required variables are set:
  - [ ] `AZURE_OPENAI_API_KEY` (your Azure OpenAI key)
  - [ ] `AZURE_OPENAI_ENDPOINT` (your Azure endpoint URL)
  - [ ] `AZURE_OPENAI_API_VERSION` (API version)
  - [ ] `AZURE_OPENAI_DEPLOYMENT_NAME` (your deployment name)
  - [ ] `AZURE_OPENAI_MODEL_NAME` (model identifier)
- [ ] Endpoint URL ends with `/` (e.g., `https://....com/`)
- [ ] No extra spaces or quotes around values

---

## 🐛 Troubleshooting

### Error: "API key is required"

**Cause**: `.env` file not found or `AZURE_OPENAI_API_KEY` not set

**Solutions:**
1. Check `.env` is in the root directory (`doc_chat/.env`)
2. Verify the file is named exactly `.env` (not `.env.txt`)
3. Check that `AZURE_OPENAI_API_KEY` is set in `.env`:
   ```bash
   # View your .env file
   cat .env
   ```

### Error: "Invalid endpoint"

**Cause**: Malformed `AZURE_OPENAI_ENDPOINT`

**Solutions:**
1. Ensure endpoint ends with `/`:
   ```env
   # ✅ Correct
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
   
   # ❌ Wrong (missing trailing slash)
   AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com
   ```

2. Use full URL including `https://`

### Error: "Deployment not found"

**Cause**: `AZURE_OPENAI_DEPLOYMENT_NAME` doesn't match your Azure deployment

**Solution:**
1. Log into Azure Portal
2. Go to your OpenAI resource
3. Navigate to "Deployments"
4. Copy the exact deployment name
5. Update `.env`:
   ```env
   AZURE_OPENAI_DEPLOYMENT_NAME=your-exact-deployment-name
   ```

### Error: "Settings not loading"

**Cause**: `python-dotenv` or `pydantic-settings` not installed

**Solution:**
```bash
pip install python-dotenv pydantic-settings pydantic
```

---

## 🔒 Security Best Practices

### ✅ DO:
- Keep `.env` in `.gitignore` (never commit to Git)
- Use environment-specific `.env` files (`.env.dev`, `.env.prod`)
- Rotate API keys regularly
- Use read-only keys when possible

### ❌ DON'T:
- Commit `.env` to version control
- Share `.env` files via email/chat
- Hardcode API keys in source code
- Use production keys in development

---

## 📝 Example .env Template

Save this as `.env.template` (commit to Git as a template):

```env
# Azure OpenAI Configuration
# Get these values from Azure Portal → Your OpenAI Resource

AZURE_OPENAI_API_KEY=your_api_key_here
AZURE_OPENAI_ENDPOINT=https://your-resource-name.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-12-01-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
AZURE_OPENAI_MODEL_NAME=gpt-4

# Optional: Override default settings
# VECTOR_SEARCH_K=10
# CACHE_EXPIRE_DAYS=7
# MAX_TOTAL_SIZE=209715200
```

Then each developer creates their own `.env` from this template.

---

## 🎯 Testing Your Configuration

Run this Python script to test if your configuration loads correctly:

```python
# test_config.py
from docchat.config.settings import settings

print("Configuration Test")
print("=" * 50)
print(f"API Key: {settings.AZURE_OPENAI_API_KEY[:10]}...{settings.AZURE_OPENAI_API_KEY[-5:]}")
print(f"Endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
print(f"API Version: {settings.AZURE_OPENAI_API_VERSION}")
print(f"Deployment: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
print(f"Model: {settings.AZURE_OPENAI_MODEL_NAME}")
print("=" * 50)

if settings.AZURE_OPENAI_API_KEY:
    print("✅ Configuration loaded successfully!")
else:
    print("❌ API key not found. Check your .env file.")
```

Run it:
```bash
python test_config.py
```

Expected output:
```
Configuration Test
==================================================
API Key: BzgYZeEHRK...bhCp
Endpoint: https://subha-mafdk4x5-eastus2.cognitiveservices.azure.com/
API Version: 2024-12-01-preview
Deployment: gpt-4
Model: gpt-4
==================================================
✅ Configuration loaded successfully!
```

---

## 📚 Additional Resources

- [Azure OpenAI Documentation](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/)
- [Pydantic Settings Documentation](https://docs.pydantic.dev/latest/usage/pydantic_settings/)
- [python-dotenv Documentation](https://github.com/theskumar/python-dotenv)

---

## 🆘 Need Help?

If you're still having issues:

1. Check the main [README.md](README.md) for general setup
2. Review [TECHNICAL_DOCUMENTATION.md](TECHNICAL_DOCUMENTATION.md) for detailed explanations
3. Verify all dependencies are installed:
   ```bash
   pip install -r requirements_docchat.txt
   ```
4. Check application logs for specific error messages

---

**Last Updated**: March 2026  
**Maintained by**: DocChat Development Team
