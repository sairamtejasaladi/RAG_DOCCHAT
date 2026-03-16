"""
Configuration Test Script for DocChat

Run this script to verify that your .env file is correctly configured
and that Azure OpenAI credentials are properly loaded.

Usage:
    python test_config.py
"""
import sys
import os

# Add project root to path
ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

def test_configuration():
    """Test if configuration is properly loaded from .env file."""
    
    print("\n" + "=" * 70)
    print("DocChat Configuration Test")
    print("=" * 70 + "\n")
    
    # Test 1: Import settings
    print("Test 1: Importing settings module...")
    try:
        from docchat.config.settings import settings
        print("✅ Settings module imported successfully\n")
    except Exception as e:
        print(f"❌ Failed to import settings: {e}\n")
        return False
    
    # Test 2: Check .env file exists
    print("Test 2: Checking .env file...")
    env_path = os.path.join(ROOT, ".env")
    if os.path.exists(env_path):
        print(f"✅ .env file found at: {env_path}\n")
    else:
        print(f"❌ .env file NOT found at: {env_path}")
        print("   Please create a .env file in the project root.\n")
        return False
    
    # Test 3: Check API Key
    print("Test 3: Validating Azure OpenAI API Key...")
    if settings.AZURE_OPENAI_API_KEY:
        # Mask the key for security
        key = settings.AZURE_OPENAI_API_KEY
        masked_key = f"{key[:10]}...{key[-5:]}" if len(key) > 15 else "***"
        print(f"✅ API Key loaded: {masked_key}\n")
    else:
        print("❌ AZURE_OPENAI_API_KEY is empty or not set\n")
        return False
    
    # Test 4: Check Endpoint
    print("Test 4: Validating Azure OpenAI Endpoint...")
    if settings.AZURE_OPENAI_ENDPOINT:
        print(f"✅ Endpoint: {settings.AZURE_OPENAI_ENDPOINT}")
        
        # Check endpoint format
        if settings.AZURE_OPENAI_ENDPOINT.startswith("https://"):
            print("   ✅ Endpoint starts with https://")
        else:
            print("   ⚠️  Warning: Endpoint should start with https://")
        
        if settings.AZURE_OPENAI_ENDPOINT.endswith("/"):
            print("   ✅ Endpoint ends with /")
        else:
            print("   ⚠️  Warning: Endpoint should end with /")
        print()
    else:
        print("❌ AZURE_OPENAI_ENDPOINT is empty or not set\n")
        return False
    
    # Test 5: Check API Version
    print("Test 5: Validating API Version...")
    if settings.AZURE_OPENAI_API_VERSION:
        print(f"✅ API Version: {settings.AZURE_OPENAI_API_VERSION}\n")
    else:
        print("⚠️  AZURE_OPENAI_API_VERSION not set (using default)\n")
    
    # Test 6: Check Deployment Name
    print("Test 6: Validating Deployment Name...")
    if settings.AZURE_OPENAI_DEPLOYMENT_NAME:
        print(f"✅ Deployment Name: {settings.AZURE_OPENAI_DEPLOYMENT_NAME}\n")
    else:
        print("❌ AZURE_OPENAI_DEPLOYMENT_NAME is empty or not set\n")
        return False
    
    # Test 7: Check Model Name
    print("Test 7: Validating Model Name...")
    if settings.AZURE_OPENAI_MODEL_NAME:
        print(f"✅ Model Name: {settings.AZURE_OPENAI_MODEL_NAME}\n")
    else:
        print("⚠️  AZURE_OPENAI_MODEL_NAME not set (using default)\n")
    
    # Test 8: Try to initialize agents
    print("Test 8: Testing agent initialization...")
    try:
        from docchat.agents.research_agent import ResearchAgent
        from docchat.agents.verification_agent import VerificationAgent
        from docchat.agents.relevance_checker import RelevanceChecker
        
        research = ResearchAgent()
        print("   ✅ ResearchAgent initialized")
        
        verification = VerificationAgent()
        print("   ✅ VerificationAgent initialized")
        
        relevance = RelevanceChecker()
        print("   ✅ RelevanceChecker initialized")
        print()
    except Exception as e:
        print(f"   ❌ Failed to initialize agents: {e}\n")
        return False
    
    # Summary
    print("=" * 70)
    print("Configuration Summary:")
    print("=" * 70)
    print(f"API Key:         {masked_key}")
    print(f"Endpoint:        {settings.AZURE_OPENAI_ENDPOINT}")
    print(f"API Version:     {settings.AZURE_OPENAI_API_VERSION}")
    print(f"Deployment:      {settings.AZURE_OPENAI_DEPLOYMENT_NAME}")
    print(f"Model:           {settings.AZURE_OPENAI_MODEL_NAME}")
    print(f"Vector Search K: {settings.VECTOR_SEARCH_K}")
    print(f"Cache Directory: {settings.CACHE_DIR}")
    print(f"Cache Expiry:    {settings.CACHE_EXPIRE_DAYS} days")
    print("=" * 70 + "\n")
    
    print("🎉 All configuration tests passed!")
    print("✅ Your DocChat setup is ready to use.\n")
    print("Next steps:")
    print("  1. Run the application: python run_docchat.py")
    print("  2. Open browser to: http://127.0.0.1:7860")
    print("  3. Upload documents and start asking questions!\n")
    
    return True


if __name__ == "__main__":
    try:
        success = test_configuration()
        sys.exit(0 if success else 1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}\n")
        import traceback
        traceback.print_exc()
        sys.exit(1)
