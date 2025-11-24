#!/usr/bin/env python3
"""
Simple test script to diagnose orchestrator issues
"""
import os
from dotenv import load_dotenv

load_dotenv()

def test_env_vars():
    """Test if required environment variables are set"""
    print("=== Environment Variables Test ===")
    
    openai_key = os.getenv("OPENAI_API_KEY")
    llm_model = os.getenv("LLM_MODEL", "gpt-4")
    
    print(f"OPENAI_API_KEY: {'✓ Set' if openai_key else '✗ Not set'}")
    print(f"LLM_MODEL: {llm_model}")
    
    return bool(openai_key)

def test_simple_classification():
    """Test simple classification without tools"""
    print("\n=== Simple Classification Test ===")
    
    try:
        from langchain_openai import ChatOpenAI
        from langchain_core.messages import HumanMessage, SystemMessage
        
        model = ChatOpenAI(
            model=os.getenv("LLM_MODEL", "gpt-4"),
            temperature=0,
            timeout=10  # Add timeout to prevent hanging
        )
        
        messages = [
            SystemMessage(content="You are a classifier. Respond with only: TechAgent, HRAgent, or FinanceAgent"),
            HumanMessage(content="How long are annual leaves?")
        ]
        
        print("Making API call...")
        result = model.invoke(messages)
        print(f"Result: {result.content.strip()}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

def test_agent_creation():
    """Test agent creation without API calls"""
    print("\n=== Agent Creation Test ===")
    
    try:
        from agents.hr_agent import HRAgent
        
        print("Creating HRAgent...")
        agent = HRAgent("Test query")
        print(f"Agent created: {agent.name}")
        
        print("Running agent...")
        result = agent.run()
        print(f"Agent result: {result}")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    print("Orchestrator Diagnostic Test")
    print("=" * 40)
    
    # Test 1: Environment variables
    env_ok = test_env_vars()
    
    # Test 2: Agent creation (no API)
    agent_ok = test_agent_creation()
    
    # Test 3: Simple API call (only if env vars are set)
    if env_ok:
        api_ok = test_simple_classification()
    else:
        print("\n⚠️  Skipping API test - OPENAI_API_KEY not set")
        api_ok = False
    
    print("\n" + "=" * 40)
    print("SUMMARY:")
    print(f"Environment: {'✓' if env_ok else '✗'}")
    print(f"Agent Creation: {'✓' if agent_ok else '✗'}")
    print(f"API Calls: {'✓' if api_ok else '✗'}")
    
    if not env_ok:
        print("\n💡 Solution: Set OPENAI_API_KEY in your .env file")
    elif not api_ok:
        print("\n💡 Solution: Check your API key and network connection")
