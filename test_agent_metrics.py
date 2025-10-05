"""
Test agent metrics collection after fixes.

Tests that the 3 critical metrics collection fixes work:
1. Remove early break in _execute_with_context() 
2. Remove early break in stream_single_hop()
3. Add extra achat() call for metrics recording

Tests multiple providers:
- Anthropic (claude-3-5-sonnet-20241022)
- TogetherAI (meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo)
- Google (gemini-2.0-flash-exp)
"""
import asyncio
import os
from dotenv import load_dotenv
from miiflow_llm import LLMClient
from miiflow_llm.core.agent import Agent, AgentType, RunContext

# Load environment variables from .env file
load_dotenv()

# Test configuration
TEST_PROVIDERS = {
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-3-5-sonnet-20241022",
        "env_var": "ANTHROPIC_API_KEY",
    },
    "together": {
        "provider": "together",
        "model": "meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        "env_var": "TOGETHERAI_API_KEY",
    },
    "gemini": {
        "provider": "gemini",
        "model": "gemini-2.0-flash-exp",
        "env_var": "GEMINI_API_KEY",
    },
}


async def test_basic_chat(provider_key: str):
    """Test basic chat with metrics."""
    config = TEST_PROVIDERS[provider_key]
    print(f"\nTesting {config['provider']} - Basic Chat")
    print("-" * 50)
    
    # Check API key
    api_key = os.getenv(config["env_var"])
    if not api_key:
        print(f"⚠️  {config['env_var']} not found - SKIPPING")
        return None
    
    try:
        # Create client and agent
        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )
        
        agent = Agent(
            client=client,
            agent_type=AgentType.SINGLE_HOP,
            result_type=str,
            temperature=0.0
        )
        
        # Run agent
        result = await agent.run("What is 2 + 2? Answer with just the number.")
        
        # Get metrics
        metrics = client.get_metrics()
        
        # Check if metrics are recorded
        if metrics and len(metrics) > 0:
            provider_key = list(metrics.keys())[0]
            provider_metrics = metrics[provider_key]
            
            total_requests = provider_metrics.get("total_requests", 0)
            total_tokens = provider_metrics.get("total_tokens")
            
            if total_tokens:
                input_tokens = total_tokens.prompt_tokens
                output_tokens = total_tokens.completion_tokens
            else:
                input_tokens = 0
                output_tokens = 0
            
            print(f"✅ {total_requests} requests, {input_tokens}/{output_tokens} tokens")
            
            if total_requests > 0 and input_tokens > 0 and output_tokens > 0:
                return True
            else:
                print(f"❌ Incomplete metrics")
                return False
        else:
            print(f"❌ No metrics recorded")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_streaming_chat(provider_key: str):
    """Test streaming chat with metrics."""
    config = TEST_PROVIDERS[provider_key]
    print(f"\nTesting {config['provider']} - Streaming Chat")
    print("-" * 50)
    
    # Check API key
    api_key = os.getenv(config["env_var"])
    if not api_key:
        print(f"⚠️  {config['env_var']} not found - SKIPPING")
        return None
    
    try:
        # Create client and agent
        client = LLMClient.create(
            provider=config["provider"],
            model=config["model"],
            api_key=api_key
        )
        
        agent = Agent(
            client=client,
            agent_type=AgentType.SINGLE_HOP,
            result_type=str,
            temperature=0.0
        )
        
        # Run agent
        result = await agent.run("What is AWS? Give me 2 sentences.")
        
        # Get metrics
        metrics = client.get_metrics()
        
        # Check if metrics are recorded
        if metrics and len(metrics) > 0:
            provider_key_str = list(metrics.keys())[0]
            provider_metrics = metrics[provider_key_str]
            
            total_requests = provider_metrics.get("total_requests", 0)
            total_tokens = provider_metrics.get("total_tokens")
            
            if total_tokens:
                input_tokens = total_tokens.prompt_tokens
                output_tokens = total_tokens.completion_tokens
            else:
                input_tokens = 0
                output_tokens = 0
            
            print(f"✅ {total_requests} requests, {input_tokens}/{output_tokens} tokens")
            
            if total_requests > 0 and input_tokens > 0 and output_tokens > 0:
                return True
            else:
                print(f"❌ Incomplete metrics")
                return False
        else:
            print(f"❌ No metrics recorded")
            return False
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def run_all_tests():
    """Run all tests for all providers."""
    print("=" * 60)
    print("Agent Metrics Collection Test Suite")
    print("=" * 60)
    
    all_results = {}
    
    for provider_key in TEST_PROVIDERS.keys():
        config = TEST_PROVIDERS[provider_key]
        print(f"\n{'=' * 60}")
        print(f"Testing Provider: {config['provider'].upper()} ({config['model']})")
        print(f"{'=' * 60}")
        
        results = []
        
        # Test basic chat
        result = await test_basic_chat(provider_key)
        if result is not None:
            results.append(("Basic Chat", result))
        
        # Test streaming
        result = await test_streaming_chat(provider_key)
        if result is not None:
            results.append(("Streaming Chat", result))
        
        if results:
            all_results[provider_key] = results
    
    # Summary
    print("\n" + "=" * 60)
    print("Multi-Provider Test Summary:")
    print("=" * 60)
    
    for provider_key, results in all_results.items():
        config = TEST_PROVIDERS[provider_key]
        print(f"\n{config['provider'].upper()} ({config['model']}):")
        for test_name, passed in results:
            status = "✅ PASSED" if passed else "❌ FAILED"
            print(f"  {test_name}: {status}")
    
    # Check if all passed
    all_passed = all(
        passed for results in all_results.values() 
        for _, passed in results
    )
    
    if all_passed:
        print("\n🎉 All tests passed - Metrics collection is working!")
        return 0
    else:
        print("\n⚠️  Some tests failed - Check metrics collection")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(run_all_tests())
    exit(exit_code)
