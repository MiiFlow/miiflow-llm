#!/usr/bin/env python3
"""
Manual test script for ReAct mode.
Mirrors examples/notebooks/react_tutorial.ipynb

Usage:
    # OpenAI (default)
    export OPENAI_API_KEY="sk-..."
    python tests/manual/test_react_manual.py

    # Google Gemini
    export GOOGLE_API_KEY="..."
    python tests/manual/test_react_manual.py --provider gemini

    # Anthropic Claude
    export ANTHROPIC_API_KEY="sk-ant-..."
    python tests/manual/test_react_manual.py --provider anthropic

    # Run all providers
    python tests/manual/test_react_manual.py --provider all

This script uses the same tools and queries as the notebook to ensure
that if this test passes, the notebook will work correctly.
"""

import argparse
import asyncio
import math
import os
import sys

import yfinance as yf

from miiflow_llm import Agent, AgentType, LLMClient, RunContext
from miiflow_llm.core.react import ReActEventType
from miiflow_llm.core.tools import tool


# =============================================================================
# Provider Configuration
# =============================================================================

PROVIDERS = {
    "openai": {
        "provider": "openai",
        "model": "gpt-4.1",
        "env_var": "OPENAI_API_KEY",
    },
    "gemini": {
        "provider": "gemini",
        "model": "gemini-2.5-flash",
        "env_var": "GOOGLE_API_KEY",
    },
    "anthropic": {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "env_var": "ANTHROPIC_API_KEY",
    },
}


# =============================================================================
# Tool Definitions (exact copy from react_tutorial.ipynb)
# =============================================================================


@tool("get_stock_quote", "Get real-time stock quote and key metrics for a symbol")
def get_stock_quote(symbol: str) -> str:
    """Fetch current stock price and basic metrics.

    Args:
        symbol: Stock ticker symbol (e.g., 'AAPL', 'MSFT', 'GOOGL')
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        if not info or "regularMarketPrice" not in info:
            return f"Unable to fetch data for '{symbol}'. Please verify the ticker symbol."

        price = info.get("regularMarketPrice", "N/A")
        prev_close = info.get("regularMarketPreviousClose", "N/A")

        if isinstance(price, (int, float)) and isinstance(prev_close, (int, float)):
            change = price - prev_close
            change_pct = (change / prev_close) * 100
            change_str = f"${change:+.2f} ({change_pct:+.2f}%)"
        else:
            change_str = "N/A"

        # Format market cap
        market_cap = info.get("marketCap", 0)
        if market_cap >= 1e12:
            cap_str = f"${market_cap/1e12:.2f}T"
        elif market_cap >= 1e9:
            cap_str = f"${market_cap/1e9:.2f}B"
        else:
            cap_str = f"${market_cap:,.0f}"

        # Format P/E ratio
        pe_ratio = info.get('trailingPE')
        pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"

        # Format 52-week range
        week_low = info.get('fiftyTwoWeekLow')
        week_high = info.get('fiftyTwoWeekHigh')
        week_low_str = f"${week_low:.2f}" if isinstance(week_low, (int, float)) else "N/A"
        week_high_str = f"${week_high:.2f}" if isinstance(week_high, (int, float)) else "N/A"

        return f"""Stock Quote for {info.get('shortName', symbol)} ({symbol.upper()}):
- Current Price: ${price:.2f}
- Change: {change_str}
- Market Cap: {cap_str}
- P/E Ratio: {pe_str}
- 52 Week Range: {week_low_str} - {week_high_str}"""
    except Exception as e:
        return f"Error fetching quote for {symbol}: {str(e)}"


@tool("get_stock_history", "Get historical stock price data")
def get_stock_history(symbol: str, period: str = "1mo") -> str:
    """Fetch historical price data.

    Args:
        symbol: Stock ticker symbol
        period: Time period - 1d, 5d, 1mo, 3mo, 6mo, 1y, ytd, max
    """
    try:
        ticker = yf.Ticker(symbol.upper())
        hist = ticker.history(period=period)

        if hist.empty:
            return f"No historical data for {symbol}"

        start_price = hist["Close"].iloc[0]
        end_price = hist["Close"].iloc[-1]
        change = end_price - start_price
        change_pct = (change / start_price) * 100

        return f"""Historical Data for {symbol.upper()} ({period}):
- Period: {hist.index[0].strftime('%Y-%m-%d')} to {hist.index[-1].strftime('%Y-%m-%d')}
- Starting Price: ${start_price:.2f}
- Ending Price: ${end_price:.2f}
- Change: ${change:+.2f} ({change_pct:+.2f}%)
- Period High: ${hist['High'].max():.2f}
- Period Low: ${hist['Low'].min():.2f}"""
    except Exception as e:
        return f"Error: {str(e)}"


@tool("calculate", "Evaluate mathematical expressions")
def calculate(expression: str) -> str:
    """Safely evaluate a math expression.

    Args:
        expression: A mathematical expression like '2 + 2' or 'sqrt(16)'
    """
    allowed = {
        "abs": abs,
        "round": round,
        "min": min,
        "max": max,
        "sqrt": math.sqrt,
        "pow": pow,
        "pi": math.pi,
        "e": math.e,
    }
    try:
        result = eval(expression, {"__builtins__": {}}, allowed)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


# =============================================================================
# Test Infrastructure
# =============================================================================


def check_environment(provider_name: str) -> bool:
    """Check that required environment variables are set for the provider."""
    config = PROVIDERS[provider_name]
    env_var = config["env_var"]
    api_key = os.environ.get(env_var)

    if not api_key:
        print(f"SKIP: {env_var} not set for {provider_name}")
        return False

    print(f"Environment check passed: {env_var} is set")
    return True


def create_agent(provider_name: str):
    """Create the ReAct agent with tools for the specified provider."""
    config = PROVIDERS[provider_name]

    client = LLMClient.create(
        config["provider"],
        model=config["model"],
        api_key=os.environ.get(config["env_var"]),
    )

    agent = Agent(
        client,
        agent_type=AgentType.REACT,
        max_iterations=10,
        system_prompt="You are a helpful financial assistant. Use tools to answer questions about stocks.",
    )

    agent.add_tool(get_stock_quote)
    agent.add_tool(get_stock_history)
    agent.add_tool(calculate)

    return agent


# =============================================================================
# Test Cases (matching notebook examples)
# =============================================================================


async def test_simple_stock_lookup(agent):
    """Test 1: Simple stock lookup (notebook cell 10)"""
    print("\n" + "=" * 60)
    print("TEST 1: Simple Stock Lookup")
    print("Query: What is the current price of Apple stock (AAPL)?")
    print("=" * 60)

    result = await agent.run("What is the current price of Apple stock (AAPL)?")

    print(f"\nResult:\n{result.data}")

    # Validate result contains expected content
    assert result.data is not None, "Result should not be None"
    assert len(result.data) > 50, "Result should contain substantial content"
    # Check for price-related content (flexible matching)
    result_lower = result.data.lower()
    assert any(
        word in result_lower for word in ["price", "aapl", "apple", "$"]
    ), "Result should mention price or AAPL"

    print("\nPASSED")


async def test_multi_step_reasoning(agent):
    """Test 2: Multi-step reasoning (notebook cell 12)"""
    print("\n" + "=" * 60)
    print("TEST 2: Multi-Step Reasoning")
    print("Query: Analyze Microsoft (MSFT) - price, monthly performance, percentage change")
    print("=" * 60)

    result = await agent.run("""
I want to analyze Microsoft (MSFT):
1. What is the current price?
2. How has it performed over the past month?
3. What's the percentage change?
""")

    print(f"\nResult:\n{result.data}")

    assert result.data is not None, "Result should not be None"
    assert len(result.data) > 100, "Multi-step result should be detailed"
    result_lower = result.data.lower()
    assert any(
        word in result_lower for word in ["msft", "microsoft", "price"]
    ), "Result should mention MSFT"

    print("\nPASSED")


async def test_stock_comparison(agent):
    """Test 3: Stock comparison (notebook cell 14)"""
    print("\n" + "=" * 60)
    print("TEST 3: Stock Comparison")
    print("Query: Compare Apple (AAPL) and Microsoft (MSFT)")
    print("=" * 60)

    result = await agent.run("""
Compare Apple (AAPL) and Microsoft (MSFT):
- Which has a higher market cap?
- Which has a better P/E ratio?
Give me a brief comparison.
""")

    print(f"\nResult:\n{result.data}")

    assert result.data is not None, "Result should not be None"
    result_lower = result.data.lower()
    # Should mention both stocks
    assert "aapl" in result_lower or "apple" in result_lower, "Should mention Apple"
    assert "msft" in result_lower or "microsoft" in result_lower, "Should mention Microsoft"

    print("\nPASSED")


async def test_calculations_with_data(agent):
    """Test 4: Calculations with real data (notebook cell 16)"""
    print("\n" + "=" * 60)
    print("TEST 4: Calculations with Real Data")
    print("Query: $10,000 investment in NVIDIA - how many shares?")
    print("=" * 60)

    result = await agent.run("""
If I have $10,000 to invest in NVIDIA (NVDA):
1. What's the current price per share?
2. How many whole shares could I buy?
3. How much money would be left over?
""")

    print(f"\nResult:\n{result.data}")

    assert result.data is not None, "Result should not be None"
    result_lower = result.data.lower()
    assert "nvda" in result_lower or "nvidia" in result_lower, "Should mention NVIDIA"
    # Should have some numbers in the result (shares, money)
    assert any(char.isdigit() for char in result.data), "Should contain numerical results"

    print("\nPASSED")


async def test_streaming(agent):
    """Test 5: Streaming events (notebook cell 18)"""
    print("\n" + "=" * 60)
    print("TEST 5: Streaming Events")
    print("Query: What's Tesla's (TSLA) current price and weekly performance?")
    print("=" * 60)

    context = RunContext(deps=None, messages=[])
    events_received = []
    final_answer = None

    async for event in agent.stream(
        "What's Tesla's (TSLA) current price and how has it performed this week?", context
    ):
        events_received.append(event.event_type)

        if event.event_type == ReActEventType.THINKING_CHUNK:
            print(event.data.get("delta", ""), end="", flush=True)
        elif event.event_type == ReActEventType.ACTION_PLANNED:
            action = event.data.get("action", "")
            print(f"\n[Calling tool: {action}]")
        elif event.event_type == ReActEventType.OBSERVATION:
            obs = str(event.data.get("observation", ""))[:100]
            print(f"[Tool result: {obs}...]")
        elif event.event_type == ReActEventType.FINAL_ANSWER:
            final_answer = event.data.get("answer", "")
            print(f"\n\nFINAL ANSWER:\n{final_answer}")

    # Validate streaming worked
    assert len(events_received) > 0, "Should receive streaming events"
    assert ReActEventType.FINAL_ANSWER in events_received, "Should receive FINAL_ANSWER event"
    assert final_answer is not None, "Should have final answer"

    print("\nPASSED")


async def test_error_handling(agent):
    """Test 6: Error handling (notebook cell 20)"""
    print("\n" + "=" * 60)
    print("TEST 6: Error Handling")
    print("Query: What is the current price of INVALIDXYZ stock?")
    print("=" * 60)

    result = await agent.run("What is the current price of INVALIDXYZ stock?")

    print(f"\nResult:\n{result.data}")

    assert result.data is not None, "Result should not be None"
    # Agent should gracefully handle the error
    result_lower = result.data.lower()
    assert any(
        word in result_lower for word in ["unable", "error", "invalid", "not found", "couldn't", "cannot"]
    ), "Should indicate error or inability to find data"

    print("\nPASSED")


# =============================================================================
# Main Runner
# =============================================================================


async def run_tests_for_provider(provider_name: str) -> tuple[int, int, list]:
    """Run all tests for a specific provider.

    Returns:
        Tuple of (passed, failed, failures list)
    """
    config = PROVIDERS[provider_name]
    print("\n" + "=" * 60)
    print(f"Testing with: {provider_name.upper()} ({config['model']})")
    print("=" * 60)

    if not check_environment(provider_name):
        return 0, 0, []  # Skipped

    print(f"\nCreating agent with {provider_name}...")
    agent = create_agent(provider_name)
    print("Agent created with tools: get_stock_quote, get_stock_history, calculate")

    tests = [
        ("Simple Stock Lookup", test_simple_stock_lookup),
        ("Multi-Step Reasoning", test_multi_step_reasoning),
        ("Stock Comparison", test_stock_comparison),
        ("Calculations with Data", test_calculations_with_data),
        ("Streaming", test_streaming),
        ("Error Handling", test_error_handling),
    ]

    passed = 0
    failed = 0
    failures = []

    for name, test_fn in tests:
        try:
            await test_fn(agent)
            passed += 1
        except AssertionError as e:
            failed += 1
            failures.append((f"{provider_name}/{name}", str(e)))
            print(f"\nFAILED: {e}")
        except Exception as e:
            failed += 1
            failures.append((f"{provider_name}/{name}", f"Exception: {e}"))
            print(f"\nFAILED with exception: {e}")

    return passed, failed, failures


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manual test script for ReAct mode with multiple providers"
    )
    parser.add_argument(
        "--provider",
        "-p",
        choices=["openai", "gemini", "anthropic", "all"],
        default="openai",
        help="LLM provider to test (default: openai)",
    )
    return parser.parse_args()


async def main():
    """Run all tests and report results."""
    args = parse_args()

    print("=" * 60)
    print("ReAct Agent Manual Test Suite")
    print("Mirrors: examples/notebooks/react_tutorial.ipynb")
    print("=" * 60)

    # First test the tools directly (provider-independent)
    print("\n" + "-" * 60)
    print("Pre-flight: Testing yfinance connection...")
    tool_result = get_stock_quote("AAPL")
    if "Error" in tool_result or "Unable" in tool_result:
        print(f"WARNING: Tool test returned: {tool_result[:100]}")
        print("yfinance may be having issues. Tests may fail.")
    else:
        print("yfinance connection OK")
    print("-" * 60)

    # Determine which providers to test
    if args.provider == "all":
        providers_to_test = list(PROVIDERS.keys())
    else:
        providers_to_test = [args.provider]

    total_passed = 0
    total_failed = 0
    total_skipped = 0
    all_failures = []

    for provider_name in providers_to_test:
        passed, failed, failures = await run_tests_for_provider(provider_name)
        if passed == 0 and failed == 0:
            total_skipped += 1
        else:
            total_passed += passed
            total_failed += failed
            all_failures.extend(failures)

    # Final Summary
    print("\n" + "=" * 60)
    print("FINAL TEST SUMMARY")
    print("=" * 60)
    print(f"Providers tested: {len(providers_to_test) - total_skipped}/{len(providers_to_test)}")
    print(f"Total Passed: {total_passed}")
    print(f"Total Failed: {total_failed}")

    if all_failures:
        print("\nAll Failures:")
        for name, error in all_failures:
            print(f"  - {name}: {error}")

    print("=" * 60)

    sys.exit(0 if total_failed == 0 else 1)


if __name__ == "__main__":
    asyncio.run(main())
