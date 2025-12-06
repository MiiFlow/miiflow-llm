#!/usr/bin/env python3
"""
Manual test script for Plan & Execute mode.
Mirrors examples/notebooks/plan_execute_tutorial.ipynb

Usage:
    # OpenAI (default)
    export OPENAI_API_KEY="sk-..."
    python tests/manual/test_plan_execute_manual.py

    # Google Gemini
    export GOOGLE_API_KEY="..."
    python tests/manual/test_plan_execute_manual.py --provider gemini

    # Anthropic Claude
    export ANTHROPIC_API_KEY="sk-ant-..."
    python tests/manual/test_plan_execute_manual.py --provider anthropic

    # Run all providers
    python tests/manual/test_plan_execute_manual.py --provider all

This script uses the same tools and queries as the notebook to ensure
that if this test passes, the notebook will work correctly.
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime

import yfinance as yf

from miiflow_llm import Agent, AgentType, LLMClient, RunContext
from miiflow_llm.core.react import PlanExecuteEventType
from miiflow_llm.core.tools import tool


# =============================================================================
# Provider Configuration
# =============================================================================

PROVIDERS = {
    "openai": {
        "provider": "openai",
        "model": "gpt-5.1",
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
# Tool Definitions (exact copy from plan_execute_tutorial.ipynb)
# =============================================================================


@tool("get_stock_quote", "Get real-time stock quote for a symbol")
def get_stock_quote(symbol: str) -> str:
    """Fetch current stock price and metrics."""
    try:
        ticker = yf.Ticker(symbol.upper())
        info = ticker.info

        if not info or "regularMarketPrice" not in info:
            return f"Unable to fetch data for '{symbol}'"

        price = info.get("regularMarketPrice", "N/A")
        prev = info.get("regularMarketPreviousClose", price)
        change = (
            price - prev
            if isinstance(price, (int, float)) and isinstance(prev, (int, float))
            else 0
        )
        change_pct = (change / prev * 100) if prev else 0

        cap = info.get("marketCap", 0)
        cap_str = (
            f"${cap/1e12:.2f}T"
            if cap >= 1e12
            else f"${cap/1e9:.2f}B" if cap >= 1e9 else f"${cap:,.0f}"
        )

        # Format P/E ratio
        pe_ratio = info.get('trailingPE')
        pe_str = f"{pe_ratio:.2f}" if isinstance(pe_ratio, (int, float)) else "N/A"

        # Format 52-week range
        week_low = info.get('fiftyTwoWeekLow', 0)
        week_high = info.get('fiftyTwoWeekHigh', 0)
        week_low_str = f"${week_low:.2f}" if isinstance(week_low, (int, float)) else "N/A"
        week_high_str = f"${week_high:.2f}" if isinstance(week_high, (int, float)) else "N/A"

        return f"""{info.get('shortName', symbol)} ({symbol.upper()}):
Price: ${price:.2f} ({change:+.2f}, {change_pct:+.2f}%)
Market Cap: {cap_str}
P/E: {pe_str}
52W Range: {week_low_str} - {week_high_str}"""
    except Exception as e:
        return f"Error: {str(e)}"


@tool("get_stock_history", "Get historical price data")
def get_stock_history(symbol: str, period: str = "1mo") -> str:
    """Fetch historical data. Period: 1d, 5d, 1mo, 3mo, 6mo, 1y"""
    try:
        hist = yf.Ticker(symbol.upper()).history(period=period)
        if hist.empty:
            return f"No data for {symbol}"

        start, end = hist["Close"].iloc[0], hist["Close"].iloc[-1]
        change = end - start
        change_pct = (change / start) * 100

        return f"""{symbol.upper()} ({period}):
Start: ${start:.2f} -> End: ${end:.2f}
Change: {change:+.2f} ({change_pct:+.2f}%)
High: ${hist['High'].max():.2f}, Low: ${hist['Low'].min():.2f}"""
    except Exception as e:
        return f"Error: {str(e)}"


@tool("get_company_info", "Get company profile information")
def get_company_info(symbol: str) -> str:
    """Fetch company profile."""
    try:
        info = yf.Ticker(symbol.upper()).info
        employees = info.get("fullTimeEmployees", "N/A")
        employees_str = f"{employees:,}" if isinstance(employees, int) else str(employees)
        return f"""{info.get('longName', symbol)} ({symbol.upper()}):
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
Employees: {employees_str}
Summary: {info.get('longBusinessSummary', 'N/A')[:300]}..."""
    except Exception as e:
        return f"Error: {str(e)}"


@tool("analyze_stocks", "Analyze and compare stock data")
def analyze_stocks(data: str, analysis_type: str = "comparison") -> str:
    """Analyze stock data. Types: comparison, trend, risk"""
    return f"""Stock Analysis ({analysis_type}):
Based on the provided data:
- Market trends show mixed signals
- Volatility is within normal ranges
- Sector performance varies

Key observations from data:
{data[:200]}...

Note: This is simulated analysis for demonstration."""


@tool("generate_report", "Generate a formatted investment report")
def generate_report(title: str, content: str) -> str:
    """Generate a formatted report."""
    return f"""
{'='*50}
{title.upper()}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
{'='*50}

{content}

{'='*50}
DISCLAIMER: For demonstration purposes only.
{'='*50}
"""


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
    """Create the Plan & Execute agent with tools for the specified provider."""
    config = PROVIDERS[provider_name]

    client = LLMClient.create(
        config["provider"],
        model=config["model"],
        api_key=os.environ.get(config["env_var"]),
    )

    agent = Agent(
        client,
        agent_type=AgentType.PLAN_AND_EXECUTE,
        max_iterations=15,
        system_prompt="You are a financial analyst. Create structured plans for research tasks.",
    )

    agent.add_tool(get_stock_quote)
    agent.add_tool(get_stock_history)
    agent.add_tool(get_company_info)
    agent.add_tool(analyze_stocks)
    agent.add_tool(generate_report)

    return agent


# =============================================================================
# Test Cases (matching notebook examples)
# =============================================================================


async def test_basic_plan_execute(agent):
    """Test 1: Basic Plan & Execute (notebook cell 8)"""
    print("\n" + "=" * 60)
    print("TEST 1: Basic Plan & Execute")
    print("Query: Research and compare AAPL, MSFT, and GOOGL")
    print("=" * 60)

    task = """
Research and compare these tech stocks: AAPL, MSFT, and GOOGL.

For each stock:
1. Get the current price and metrics
2. Check the monthly performance

Then provide a comparison summary.
"""

    result = await agent.run(task)

    print(f"\nResult:\n{result.data}")

    # Validate result
    assert result.data is not None, "Result should not be None"
    assert len(result.data) > 100, "Result should contain substantial content"
    result_lower = result.data.lower()
    # Should mention at least some of the stocks
    stocks_mentioned = sum(
        1 for stock in ["aapl", "apple", "msft", "microsoft", "googl", "google"]
        if stock in result_lower
    )
    assert stocks_mentioned >= 2, "Result should mention multiple stocks"

    print("\nPASSED")


async def test_streaming_events(agent):
    """Test 2: Streaming with event handling (notebook cell 10)"""
    print("\n" + "=" * 60)
    print("TEST 2: Streaming Events")
    print("Query: Compare Tesla (TSLA) and Ford (F)")
    print("=" * 60)

    task = """
Compare Tesla (TSLA) and Ford (F):
1. Get current prices for both
2. Get company profiles for both
3. Provide a comparison
"""

    context = RunContext(deps=None, messages=[])
    events_received = []
    final_answer = None
    planning_complete = False
    subtasks_started = 0

    async for event in agent.stream(task, context):
        event_type = event.event_type
        events_received.append(event_type)

        if event_type == PlanExecuteEventType.PLANNING_START:
            print("\n[PHASE 1: PLANNING]")
            print("Creating execution plan...")

        elif event_type == PlanExecuteEventType.PLANNING_COMPLETE:
            subtask_count = event.data.get("subtask_count", 0)
            print(f"\nPlan created with {subtask_count} subtasks!")
            print("\n[PHASE 2: EXECUTION]")
            planning_complete = True

        elif event_type == PlanExecuteEventType.SUBTASK_START:
            desc = event.data.get("description", "")[:60]
            print(f"\n  Subtask: {desc}...")
            subtasks_started += 1

        elif event_type == PlanExecuteEventType.SUBTASK_COMPLETE:
            result_preview = str(event.data.get("result", ""))[:80]
            print(f"  Done: {result_preview}...")

        elif event_type == PlanExecuteEventType.FINAL_ANSWER:
            final_answer = event.data.get("answer", "")
            print("\n" + "=" * 50)
            print("[PHASE 3: FINAL ANSWER]")
            print("=" * 50)
            print(final_answer[:500] + "..." if len(final_answer) > 500 else final_answer)

    # Validate streaming worked
    assert len(events_received) > 0, "Should receive streaming events"
    assert planning_complete, "Should complete planning phase"
    assert PlanExecuteEventType.FINAL_ANSWER in events_received, "Should receive FINAL_ANSWER"
    assert final_answer is not None, "Should have final answer"

    print("\nPASSED")


async def test_comprehensive_report(agent):
    """Test 3: Comprehensive research report (notebook cell 12)"""
    print("\n" + "=" * 60)
    print("TEST 3: Comprehensive Research Report")
    print("Query: Create investment report on AAPL, MSFT, NVDA")
    print("=" * 60)

    task = """
Create a comprehensive investment report on the "Magnificent 7" tech stocks.
Focus on: AAPL, MSFT, and NVDA (for time efficiency).

Your report should include:
1. Current price and market cap for each stock
2. Monthly performance trends
3. Brief company profiles
4. A final comparison and investment considerations

Generate a professional formatted report at the end.
"""

    print("Creating comprehensive report...")
    print("This may take a moment as the agent plans and executes multiple subtasks.\n")

    result = await agent.run(task)

    print("=" * 60)
    print("GENERATED REPORT")
    print("=" * 60)
    # Print first 1000 chars to avoid overwhelming output
    print(result.data[:1000] + "..." if len(result.data) > 1000 else result.data)

    # Validate result
    assert result.data is not None, "Result should not be None"
    assert len(result.data) > 200, "Comprehensive report should be detailed"
    result_lower = result.data.lower()
    # Should be a substantial report
    stocks_mentioned = sum(
        1 for stock in ["aapl", "apple", "msft", "microsoft", "nvda", "nvidia"]
        if stock in result_lower
    )
    assert stocks_mentioned >= 2, "Report should mention the requested stocks"

    print("\nPASSED")


async def test_single_stock_deep_dive(agent):
    """Test 4: Single stock deep dive (notebook cell 14)"""
    print("\n" + "=" * 60)
    print("TEST 4: Single Stock Deep Dive")
    print("Query: Deep dive analysis on NVIDIA (NVDA)")
    print("=" * 60)

    task = """
Perform a deep dive analysis on NVIDIA (NVDA):

1. Get current stock price and key metrics
2. Analyze 3-month price history
3. Review company profile and business
4. Provide investment considerations
"""

    result = await agent.run(task)

    print(f"\nResult:\n{result.data[:800]}..." if len(result.data) > 800 else result.data)

    # Validate result
    assert result.data is not None, "Result should not be None"
    assert len(result.data) > 100, "Deep dive should be detailed"
    result_lower = result.data.lower()
    assert "nvda" in result_lower or "nvidia" in result_lower, "Should mention NVIDIA"

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
    print(
        "Agent created with tools: get_stock_quote, get_stock_history, "
        "get_company_info, analyze_stocks, generate_report"
    )

    tests = [
        ("Basic Plan & Execute", test_basic_plan_execute),
        ("Streaming Events", test_streaming_events),
        ("Comprehensive Report", test_comprehensive_report),
        ("Single Stock Deep Dive", test_single_stock_deep_dive),
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
            import traceback

            traceback.print_exc()

    return passed, failed, failures


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Manual test script for Plan & Execute mode with multiple providers"
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
    print("Plan & Execute Agent Manual Test Suite")
    print("Mirrors: examples/notebooks/plan_execute_tutorial.ipynb")
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
