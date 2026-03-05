"""
AI Trading Signal Agent — Entry Point

Usage:
    python main.py AAPL          # Analyze a single ticker
    python main.py --eval        # Run evaluation on multiple tickers
"""

import sys
from dotenv import load_dotenv

load_dotenv()


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python main.py AAPL          # Analyze a single ticker")
        print("  python main.py --eval        # Run evaluation suite")
        sys.exit(1)

    if sys.argv[1] == "--eval":
        from eval import evaluate_agent, print_report
        import json

        tickers = sys.argv[2:] if len(sys.argv) > 2 else ["AAPL", "BTC-USD", "TSLA", "NVDA", "GLD"]
        print(f"Running evaluation on: {', '.join(tickers)}\n")

        report = evaluate_agent(tickers)
        print_report(report)

        with open("eval_report.json", "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to eval_report.json")
    else:
        from agent import run_analysis

        ticker = sys.argv[1].upper()
        print(f"Analyzing {ticker}...\n")
        print("The agent will:")
        print("  1. Fetch price data")
        print("  2. Compute technical indicators")
        print("  3. Search recent news")
        print("  4. Generate a trading signal\n")

        result = run_analysis(ticker)
        print(result)


if __name__ == "__main__":
    main()
