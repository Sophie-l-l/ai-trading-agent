"""
Evaluation framework for the trading agent.

This is "LLM evaluation" (evals) — systematically testing the agent's
output quality against real-world outcomes.

We run the agent on multiple tickers, collect its signals,
then check against actual price movements to measure accuracy.
"""

import json
import time
from datetime import datetime
from agent import run_analysis


def parse_signal(response: str) -> dict:
    """Extract structured signal from agent's response text.

    Args:
        response: The agent's full text response

    Returns:
        Dict with signal, confidence, ticker parsed from the response
    """
    result = {"signal": "UNKNOWN", "confidence": 0.0, "ticker": ""}

    for line in response.split("\n"):
        line_lower = line.lower().strip()
        if "signal:" in line_lower:
            if "buy" in line_lower:
                result["signal"] = "BUY"
            elif "sell" in line_lower:
                result["signal"] = "SELL"
            elif "hold" in line_lower:
                result["signal"] = "HOLD"
        if "confidence:" in line_lower:
            # Extract number from line like "## Confidence: 0.73"
            parts = line.split(":")
            if len(parts) > 1:
                try:
                    result["confidence"] = float(
                        parts[-1].strip().strip("[]").strip()
                    )
                except ValueError:
                    pass
        if "ticker:" in line_lower:
            parts = line.split(":")
            if len(parts) > 1:
                result["ticker"] = parts[-1].strip().strip("[]").strip()

    return result


def evaluate_agent(
    tickers: list[str],
    model_name: str = "claude-sonnet-4-20250514",
    delay: float = 2.0,
) -> dict:
    """Run the agent on multiple tickers and collect evaluation metrics.

    Args:
        tickers: List of ticker symbols to analyze
        model_name: LLM model to use
        delay: Seconds between API calls (rate limiting)

    Returns:
        Evaluation report with per-ticker results and aggregate metrics
    """
    results = []

    for ticker in tickers:
        print(f"\n{'='*50}")
        print(f"Analyzing {ticker}...")
        print(f"{'='*50}")

        start_time = time.time()
        try:
            response = run_analysis(ticker, model_name)
            latency = time.time() - start_time

            signal = parse_signal(response)

            result = {
                "ticker": ticker,
                "signal": signal["signal"],
                "confidence": signal["confidence"],
                "latency_seconds": round(latency, 2),
                "response_length": len(response),
                "full_response": response,
                "error": None,
            }

            print(f"Signal: {signal['signal']} (confidence: {signal['confidence']})")
            print(f"Latency: {latency:.2f}s")

        except Exception as e:
            latency = time.time() - start_time
            result = {
                "ticker": ticker,
                "signal": "ERROR",
                "confidence": 0.0,
                "latency_seconds": round(latency, 2),
                "response_length": 0,
                "full_response": "",
                "error": str(e),
            }
            print(f"Error: {e}")

        results.append(result)

        if delay > 0 and ticker != tickers[-1]:
            time.sleep(delay)

    # Aggregate metrics
    valid_results = [r for r in results if r["signal"] != "ERROR"]
    signals = [r["signal"] for r in valid_results]
    confidences = [r["confidence"] for r in valid_results]
    latencies = [r["latency_seconds"] for r in valid_results]

    report = {
        "timestamp": datetime.now().isoformat(),
        "model": model_name,
        "tickers_analyzed": len(tickers),
        "successful": len(valid_results),
        "errors": len(results) - len(valid_results),
        "signal_distribution": {
            "BUY": signals.count("BUY"),
            "SELL": signals.count("SELL"),
            "HOLD": signals.count("HOLD"),
            "UNKNOWN": signals.count("UNKNOWN"),
        },
        "avg_confidence": round(sum(confidences) / len(confidences), 3) if confidences else 0,
        "avg_latency_seconds": round(sum(latencies) / len(latencies), 2) if latencies else 0,
        "max_latency_seconds": round(max(latencies), 2) if latencies else 0,
        "results": results,
    }

    return report


def print_report(report: dict):
    """Pretty-print an evaluation report."""
    print(f"\n{'='*60}")
    print(f"  EVALUATION REPORT")
    print(f"  {report['timestamp']}")
    print(f"  Model: {report['model']}")
    print(f"{'='*60}")
    print(f"  Tickers analyzed: {report['tickers_analyzed']}")
    print(f"  Successful:       {report['successful']}")
    print(f"  Errors:           {report['errors']}")
    print(f"  Avg confidence:   {report['avg_confidence']}")
    print(f"  Avg latency:      {report['avg_latency_seconds']}s")
    print(f"  Max latency:      {report['max_latency_seconds']}s")
    print(f"\n  Signal Distribution:")
    for signal, count in report["signal_distribution"].items():
        bar = "█" * count
        print(f"    {signal:8s} {count:2d}  {bar}")
    print(f"\n  Per-Ticker Results:")
    for r in report["results"]:
        status = "✓" if r["signal"] != "ERROR" else "✗"
        print(
            f"    {status} {r['ticker']:10s} → {r['signal']:6s} "
            f"(conf: {r['confidence']:.2f}, {r['latency_seconds']:.1f}s)"
        )
    print(f"{'='*60}")


if __name__ == "__main__":
    # Run evaluation on a diverse set of tickers
    test_tickers = ["AAPL", "BTC-USD", "TSLA", "NVDA", "GLD"]

    print("Starting agent evaluation...")
    report = evaluate_agent(test_tickers)
    print_report(report)

    # Save report to JSON
    with open("eval_report.json", "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nReport saved to eval_report.json")
