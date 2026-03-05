"""
Tools that the AI agent can call autonomously.

This is "tool calling" / "function calling" — you define Python functions,
describe them to the LLM, and the LLM decides when to invoke them.
The agent sees the function name, description, and parameter schema,
then outputs structured JSON to call whichever tool it needs.
"""

import yfinance as yf
import pandas as pd
import numpy as np
from langchain_core.tools import tool


@tool
def fetch_price_data(ticker: str, period: str = "3mo") -> str:
    """Fetch historical price data for a stock or crypto ticker.

    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'BTC-USD', 'TSLA')
        period: Time period - '1mo', '3mo', '6mo', '1y'

    Returns:
        CSV-formatted price data with Date, Open, High, Low, Close, Volume
    """
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        return f"Error: No data found for ticker '{ticker}'"

    # Flatten multi-level columns if present
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    # Keep last 60 rows to fit in context
    data = data.tail(60)
    data.index = data.index.strftime("%Y-%m-%d")
    return data.to_csv()


@tool
def compute_indicators(ticker: str, period: str = "6mo") -> str:
    """Compute technical indicators for a given ticker.

    Calculates: SMA (20-day, 50-day), RSI (14-day), z-score,
    MACD, Bollinger Bands, and daily returns.

    Args:
        ticker: Stock ticker symbol
        period: Time period for data

    Returns:
        Latest indicator values and interpretation
    """
    data = yf.download(ticker, period=period, progress=False)
    if data.empty:
        return f"Error: No data found for ticker '{ticker}'"

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = data["Close"]

    # Simple Moving Averages
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()

    # RSI (Relative Strength Index)
    delta = close.diff()
    gain = delta.where(delta > 0, 0.0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))

    # Z-score (how many std devs from 20-day mean)
    z_score = (close - sma_20) / close.rolling(20).std()

    # MACD
    ema_12 = close.ewm(span=12).mean()
    ema_26 = close.ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal_line = macd.ewm(span=9).mean()

    # Bollinger Bands
    bb_std = close.rolling(20).std()
    bb_upper = sma_20 + 2 * bb_std
    bb_lower = sma_20 - 2 * bb_std

    # Latest values
    latest = close.iloc[-1]
    result = {
        "ticker": ticker,
        "latest_price": round(float(latest), 2),
        "sma_20": round(float(sma_20.iloc[-1]), 2),
        "sma_50": round(float(sma_50.iloc[-1]), 2),
        "rsi_14": round(float(rsi.iloc[-1]), 2),
        "z_score": round(float(z_score.iloc[-1]), 3),
        "macd": round(float(macd.iloc[-1]), 3),
        "macd_signal": round(float(signal_line.iloc[-1]), 3),
        "bollinger_upper": round(float(bb_upper.iloc[-1]), 2),
        "bollinger_lower": round(float(bb_lower.iloc[-1]), 2),
        "daily_return_pct": round(float(close.pct_change().iloc[-1] * 100), 2),
        "volatility_20d": round(float(close.pct_change().rolling(20).std().iloc[-1] * 100), 2),
    }

    # Interpretations
    interpretations = []
    if result["rsi_14"] < 30:
        interpretations.append("RSI < 30: OVERSOLD signal")
    elif result["rsi_14"] > 70:
        interpretations.append("RSI > 70: OVERBOUGHT signal")

    if result["z_score"] < -2:
        interpretations.append(f"Z-score {result['z_score']}: price significantly BELOW mean")
    elif result["z_score"] > 2:
        interpretations.append(f"Z-score {result['z_score']}: price significantly ABOVE mean")

    if result["macd"] > result["macd_signal"]:
        interpretations.append("MACD above signal line: BULLISH crossover")
    else:
        interpretations.append("MACD below signal line: BEARISH crossover")

    if latest < result["bollinger_lower"]:
        interpretations.append("Price below lower Bollinger Band: potential reversal UP")
    elif latest > result["bollinger_upper"]:
        interpretations.append("Price above upper Bollinger Band: potential reversal DOWN")

    result["interpretations"] = interpretations
    return str(result)


@tool
def search_news(query: str) -> str:
    """Search for recent financial news about a ticker or topic.

    Uses Yahoo Finance news feed for the given ticker.

    Args:
        query: Ticker symbol or search query (e.g., 'AAPL', 'Bitcoin regulation')

    Returns:
        Recent news headlines and summaries
    """
    try:
        stock = yf.Ticker(query)
        news = stock.news
        if not news:
            return f"No recent news found for '{query}'"

        results = []
        for item in news[:5]:  # Top 5 articles
            content = item.get("content", {})
            title = content.get("title", "No title")
            summary = content.get("summary", "No summary available")
            pub_date = content.get("pubDate", "Unknown date")
            results.append(f"- [{pub_date}] {title}\n  {summary}")

        return f"Recent news for '{query}':\n\n" + "\n\n".join(results)
    except Exception as e:
        return f"Error fetching news for '{query}': {e}"


@tool
def backtest_signal(ticker: str, signal: str, horizon_days: int = 20) -> str:
    """Backtest a trading signal against recent historical data.

    Checks what would have happened if you followed this signal N days ago.

    Args:
        ticker: Stock ticker symbol
        signal: The signal to test - 'BUY', 'SELL', or 'HOLD'
        horizon_days: Number of days to look back for the test

    Returns:
        Backtest result showing if the signal would have been profitable
    """
    data = yf.download(ticker, period="6mo", progress=False)
    if data.empty:
        return f"Error: No data found for ticker '{ticker}'"

    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)

    close = data["Close"]

    if len(close) < horizon_days + 1:
        return "Error: Not enough historical data for backtest"

    # Price N days ago vs today
    entry_price = float(close.iloc[-(horizon_days + 1)])
    current_price = float(close.iloc[-1])
    pct_change = ((current_price - entry_price) / entry_price) * 100

    # Was the signal correct?
    if signal.upper() == "BUY":
        correct = pct_change > 0
        outcome = f"Price went {'UP' if pct_change > 0 else 'DOWN'} {abs(pct_change):.2f}%"
    elif signal.upper() == "SELL":
        correct = pct_change < 0
        outcome = f"Price went {'DOWN' if pct_change < 0 else 'UP'} {abs(pct_change):.2f}%"
    else:  # HOLD
        correct = abs(pct_change) < 5  # Hold is "correct" if price didn't move much
        outcome = f"Price changed {pct_change:+.2f}%"

    # Additional context
    max_price = float(close.iloc[-horizon_days:].max())
    min_price = float(close.iloc[-horizon_days:].min())
    max_drawdown = ((min_price - entry_price) / entry_price) * 100

    return (
        f"Backtest: {signal.upper()} {ticker} ({horizon_days} days ago)\n"
        f"Entry: ${entry_price:.2f} → Current: ${current_price:.2f}\n"
        f"Result: {outcome}\n"
        f"Signal was: {'CORRECT' if correct else 'INCORRECT'}\n"
        f"Period high: ${max_price:.2f}, low: ${min_price:.2f}\n"
        f"Max drawdown: {max_drawdown:.2f}%"
    )


# List of all tools for the agent
all_tools = [fetch_price_data, compute_indicators, search_news, backtest_signal]
