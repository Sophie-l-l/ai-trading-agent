"""
LangGraph ReAct Agent for trading signal generation.

HOW IT WORKS:
1. LangGraph creates a state graph with two nodes: "agent" and "tools"
2. The "agent" node calls the LLM, which can decide to:
   a) Call a tool (fetch data, compute indicators, etc.)
   b) Return a final answer (the trading signal)
3. If the LLM calls a tool, the "tools" node executes it and feeds
   the result back to the agent
4. This loop repeats until the agent has enough info to give a signal

This is the "ReAct" pattern: Reason → Act → Observe → Repeat
"""

from langgraph.prebuilt import create_react_agent
from langchain_anthropic import ChatAnthropic
from tools import all_tools

SYSTEM_PROMPT = """You are an expert financial analyst and quantitative trading assistant.

When asked to analyze a ticker, you MUST:
1. First fetch the price data to see recent trends
2. Compute technical indicators (RSI, MACD, z-score, Bollinger Bands)
3. Search for recent news that might affect the price
4. Optionally backtest to validate your reasoning

Then provide your analysis in this EXACT format:

## Signal: [BUY / SELL / HOLD]
## Confidence: [0.0 - 1.0]
## Ticker: [SYMBOL]

### Technical Analysis
- RSI: [value] — [interpretation]
- MACD: [value] — [interpretation]
- Z-score: [value] — [interpretation]
- Trend: [description]

### News Sentiment
- [positive/negative/neutral] — [brief summary]

### Reasoning
[2-3 sentences explaining why you chose this signal]

### Risk Factors
- [key risk 1]
- [key risk 2]

Be data-driven. Never recommend without checking indicators first.
If the data is unclear or conflicting, prefer HOLD.
"""


def create_agent(model_name: str = "claude-sonnet-4-20250514"):
    """Create and return the trading agent.

    Args:
        model_name: Anthropic model to use. Default is Claude Sonnet 4
                    (good balance of speed and quality for tool calling).

    Returns:
        A LangGraph agent that can analyze tickers and generate signals.
    """
    llm = ChatAnthropic(model=model_name)

    # create_react_agent builds the full ReAct loop:
    # LLM decides → tool executes → result fed back → LLM decides again
    agent = create_react_agent(
        model=llm,
        tools=all_tools,
        prompt=SYSTEM_PROMPT,
    )

    return agent


def run_analysis(ticker: str, model_name: str = "claude-sonnet-4-20250514") -> str:
    """Run a full trading analysis for a ticker.

    Args:
        ticker: Stock/crypto symbol (e.g., 'AAPL', 'BTC-USD')
        model_name: LLM model to use

    Returns:
        The agent's full analysis with signal, confidence, and reasoning
    """
    agent = create_agent(model_name)

    result = agent.invoke({
        "messages": [
            {"role": "user", "content": f"Analyze {ticker} and give me a trading signal."}
        ]
    })

    # The last message contains the agent's final response
    last_message = result["messages"][-1]
    return last_message.content
