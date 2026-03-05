# AI Trading Signal Agent

An autonomous trading signal agent built with LangGraph and Claude. The agent fetches real-time market data, computes technical indicators, searches financial news, and generates structured BUY/SELL/HOLD signals with confidence scores.

## Architecture

```
User query ("Analyze AAPL")
        │
        ▼
┌─────────────────────────────────────────┐
│          LangGraph ReAct Agent          │
│                                         │
│   Reason → Act → Observe → Repeat       │
│                                         │
│   ┌───────────┐  ┌──────────────────┐   │
│   │  Claude    │  │  Tool Executor   │   │
│   │  (LLM)    │◄─┤                  │   │
│   │           │──►│  Autonomous      │   │
│   └───────────┘  │  tool calling    │   │
│                   └──────────────────┘   │
└─────────────────────────────────────────┘
        │                    │
        ▼                    ▼
  Structured Signal    Tool Results
  (BUY/SELL/HOLD)    ┌─────────────────┐
                     │ fetch_price_data │ yfinance
                     │ compute_indicators│ RSI, MACD, z-score, Bollinger
                     │ search_news      │ Yahoo Finance
                     │ backtest_signal  │ Historical validation
                     └─────────────────┘
```

The agent uses the **ReAct pattern** (Reason-Act): the LLM autonomously decides which tools to call, interprets results, and iterates until it has enough information to produce a trading signal.

## Tools

| Tool | Description |
|------|-------------|
| `fetch_price_data` | Fetches historical OHLCV data via yfinance |
| `compute_indicators` | Calculates RSI (14-day), MACD, z-score, Bollinger Bands, SMA (20/50), volatility |
| `search_news` | Retrieves recent financial news headlines from Yahoo Finance |
| `backtest_signal` | Validates a signal against recent price history |

## Output Format

```
## Signal: BUY
## Confidence: 0.73
## Ticker: AAPL

### Technical Analysis
- RSI: 42.3 — neutral, room to run
- MACD: 1.23 — bullish crossover
- Z-score: -0.8 — slightly below mean
- Trend: upward momentum on 20-day SMA

### News Sentiment
- positive — strong earnings beat, raised guidance

### Reasoning
RSI is neutral with MACD showing bullish momentum...

### Risk Factors
- Broader market uncertainty
- Elevated valuation multiples
```

## Setup

```bash
# Clone
git clone https://github.com/Sophie-l-l/ai-trading-agent.git
cd ai-trading-agent

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -e .

# Add your Anthropic API key
cp .env.example .env
# Edit .env with your key
```

## Usage

```bash
# Analyze a single ticker
python main.py AAPL

# Analyze crypto
python main.py BTC-USD

# Run multi-ticker evaluation
python main.py --eval

# Evaluate specific tickers
python main.py --eval AAPL TSLA NVDA
```

## Evaluation Framework

The eval module (`eval.py`) runs the agent across multiple tickers and produces aggregate metrics:

- **Signal distribution** — counts of BUY/SELL/HOLD across tickers
- **Confidence scores** — average and per-ticker confidence
- **Latency tracking** — time per analysis
- **Signal parsing** — extracts structured data from free-text output

Results are saved to `eval_report.json`.

## Tech Stack

- **Agent**: LangGraph (ReAct pattern), LangChain
- **LLM**: Claude (Anthropic) via tool calling / function calling
- **Data**: yfinance (real-time market data)
- **Analysis**: Pandas, NumPy (technical indicators)
- **Python 3.11+**

## Project Structure

```
ai-trading-agent/
├── main.py          # CLI entry point
├── agent.py         # LangGraph ReAct agent + system prompt
├── tools.py         # 4 LangChain tools (price, indicators, news, backtest)
├── eval.py          # Multi-ticker evaluation framework
├── pyproject.toml   # Dependencies
└── .env.example     # API key template
```
