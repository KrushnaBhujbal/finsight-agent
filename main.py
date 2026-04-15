import os
from dotenv import load_dotenv
from groq import Groq
from pydantic import BaseModel, ValidationError
import json

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# ── 1. Define the output schema ──────────────────────────────
class StockSummary(BaseModel):
    ticker: str
    company_name: str
    current_price: float
    summary: str
    sentiment: str        # "bullish", "bearish", or "neutral"
    confidence: str       # "high", "medium", or "low"
    key_risks: list[str]  # exactly 2 risks

# ── 2. The prompt ─────────────────────────────────────────────
def analyze_stock(ticker: str, price: float, pe_ratio: float, market_cap: int) -> StockSummary:

    system_prompt = """You are FinSight, an autonomous investment research agent.

Your job is to analyze stock data and return a structured JSON response.

STRICT RULES:
- Always respond with valid JSON only — no extra text, no markdown, no backticks
- sentiment must be exactly one of: "bullish", "bearish", "neutral"
- confidence must be exactly one of: "high", "medium", "low"
- key_risks must be a list of exactly 2 short strings
- summary must be 2-3 sentences maximum

JSON format:
{
  "ticker": "string",
  "company_name": "string",
  "current_price": float,
  "summary": "string",
  "sentiment": "bullish" | "bearish" | "neutral",
  "confidence": "high" | "medium" | "low",
  "key_risks": ["risk1", "risk2"]
}"""

    user_prompt = f"""Analyze this stock:
Ticker: {ticker}
Current Price: ${price}
P/E Ratio: {pe_ratio}
Market Cap: ${market_cap:,}

Return JSON only."""

    response = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        temperature=0.1
    )

    raw = response.choices[0].message.content.strip()

    # ── 3. Parse and validate with Pydantic ───────────────────
    try:
        data = json.loads(raw)
        result = StockSummary(**data)
        return result
    except json.JSONDecodeError:
        print(f"Raw output was not valid JSON:\n{raw}")
        raise
    except ValidationError as e:
        print(f"Schema validation failed:\n{e}")
        raise

# ── 4. Run it on 3 tickers ────────────────────────────────────
import yfinance as yf

tickers = ["AAPL", "MSFT", "NVDA"]

for symbol in tickers:
    print(f"\n{'='*50}")
    print(f"Analyzing {symbol}...")

    info = yf.Ticker(symbol).info
    price = info.get("currentPrice", 0)
    pe = info.get("trailingPE", 0)
    mcap = info.get("marketCap", 0)

    result = analyze_stock(symbol, price, pe, mcap)

    print(f"Company:    {result.company_name}")
    print(f"Price:      ${result.current_price}")
    print(f"Sentiment:  {result.sentiment.upper()}")
    print(f"Confidence: {result.confidence}")
    print(f"Summary:    {result.summary}")
    print(f"Risks:      {result.key_risks[0]}")
    print(f"            {result.key_risks[1]}")

print(f"\n{'='*50}")
print("Day 2 complete. Structured output working.")