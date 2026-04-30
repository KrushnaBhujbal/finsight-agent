import os
import logging
from pydantic import BaseModel
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from tools.stock_data import fetch_stock_data, StockData

log = logging.getLogger("finsight.agents")

class CollectorOutput(BaseModel):
    ticker: str
    company_name: str
    price: float
    pe_ratio: float
    market_cap: int
    revenue: int
    profit_margin: float
    fifty_two_week_high: float
    fifty_two_week_low: float
    analyst_recommendation: str
    valuation_signal: str    # "overvalued" | "fairly_valued" | "undervalued"
    price_vs_52w: str        # "near_high" | "mid_range" | "near_low"
    summary: str

class DataCollectorAgent:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        log.info("DataCollectorAgent initialized")

    def _assess_valuation(self, pe: float) -> str:
        if pe <= 0:
            return "unknown"
        elif pe < 15:
            return "undervalued"
        elif pe < 25:
            return "fairly_valued"
        else:
            return "overvalued"

    def _assess_price_position(self, price: float, high: float, low: float) -> str:
        if high == low:
            return "unknown"
        position = (price - low) / (high - low)
        if position >= 0.75:
            return "near_high"
        elif position >= 0.35:
            return "mid_range"
        else:
            return "near_low"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        before_sleep=lambda rs: log.warning(f"LLM retry {rs.attempt_number}")
    )
    def _generate_summary(self, data: StockData, valuation: str, position: str) -> str:
        prompt = f"""Write a 2-sentence factual summary of this stock's current market position.
Be specific — use the actual numbers provided.

Company: {data.company_name} ({data.ticker})
Price: ${data.current_price}
P/E Ratio: {data.pe_ratio}
Market Cap: ${data.market_cap:,}
Revenue: ${data.revenue:,}
Profit Margin: {data.profit_margin:.1%}
52w High: ${data.fifty_two_week_high} | 52w Low: ${data.fifty_two_week_low}
Analyst Rec: {data.analyst_recommendation.upper()}
Valuation signal: {valuation}
Price position: {position}

Write exactly 2 sentences. Use specific numbers. No speculation."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial data analyst. Be factual, concise, and specific."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def run(self, ticker: str) -> CollectorOutput:
        log.info(f"DataCollectorAgent running for {ticker}")

        data = fetch_stock_data(ticker)
        valuation = self._assess_valuation(data.pe_ratio)
        position = self._assess_price_position(
            data.current_price,
            data.fifty_two_week_high,
            data.fifty_two_week_low
        )
        summary = self._generate_summary(data, valuation, position)

        output = CollectorOutput(
            ticker=ticker,
            company_name=data.company_name,
            price=data.current_price,
            pe_ratio=data.pe_ratio,
            market_cap=data.market_cap,
            revenue=data.revenue,
            profit_margin=data.profit_margin,
            fifty_two_week_high=data.fifty_two_week_high,
            fifty_two_week_low=data.fifty_two_week_low,
            analyst_recommendation=data.analyst_recommendation,
            valuation_signal=valuation,
            price_vs_52w=position,
            summary=summary
        )

        log.info(f"DataCollectorAgent complete: {ticker} | {valuation} | {position}")
        return output

    def format_for_next_agent(self, output: CollectorOutput) -> str:
        return f"""=== STOCK DATA: {output.company_name} ({output.ticker}) ===
Price:              ${output.price}
P/E Ratio:          {output.pe_ratio}
Market Cap:         ${output.market_cap:,}
Revenue:            ${output.revenue:,}
Profit Margin:      {output.profit_margin:.1%}
52w High:           ${output.fifty_two_week_high}
52w Low:            ${output.fifty_two_week_low}
Analyst Rec:        {output.analyst_recommendation.upper()}
Valuation Signal:   {output.valuation_signal.upper()}
Price Position:     {output.price_vs_52w.upper()}
Summary:            {output.summary}"""