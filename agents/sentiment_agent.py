import os
import logging
from pydantic import BaseModel
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential

from tools.news_search import fetch_news, NewsArticle

log = logging.getLogger("finsight.agents")

class ArticleSentiment(BaseModel):
    title: str
    source: str
    sentiment: str        # "bullish" | "bearish" | "neutral"
    relevance: str        # "high" | "medium" | "low"
    key_point: str

class SentimentOutput(BaseModel):
    ticker: str
    company_name: str
    overall_sentiment: str      # "bullish" | "bearish" | "neutral" | "mixed"
    sentiment_score: float      # -1.0 to 1.0
    bullish_count: int
    bearish_count: int
    neutral_count: int
    articles: list[ArticleSentiment]
    sentiment_summary: str
    news_driven_risk: str       # "high" | "medium" | "low"

class SentimentAgent:

    def __init__(self):
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.model = "llama-3.1-8b-instant"
        log.info("SentimentAgent initialized")

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        before_sleep=lambda rs: log.warning(f"LLM retry {rs.attempt_number}")
    )
    def _score_articles(self, ticker: str, company: str, articles: list[NewsArticle]) -> list[ArticleSentiment]:
        if not articles:
            return []

        articles_text = "\n".join([
            f"{i+1}. Title: {a.title}\n   Source: {a.source}\n   Summary: {a.summary}"
            for i, a in enumerate(articles)
        ])

        prompt = f"""Analyze the sentiment of these news articles about {company} ({ticker}).

For each article return a JSON array with objects containing:
- title: exact article title
- source: source name
- sentiment: exactly "bullish", "bearish", or "neutral"
- relevance: exactly "high", "medium", or "low" (is this directly about {company}?)
- key_point: one sentence summarizing the investment-relevant point

Articles:
{articles_text}

Return ONLY a valid JSON array. No markdown, no backticks, no extra text.
Example: [{{"title":"...", "source":"...", "sentiment":"bullish", "relevance":"high", "key_point":"..."}}]"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial news analyst. Return only valid JSON arrays."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )

        import json
        raw = response.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        data = json.loads(raw)
        return [ArticleSentiment(**item) for item in data]

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=8),
        before_sleep=lambda rs: log.warning(f"LLM retry {rs.attempt_number}")
    )
    def _generate_sentiment_summary(
        self,
        ticker: str,
        company: str,
        scored: list[ArticleSentiment],
        overall: str,
        score: float
    ) -> str:

        articles_summary = "\n".join([
            f"- [{a.sentiment.upper()}] {a.title}: {a.key_point}"
            for a in scored
        ])

        prompt = f"""Write a 2-sentence news sentiment summary for {company} ({ticker}).

Overall sentiment: {overall} (score: {score:.2f})
Scored articles:
{articles_summary}

Be specific about what the news says. Reference actual headlines.
Write exactly 2 sentences."""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a financial analyst summarizing news sentiment."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        return response.choices[0].message.content.strip()

    def _calculate_score(self, scored: list[ArticleSentiment]) -> tuple[float, str]:
        if not scored:
            return 0.0, "neutral"

        weights = {"high": 1.5, "medium": 1.0, "low": 0.5}
        values = {"bullish": 1, "neutral": 0, "bearish": -1}

        total_weight = 0.0
        weighted_sum = 0.0

        for a in scored:
            w = weights.get(a.relevance, 1.0)
            v = values.get(a.sentiment, 0)
            weighted_sum += v * w
            total_weight += w

        score = weighted_sum / total_weight if total_weight > 0 else 0.0
        score = max(-1.0, min(1.0, score))

        if score >= 0.4:
            label = "bullish"
        elif score <= -0.4:
            label = "bearish"
        elif score > 0.1:
            label = "mixed"
        else:
            label = "neutral"

        return round(score, 3), label

    def _assess_news_risk(self, scored: list[ArticleSentiment], score: float) -> str:
        bearish_high = sum(1 for a in scored if a.sentiment == "bearish" and a.relevance == "high")
        if bearish_high >= 2 or score <= -0.5:
            return "high"
        elif bearish_high == 1 or score <= -0.2:
            return "medium"
        else:
            return "low"

    def run(self, ticker: str, company_name: str = "") -> SentimentOutput:
        log.info(f"SentimentAgent running for {ticker}")

        news = fetch_news(ticker, company_name or ticker)

        if not news.articles:
            log.warning(f"No articles found for {ticker}")
            return SentimentOutput(
                ticker=ticker,
                company_name=company_name or ticker,
                overall_sentiment="neutral",
                sentiment_score=0.0,
                bullish_count=0,
                bearish_count=0,
                neutral_count=0,
                articles=[],
                sentiment_summary="No recent news found for this ticker.",
                news_driven_risk="low"
            )

        try:
            scored = self._score_articles(ticker, company_name, news.articles)
        except Exception as e:
            log.error(f"Article scoring failed: {e}")
            scored = []

        score, label = self._calculate_score(scored)
        risk = self._assess_news_risk(scored, score)

        bullish = sum(1 for a in scored if a.sentiment == "bullish")
        bearish = sum(1 for a in scored if a.sentiment == "bearish")
        neutral = sum(1 for a in scored if a.sentiment == "neutral")

        try:
            summary = self._generate_sentiment_summary(ticker, company_name, scored, label, score)
        except Exception as e:
            log.error(f"Summary generation failed: {e}")
            summary = f"News sentiment for {ticker}: {label} (score: {score:.2f})"

        output = SentimentOutput(
            ticker=ticker,
            company_name=company_name or ticker,
            overall_sentiment=label,
            sentiment_score=score,
            bullish_count=bullish,
            bearish_count=bearish,
            neutral_count=neutral,
            articles=scored,
            sentiment_summary=summary,
            news_driven_risk=risk
        )

        log.info(f"SentimentAgent complete: {ticker} | {label} | score: {score} | risk: {risk}")
        return output

    def format_for_next_agent(self, output: SentimentOutput) -> str:
        articles_text = "\n".join([
            f"  [{a.sentiment.upper()}][{a.relevance}] {a.title} — {a.key_point}"
            for a in output.articles
        ])
        return f"""=== NEWS SENTIMENT: {output.company_name} ({output.ticker}) ===
Overall Sentiment:   {output.overall_sentiment.upper()}
Sentiment Score:     {output.sentiment_score} (range: -1.0 bearish to +1.0 bullish)
Breakdown:          {output.bullish_count} bullish | {output.neutral_count} neutral | {output.bearish_count} bearish
News Risk:          {output.news_driven_risk.upper()}
Summary:            {output.sentiment_summary}

Scored Articles:
{articles_text}"""