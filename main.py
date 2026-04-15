import os
from dotenv import load_dotenv
from groq import Groq
import yfinance as yf
import chromadb

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    raise ValueError("GROQ_API_KEY not found — check your .env file")

print("API key loaded.")

client = Groq(api_key=api_key)

response = client.chat.completions.create(
   model="llama-3.1-8b-instant",
    messages=[
        {"role": "system", "content": "You are FinSight, an autonomous investment research agent."},
        {"role": "user", "content": "Say exactly this: FinSight agent is online and ready."}
    ]
)
print(response.choices[0].message.content)

ticker = yf.Ticker("AAPL")
info = ticker.info
print(f"\nAAPL price: ${info.get('currentPrice', 'N/A')}")
print(f"P/E ratio:  {info.get('trailingPE', 'N/A')}")
print(f"Market cap: ${info.get('marketCap', 'N/A'):,}")

chroma_client = chromadb.Client()
chroma_client.create_collection("test")
print("\nChromaDB connected.")

print("\nDay 1 complete. All systems go.")