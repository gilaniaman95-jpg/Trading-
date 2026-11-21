"""Trading analysis bot: pulls recent price data for a ticker, has Claude
search the web for current news, and prints a structured analysis.

Usage:
    python trading_bot.py AAPL
    python trading_bot.py AAPL MSFT NVDA --period 1y --effort high

This is research output, not financial advice.
"""

import argparse
import sys

import anthropic
import pandas as pd
import yfinance as yf

MODEL = "claude-opus-5-5"
MAX_PAUSE_RESUMES = 5

SYSTEM_PROMPT = """You are an equity and market analyst. You get recent price \
data and technical indicators for one ticker, and you can search the web for \
current news, earnings and macro context.

Write a concise analysis with these sections:
1. Snapshot: price, trend and where it sits in its 52-week range.
2. Technicals: moving averages, RSI, momentum and volatility, read from the data given.
3. News and catalysts: what is moving the stock now and what is coming up \
(earnings dates, product events, macro). Search the web for this and name your sources.
4. Bull case and bear case: two or three points each.
5. Key levels to watch: support and resistance taken from the data.
6. Overall view: bullish, neutral or bearish over the next few weeks, with a \
confidence level and what would change your mind.

Use only the numbers provided or found in sources; do not invent figures. \
Keep it under about 500 words. End with one line saying this is not financial advice."""


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / period, adjust=False).mean()
    return 100 - 100 / (1 + gain / loss)


def market_summary(ticker: str, period: str) -> str:
    """Download daily prices and return a compact text summary for the prompt."""
    hist = yf.Ticker(ticker).history(period=period, auto_adjust=True)
    if hist.empty:
        raise ValueError(f"No price data found for {ticker!r}")

    close = hist["Close"]
    last = close.iloc[-1]
    year = yf.Ticker(ticker).history(period="1y", auto_adjust=True)["Close"]

    def pct(days: int) -> str:
        if len(close) <= days:
            return "n/a"
        return f"{(last / close.iloc[-days - 1] - 1) * 100:+.2f}%"

    sma20 = close.rolling(20).mean().iloc[-1]
    sma50 = close.rolling(50).mean().iloc[-1]
    sma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else float("nan")
    vol = close.pct_change().tail(20).std() * (252 ** 0.5) * 100
    avg_vol = hist["Volume"].tail(20).mean()

    recent = hist.tail(15)[["Open", "High", "Low", "Close", "Volume"]].round(2)
    recent.index = recent.index.strftime("%Y-%m-%d")

    return f"""Ticker: {ticker}
As of: {hist.index[-1]:%Y-%m-%d}
Last close: {last:.2f}
Change: 1d {pct(1)}, 5d {pct(5)}, 1m {pct(21)}, 3m {pct(63)}
52-week closing high / low: {year.max():.2f} / {year.min():.2f}
SMA20 / SMA50 / SMA200: {sma20:.2f} / {sma50:.2f} / {sma200:.2f}
RSI(14): {rsi(close).iloc[-1]:.1f}
20-day annualized volatility: {vol:.1f}%
20-day average volume: {avg_vol:,.0f}

Last 15 sessions:
{recent.to_string()}"""


def analyze(client: anthropic.Anthropic, ticker: str, period: str, effort: str) -> str:
    messages = [{
        "role": "user",
        "content": f"Analyze {ticker} using this data, plus current news you find on the web.\n\n"
                   + market_summary(ticker, period),
    }]

    for _ in range(MAX_PAUSE_RESUMES + 1):
        response = client.beta.messages.create(
            model=MODEL,
            max_tokens=16000,
            system=SYSTEM_PROMPT,
            output_config={"effort": effort},
            tools=[{"type": "web_search_20260209", "name": "web_search", "max_uses": 5}],
            # If a safety check declines the request, the API retries it on a suitable model.
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=messages,
        )

        if response.stop_reason == "pause_turn":
            # A long web-search turn was paused; send it back so the server continues.
            messages.append({"role": "assistant", "content": response.content})
            continue
        if response.stop_reason == "refusal":
            details = response.stop_details
            reason = f" ({details.category})" if details and details.category else ""
            return f"Claude declined to analyze {ticker}{reason}."

        text = "".join(b.text for b in response.content if b.type == "text")
        if response.stop_reason == "max_tokens":
            text += "\n\n[Output was cut off at the token limit.]"
        return text

    return f"Analysis of {ticker} was still running after {MAX_PAUSE_RESUMES} resumes; try again."


def main() -> int:
    parser = argparse.ArgumentParser(description="Claude-powered stock analysis")
    parser.add_argument("tickers", nargs="+", help="Ticker symbols, e.g. AAPL MSFT")
    parser.add_argument("--period", default="1y", help="History to load (yfinance period, default 1y)")
    parser.add_argument("--effort", default="high", choices=["low", "medium", "high", "xhigh", "max"])
    args = parser.parse_args()

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY or an `ant auth login` profile
    status = 0

    for ticker in (t.upper() for t in args.tickers):
        print(f"\n{'=' * 60}\n{ticker}\n{'=' * 60}")
        try:
            print(analyze(client, ticker, args.period, args.effort))
        except ValueError as e:
            print(e)
            status = 1
        except anthropic.AuthenticationError:
            print("Authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`.")
            return 1
        except anthropic.RateLimitError:
            print("Rate limited by the Claude API; wait a bit and retry.")
            status = 1
        except anthropic.APIStatusError as e:
            print(f"Claude API error {e.status_code}: {e.message}")
            status = 1
        except anthropic.APIConnectionError:
            print("Could not reach the Claude API; check your network.")
            status = 1

    return status


if __name__ == "__main__":
    sys.exit(main())
