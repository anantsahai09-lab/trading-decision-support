"""
chart_reader.py — read a TradingView screenshot, then score it against the rules.

WHAT THIS DOES
    1. You upload a chart screenshot.
    2. A vision model reads the levels off it and returns JSON.
    3. Those levels go into manual.check(), which runs every engine rule.

    The vision model does one job: transcribe what is visibly on the chart. It
    is not asked to judge the setup. Judgement stays with the rules engine and
    with you.

WHAT IT READS RELIABLY
    Printed text: the ticker, the price axis, timestamps, and the numbers on
    any position tool or horizontal lines you have drawn.

WHAT IT DOES NOT
    Anything requiring inference from pixels — where the opening range "really"
    is, whether accumulation looks clean, which index is leading. Those are
    reads, not transcription, and a vision model will guess confidently and
    sometimes wrongly. Levels it reports are treated as a starting point to be
    confirmed, not as ground truth.

SETUP
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...      (or pass api_key= directly)

QUICK TEST
    python src/chart_reader.py chart.png --key sk-ant-...

USE
    from .chart_reader import read_chart, score_chart

    levels = read_chart("chart.png")          # -> dict of what it saw
    score_chart("chart.png", data_1m=data)    # read + run every rule

PROVIDERS
    Anthropic by default. provider="openai" or "google" also supported; the
    prompt and parsing are shared, only the transport differs.
"""

from __future__ import annotations

import base64
import json
import os
import re

EXTRACTION_PROMPT = """You are transcribing a futures trading chart. Report only
what is visibly printed or drawn. Do not infer, estimate, or judge the setup.

Return a single JSON object, no prose, no code fences:

{
  "symbol": "NQ" | "ES" | "MNQ" | "MES" | null,
  "timeframe": "1m" | "5m" | "15m" | null,
  "date": "YYYY-MM-DD" or null,
  "current_price": number or null,
  "entry": number or null,
  "stop": number or null,
  "target": number or null,
  "direction": "bull" | "bear" | null,
  "drawn_levels": [ {"price": number, "label": "string or null"} ],
  "session_time": "HH:MM" or null,
  "notes": "anything relevant you can read but could not place in a field",
  "confidence": {"entry": "high|medium|low", "stop": "...", "target": "..."}
}

Rules:
- Read prices from the axis and from any position tool or horizontal line.
- If a long/short position tool is present, its entry, stop and target are
  authoritative; set direction from which way it points.
- If a value is not visible, use null. Never guess a number.
- Mark confidence "low" for anything you had to read from a blurry or partially
  obscured area.
"""


# ---------------------------------------------------------------------
# transport
# ---------------------------------------------------------------------

def _encode(path: str) -> tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()
    media = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
             ".gif": "image/gif", ".webp": "image/webp"}.get(ext, "image/png")
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode(), media


def _call_anthropic(b64, media, api_key, model):
    import anthropic
    client = anthropic.Anthropic(api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model=model or "claude-sonnet-4-6",
        max_tokens=1000,
        messages=[{"role": "user", "content": [
            {"type": "image",
             "source": {"type": "base64", "media_type": media, "data": b64}},
            {"type": "text", "text": EXTRACTION_PROMPT},
        ]}],
    )
    return msg.content[0].text


def _call_openai(b64, media, api_key, model):
    from openai import OpenAI
    client = OpenAI(api_key=api_key or os.environ.get("OPENAI_API_KEY"))
    r = client.chat.completions.create(
        model=model or "gpt-4o",
        max_tokens=1000,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": EXTRACTION_PROMPT},
            {"type": "image_url",
             "image_url": {"url": f"data:{media};base64,{b64}"}},
        ]}],
    )
    return r.choices[0].message.content


def _call_google(b64, media, api_key, model):
    import google.generativeai as genai
    genai.configure(api_key=api_key or os.environ.get("GOOGLE_API_KEY"))
    m = genai.GenerativeModel(model or "gemini-1.5-pro")
    r = m.generate_content([
        {"mime_type": media, "data": base64.b64decode(b64)},
        EXTRACTION_PROMPT,
    ])
    return r.text


_PROVIDERS = {"anthropic": _call_anthropic, "openai": _call_openai,
              "google": _call_google}


def _parse(raw: str) -> dict:
    """Pull the JSON object out of a model response."""
    txt = raw.strip()
    txt = re.sub(r"^```(?:json)?|```$", "", txt, flags=re.M).strip()
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", txt, re.S)
        if not m:
            raise ValueError(f"no JSON in response:\n{raw[:400]}")
        return json.loads(m.group(0))


# ---------------------------------------------------------------------
# public
# ---------------------------------------------------------------------

def read_chart(path: str, provider: str = "anthropic", api_key: str | None = None,
               model: str | None = None, verbose: bool = True) -> dict:
    """Transcribe a chart screenshot into structured levels."""
    if provider not in _PROVIDERS:
        raise ValueError(f"provider must be one of {list(_PROVIDERS)}")
    b64, media = _encode(path)
    out = _parse(_PROVIDERS[provider](b64, media, api_key, model))

    if verbose:
        print("=" * 66)
        print(f"  READ FROM {os.path.basename(path)}")
        print("=" * 66)
        for k in ("symbol", "timeframe", "date", "session_time", "direction",
                  "current_price", "entry", "stop", "target"):
            v = out.get(k)
            if v is not None:
                conf = (out.get("confidence") or {}).get(k)
                print(f"  {k:<15} {v}" + (f"   [{conf} confidence]" if conf else ""))
        lv = out.get("drawn_levels") or []
        if lv:
            print(f"  drawn levels    " +
                  ", ".join(f"{l.get('price')}" +
                            (f" ({l['label']})" if l.get("label") else "")
                            for l in lv))
        if out.get("notes"):
            print(f"  notes           {out['notes']}")
        low = [k for k, v in (out.get("confidence") or {}).items() if v == "low"]
        if low:
            print(f"\n  Low confidence on: {', '.join(low)} — verify before acting.")
    return out


def score_chart(path: str, data_1m: dict | None = None, account: float = 150_000.0,
                risk_pct: float = 0.015, rr: float = 2.0,
                provider: str = "anthropic", api_key: str | None = None,
                model: str | None = None, **overrides) -> dict:
    """
    Read a screenshot and run every engine rule against what it found.

    Anything the model could not read can be supplied directly:

        score_chart("chart.png", entry=21050, direction="bull", data_1m=data)
    """
    from .manual import check

    seen = read_chart(path, provider, api_key, model)
    seen.update({k: v for k, v in overrides.items() if v is not None})

    missing = [k for k in ("symbol", "direction", "entry") if not seen.get(k)]
    if missing:
        raise ValueError(
            f"could not read {missing} from the chart — pass them directly, e.g. "
            f"score_chart({path!r}, " +
            ", ".join(f"{k}=..." for k in missing) + ")")

    sym = seen["symbol"].upper()
    if sym in ("MNQ", "MES"):          # micros share the parent's tick geometry
        sym = sym[1:]

    print()
    return check(
        symbol=sym,
        direction=seen["direction"],
        entry=float(seen["entry"]),
        stop=float(seen["stop"]) if seen.get("stop") else None,
        target=float(seen["target"]) if seen.get("target") else None,
        stop_ticks=None if seen.get("stop") else 200.0,
        date=seen.get("date"),
        time_of_day=seen.get("session_time"),
        account=account, risk_pct=risk_pct, rr=rr,
        data_1m=data_1m,
    )


def upload_and_score(data_1m: dict | None = None, **kwargs):
    """Colab convenience: file picker, then read and score."""
    try:
        from google.colab import files
    except ImportError:
        raise RuntimeError("upload_and_score() is Colab-only; "
                           "use score_chart(path) elsewhere")
    up = files.upload()
    if not up:
        print("no file chosen")
        return None
    return score_chart(next(iter(up)), data_1m=data_1m, **kwargs)


# ---------------------------------------------------------------------
# standalone test entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    import argparse
    import sys

    ap = argparse.ArgumentParser(
        description="Read a TradingView screenshot and score it against the rules")
    ap.add_argument("image", help="path to the screenshot")
    ap.add_argument("--key", help="API key (or set ANTHROPIC_API_KEY)")
    ap.add_argument("--provider", default="anthropic",
                    choices=["anthropic", "openai", "google"])
    ap.add_argument("--model", help="override the default model")
    ap.add_argument("--data-dir", default="data",
                    help="folder holding nq_1m.parquet / es_1m.parquet")
    ap.add_argument("--account", type=float, default=150_000)
    ap.add_argument("--risk", type=float, default=0.015)
    ap.add_argument("--rr", type=float, default=2.0)
    ap.add_argument("--no-data", action="store_true",
                    help="skip the liquidity and opening-range checks")
    a = ap.parse_args()

    if not os.path.exists(a.image):
        sys.exit(f"not found: {a.image}")

    data = None
    if not a.no_data:
        nq = os.path.join(a.data_dir, "nq_1m.parquet")
        es = os.path.join(a.data_dir, "es_1m.parquet")
        if os.path.exists(nq) and os.path.exists(es):
            import pandas as pd
            print("loading market data ...")
            data = {"NQ": pd.read_parquet(nq), "ES": pd.read_parquet(es)}
        else:
            print(f"(no parquets in {a.data_dir} — skipping liquidity checks)")

    try:
        score_chart(a.image, data_1m=data, provider=a.provider,
                    api_key=a.key, model=a.model,
                    account=a.account, risk_pct=a.risk, rr=a.rr)
    except Exception as exc:
        sys.exit(f"\n{type(exc).__name__}: {exc}")
