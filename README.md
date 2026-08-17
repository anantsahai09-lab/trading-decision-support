# Trading Decision-Support Tool

A rules engine for a discretionary futures strategy (NQ / ES). It surfaces
qualifying setups, explains why every candidate passed or failed, and measures
how those rules would have performed historically.

**Decision support, not automation.** No orders are ever placed.

---

## Quick start

```bash
git clone <your-repo-url>
cd trading-decision-support
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python cli.py check
```

Put `nq_1m.parquet` and `es_1m.parquet` in `data/`, or build them from a raw
Databento pull:

```bash
python cli.py data --raw-nq NQ.csv.zst --raw-es ES.csv.zst --build-ml
```

Run the web app:

```bash
streamlit run app.py
```

---

## Layout

```
├── app.py              Streamlit UI (deploy target)
├── cli.py              command line — use this for long backtests
├── src/
│   ├── swings.py           §2  structure, BOS, trend
│   ├── opening_range.py    §3  opening range, double test, single sweep
│   ├── amd.py              §7  accumulation / manipulation / distribution
│   ├── sessions.py         §5  session liquidity, §9 gate
│   ├── confluences.py      §6  FVG, IFVG, SMT
│   ├── orchestrator.py     §1 §4 §8  the decision pipeline
│   ├── backtest.py             fill simulation and metrics
│   ├── data_loader.py          Databento → continuous contract
│   ├── dataset.py              ML feature engineering
│   ├── ml_pipeline.py          five-model pipeline
│   ├── manual.py               score a chart read you type in
│   └── chart_reader.py         vision model → levels → rules
└── tests/              89 tests
```

---

## The strategy

| § | Rule | Status |
|---|---|---|
| 1 | Entry window 09:50–12:00 ET | done |
| 2 | Order flow bias, 15m | done |
| 3 | Opening range from the 09:30/09:35 wicks | done |
| 3.1 | Double test **and** first sweep-and-return | done |
| 4 | Leading index selection | **proxy — probably wrong** |
| 5 | Asia / London / previous-day liquidity | done |
| 6 | FVG, IFVG, SMT, 1m BOS confluences | done |
| 7 | **AMD** — accumulation, manipulation, distribution | done |
| 8 | Stops, 2:1 targets, sizing, daily limits | done |
| 9 | Skip conditions | done |

**AMD is the single largest factor measured.** It treats the session as a shape:
price accumulates in the opening range, manipulates out of it, then distributes
in the opposite direction. Days with no manipulation, or a manipulation that
never returns inside, are not AMD days and are skipped entirely.

Controlled comparison — same window, same year, AMD the only variable:

| Year | Config | Trades | Win | Expectancy |
|---|---|---|---|---|
| 2016 | AMD off | 39 | 35.9% | −0.120R |
| 2016 | **AMD on** | 20 | **50.0%** | **+0.274R** |
| 2017 | AMD off | 98 | 25.5% | −0.422R |
| 2017 | **AMD on** | 35 | **42.9%** | **+0.072R** |

---

## Measured results

**AMD + double-test trigger, 2016–2021, 2:1 RR**

| | |
|---|---|
| Trades | 240 |
| Win rate | 52.1% |
| Profit factor | 1.45 |
| Expectancy | +0.409R |
| Net | +$80,010 |
| Years positive | 6 of 6 |
| Frequency | 0.85 / week |

**Adding the single-sweep trigger, 2016 H2**

| | |
|---|---|
| Trades | 74 |
| Win rate | 52.7% |
| Profit factor | **1.63** |
| Expectancy | +0.427R |
| Frequency | **2.93 / week** |

Trade count more than tripled while win rate and expectancy *rose* — the
double-test requirement was discarding valid setups rather than filtering bad
ones.

2017–2021 in that configuration has not been run yet. That is the number worth
having:

```bash
python cli.py backtest --start 2017-01-01 --end 2021-12-31 --out trades.csv
```

---

## On win rate

Win rate alone is not evidence of an edge — it is a function of target distance.
Same signals, only the target moved:

| RR | Win rate | Breakeven | Net expectancy |
|---|---|---|---|
| 0.5:1 | 64.9% | 66.7% | −0.132R |
| 1.0:1 | 54.8% | 50.0% | −0.027R |
| 2.0:1 | 52.7% | 33.3% | **+0.427R** |

65% is reachable at 0.5:1 — below its own breakeven, losing faster than any
other setting. Every performance table here reports expectancy and profit factor
alongside win rate for that reason.

---

## ML pipeline

Five model types predicting **volatility expansion** — whether the next 30
minutes move more than 1.62 × ATR. 300,101 samples, 17 features, chronological
split.

| Model | Accuracy |
|---|---|
| Baseline (majority class) | 48.2% |
| Linear (LogisticRegression) | 57.3% |
| KNN | 56.3% |
| Decision tree | 56.4% |
| Random forest | 57.2% |
| Neural net (MLP) | **57.6%** |

Best `max_depth`: 8 for the tree, 12 for the forest. Top feature is
`minutes_from_open` at 44% importance — time of day dominates, which is the
kill-zone concept confirmed from data.

Return prediction fails (every R² negative) and that is reported rather than
tuned away: returns are near-unpredictable, but volatility clusters.

**The ML gates no trades.** It runs as a separate analysis track.

```bash
python cli.py ml --sample 60000
```

---

## Deploy

**Railway**

1. Push to GitHub
2. New Project → Deploy from GitHub repo
3. Railway reads `railway.json` and starts Streamlit automatically

**Streamlit Cloud**

1. Push to GitHub
2. share.streamlit.io → New app → point at `app.py`

**Data on a deployed instance.** The parquets are ~110 MB combined and are
gitignored: GitHub caps files at 100 MB, and both hosts have ephemeral
filesystems, so baking data into the image is slow and fragile. Upload through
the app's sidebar, or serve the files from cloud storage and set `DATA_URL`.

---

## Known limitations

**§4 leading-index selection is a proxy.** "Fewest opening-range breaches"
stands in for "cleaner price action" and may be backwards — a choppy instrument
that stays inside its range scores as clean while a smoothly trending one that
broke out early scores as dirty. Flagged as the weakest rule in the system.

**Not validated against live trades.** No recap day has been reproduced
end-to-end, so there is no evidence yet that the encoded rules match the strategy
as actually traded. This remains the highest-value next step: run a day you
traded through `cli.py today` and compare.

**The screenshot reader transcribes, it does not judge.** A vision model reads
printed text — ticker, axis prices, your drawn levels — reliably. Anything
requiring inference from pixels is a read, not a transcription, and should be
confirmed by eye.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

89 tests covering swing detection, opening range, session liquidity and
confluences. The no-lookahead guarantees have dedicated tests — a swing cannot
be broken on the bar that confirms it, and a label never uses a bar its features
also saw.
