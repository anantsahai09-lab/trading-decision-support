# Deploying

## GitHub

```bash
cd trading-decision-support
git init
git add .
git commit -m "Trading decision-support tool"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git push -u origin main
```

`.gitignore` already excludes the parquets, `.env`, and Streamlit secrets. Check
before the first push:

```bash
git status --short          # no .parquet should appear
du -sh .git                 # should be a few MB, not hundreds
```

---

## Railway

1. railway.app → New Project → Deploy from GitHub repo
2. Pick the repo. Nixpacks reads `requirements.txt` and `runtime.txt`.
3. `railway.json` supplies the start command; no manual config needed.
4. Settings → Networking → Generate Domain

Environment variables (Settings → Variables):

| Key | Purpose |
|---|---|
| `ANTHROPIC_API_KEY` | screenshot reading (optional) |
| `TRADINGBOT_DATA` | data directory, defaults to `data` |

---

## Streamlit Cloud

1. share.streamlit.io → New app
2. Repo, branch `main`, main file `app.py`
3. Advanced settings → Secrets:

```toml
ANTHROPIC_API_KEY = "sk-ant-..."
```

---

## Data on a deployed instance

The parquets are ~110 MB and are not in the repo. Three options:

**Upload in the browser.** The sidebar accepts both files. Simplest, but they
are re-uploaded whenever the instance restarts.

**Cloud storage.** Put them in S3 / R2 / GCS and fetch on boot:

```python
import os, urllib.request
os.makedirs("data", exist_ok=True)
for name in ("nq_1m.parquet", "es_1m.parquet"):
    if not os.path.exists(f"data/{name}"):
        urllib.request.urlretrieve(f"{os.environ['DATA_URL']}/{name}", f"data/{name}")
```

**Git LFS.** Works, but both hosts have ephemeral filesystems and LFS bandwidth
is metered — cloud storage is cleaner.

---

## Local development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
streamlit run app.py
```

Long backtests belong in the terminal, not the browser — a ten-year run takes
longer than a browser session will reliably stay open:

```bash
python cli.py backtest --start 2017-01-01 --end 2021-12-31 --out trades.csv
```
