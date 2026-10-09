"""
data_loader.py — Databento GLBX parent-symbol -> continuous front-month series.

WHY THIS EXISTS
    The Databento pull used `stype_in=parent` with symbols like `NQ.FUT`, so the
    file contains EVERY NQ instrument: all outright expirations (NQU6, NQZ6, ...)
    plus calendar spreads (NQU6-NQZ6). Multiple contracts print bars at the same
    timestamp. A backtest needs ONE price series, so a continuous contract has to
    be constructed.

METHOD — volume roll
    1. Drop spread instruments (symbol contains '-').
    2. For each trading day, total the volume per outright contract.
    3. The front month for that day is the contract with the most volume.
    4. Keep only front-month bars, and forbid rolling backwards to an earlier
       expiration once a roll has happened (guards against a thin day flipping
       the series back).

    Volume rolling matches how a trader actually tracks "the active contract" —
    you move when liquidity moves, not on a fixed calendar date.

ROLL GAPS
    Prices jump at a roll (a Sep contract and a Dec contract trade at different
    levels). This strategy is intraday and never holds overnight, so entries and
    exits are unaffected. It DOES affect §5's previous-day high/low on roll days,
    where the prior day's levels came from a different contract. Roll dates are
    recorded so those days can be flagged or skipped.

TIMEZONE
    ts_event is UTC; everything is converted to America/New_York, which is what
    the strategy runs on.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
import pandas as pd

try:
    import zstandard as zstd
except ImportError:  # pragma: no cover
    zstd = None

TZ = "America/New_York"
USE_COLS = ["ts_event", "open", "high", "low", "close", "volume", "symbol"]


# Futures month codes, in calendar order — used to compare expirations.
MONTH_CODES = "FGHJKMNQUVXZ"


def _open_maybe_zst(path: str):
    """Return a file-like object of decompressed text."""
    if str(path).endswith(".zst"):
        if zstd is None:
            raise RuntimeError("zstandard not installed: pip install zstandard")
        fh = open(path, "rb")
        return io.TextIOWrapper(
            zstd.ZstdDecompressor().stream_reader(fh), encoding="utf-8"
        )
    return open(path, "r", encoding="utf-8")


def parse_contract(symbol: str) -> tuple[str, int, int] | None:
    """
    'NQU6' -> ('NQ', 8, 6)   root, month index (0-11), single-digit year.

    Returns None for spreads or anything unparseable.
    """
    if "-" in symbol:
        return None
    s = symbol.strip()
    if len(s) < 3:
        return None
    year_part = s[-1]
    month_part = s[-2]
    root = s[:-2]
    if not year_part.isdigit() or month_part not in MONTH_CODES:
        return None
    return root, MONTH_CODES.index(month_part), int(year_part)


def expiry_rank(symbol: str, ref_year: int) -> float:
    """
    Sortable expiration key. Databento uses a single-digit year, so the decade is
    inferred from the bar's own year — a contract can be at most ~2 years out.
    """
    parsed = parse_contract(symbol)
    if parsed is None:
        return float("inf")
    _, month_idx, yr = parsed
    decade = (ref_year // 10) * 10
    full_year = decade + yr
    # a contract more than 2 years behind the reference belongs to the next decade
    if full_year < ref_year - 1:
        full_year += 10
    return full_year * 12 + month_idx


@dataclass
class LoadReport:
    symbol_root: str
    rows_raw: int
    rows_kept: int
    spreads_dropped: int
    contracts_used: int
    rolls: pd.DataFrame
    start: pd.Timestamp
    end: pd.Timestamp

    def __repr__(self) -> str:  # pragma: no cover
        return (f"LoadReport({self.symbol_root}: {self.rows_kept:,}/{self.rows_raw:,} "
                f"rows, {self.contracts_used} contracts, {len(self.rolls)} rolls)")


def build_continuous(
    path: str,
    chunksize: int = 1_000_000,
    tz: str = TZ,
    verbose: bool = True,
) -> tuple[pd.DataFrame, LoadReport]:
    """
    Stream a Databento parent-symbol OHLCV file into one continuous 1m series.

    Returns (df, report) where df is indexed by tz-aware bar-open timestamp with
    columns open/high/low/close/volume/symbol.
    """
    # ---- pass 1: daily volume per outright contract -------------------
    daily_vol: dict[tuple[pd.Timestamp, str], int] = {}
    rows_raw = spreads = 0

    reader = pd.read_csv(
        _open_maybe_zst(path), usecols=USE_COLS, chunksize=chunksize,
        dtype={"symbol": str},
    )
    for chunk in reader:
        rows_raw += len(chunk)
        is_spread = chunk["symbol"].str.contains("-", regex=False)
        spreads += int(is_spread.sum())
        chunk = chunk[~is_spread]
        if chunk.empty:
            continue
        ts = pd.to_datetime(chunk["ts_event"], utc=True, format="ISO8601")
        local = ts.dt.tz_convert(tz)
        # trading day: bars at/after 18:00 belong to the next session
        day = local.dt.normalize() + pd.to_timedelta(
            (local.dt.hour >= 18).astype(int), unit="D"
        )
        g = (
            pd.DataFrame({"day": day.to_numpy(),
                          "symbol": chunk["symbol"].to_numpy(),
                          "volume": chunk["volume"].to_numpy()})
            .groupby(["day", "symbol"], sort=False)["volume"].sum()
        )
        for (d, s), v in g.items():
            daily_vol[(d, s)] = daily_vol.get((d, s), 0) + int(v)

    if not daily_vol:
        raise ValueError("no outright contract rows found")

    vol = (
        pd.Series(daily_vol)
        .rename_axis(["day", "symbol"])
        .rename("volume")
        .reset_index()
    )

    # ---- choose front month per day, never rolling backwards ----------
    front: dict[pd.Timestamp, str] = {}
    last_rank = -np.inf
    for day, g in vol.groupby("day", sort=True):
        ref_year = int(pd.Timestamp(day).year)
        g = g.copy()
        g["rank"] = [expiry_rank(s, ref_year) for s in g["symbol"]]
        g = g[np.isfinite(g["rank"])]
        eligible = g[g["rank"] >= last_rank]
        pick = (eligible if not eligible.empty else g).sort_values(
            "volume", ascending=False
        ).iloc[0]
        front[day] = pick["symbol"]
        last_rank = max(last_rank, pick["rank"])

    front_s = pd.Series(front).sort_index()
    changes = front_s != front_s.shift(1)
    rolls = pd.DataFrame({
        "day": front_s.index[changes],
        "from": front_s.shift(1)[changes].to_numpy(),
        "to": front_s[changes].to_numpy(),
    }).iloc[1:].reset_index(drop=True)   # first entry is not a roll

    if verbose:
        print(f"  pass 1: {rows_raw:,} rows, {spreads:,} spread rows dropped, "
              f"{front_s.nunique()} contracts, {len(rolls)} rolls")

    # ---- pass 2: keep only front-month bars ---------------------------
    keep_parts = []
    rows_kept = 0
    reader = pd.read_csv(
        _open_maybe_zst(path), usecols=USE_COLS, chunksize=chunksize,
        dtype={"symbol": str},
    )
    for chunk in reader:
        chunk = chunk[~chunk["symbol"].str.contains("-", regex=False)]
        if chunk.empty:
            continue
        ts = pd.to_datetime(chunk["ts_event"], utc=True, format="ISO8601")
        local = ts.dt.tz_convert(tz)
        day = local.dt.normalize() + pd.to_timedelta(
            (local.dt.hour >= 18).astype(int), unit="D"
        )
        want = day.map(front_s)
        mask = (want.to_numpy() == chunk["symbol"].to_numpy())
        sel = chunk[mask]
        if sel.empty:
            continue
        out = pd.DataFrame(
            {
                "open": sel["open"].to_numpy(float),
                "high": sel["high"].to_numpy(float),
                "low": sel["low"].to_numpy(float),
                "close": sel["close"].to_numpy(float),
                "volume": sel["volume"].to_numpy(float),
                "symbol": sel["symbol"].to_numpy(),
            },
            index=local[mask],
        )
        keep_parts.append(out)
        rows_kept += len(out)

    df = pd.concat(keep_parts).sort_index()
    df = df[~df.index.duplicated(keep="last")]
    df.index.name = "timestamp"

    root = parse_contract(str(front_s.iloc[0]))
    report = LoadReport(
        symbol_root=root[0] if root else "?",
        rows_raw=rows_raw,
        rows_kept=len(df),
        spreads_dropped=spreads,
        contracts_used=int(front_s.nunique()),
        rolls=rolls,
        start=df.index.min(),
        end=df.index.max(),
    )
    if verbose:
        print(f"  pass 2: kept {len(df):,} bars  "
              f"{report.start.date()} -> {report.end.date()}")
    return df, report


def data_health_report(df: pd.DataFrame, name: str = "data") -> dict:
    """
    Checks worth running before trusting any backtest built on this file.

    Returns a dict of findings as well as printing them.
    """
    out = {}
    print(f"=== {name} ===")
    print(f"rows            : {len(df):,}")
    print(f"range           : {df.index.min()}  ->  {df.index.max()}")
    print(f"timezone        : {df.index.tz}")

    out["bad_hl"] = int((df["high"] < df["low"]).sum())
    out["open_outside"] = int(((df["open"] > df["high"]) | (df["open"] < df["low"])).sum())
    out["close_outside"] = int(((df["close"] > df["high"]) | (df["close"] < df["low"])).sum())
    out["dupes"] = int(df.index.duplicated().sum())
    print(f"high < low      : {out['bad_hl']}")
    print(f"open outside HL : {out['open_outside']}")
    print(f"close outside HL: {out['close_outside']}")
    print(f"dup timestamps  : {out['dupes']}")

    diffs = df.index.to_series().diff()
    big = diffs[diffs > pd.Timedelta("1h")]
    weekend = big[big > pd.Timedelta("12h")]
    out["gaps_1h"] = len(big)
    out["gaps_12h"] = len(weekend)
    print(f"gaps > 1h       : {len(big):,}   (>12h / weekends: {len(weekend):,})")

    flat = int((df["high"] == df["low"]).sum())
    out["flat_bars"] = flat
    print(f"flat bars       : {flat:,}  ({flat / max(len(df), 1):.2%})")

    hours = set(df.index.hour.unique())
    checks = [
        ("Asia   18:00-03:00", set(range(18, 24)) | set(range(0, 3))),
        ("London 03:00-08:30", set(range(3, 9))),
        ("NY     08:30-17:00", set(range(8, 17))),
    ]
    for label, rng in checks:
        present = bool(rng & hours)
        out[label.split()[0].lower()] = present
        print(f"{label} present: {present}")

    sessions = df.index.normalize().nunique()
    out["sessions"] = sessions
    print(f"distinct dates  : {sessions:,}")
    return out


def resample_ohlcv(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """1m bars -> higher timeframe. rule: '5min', '15min', '1h', '1D'."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df.columns:
        agg["volume"] = "sum"
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna(subset=["open", "high", "low", "close"])


def save_parquet(df: pd.DataFrame, path: str) -> None:
    """Cache the continuous series so the slow rebuild only happens once."""
    df.to_parquet(path)


def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
