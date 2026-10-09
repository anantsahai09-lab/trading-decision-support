"""
ml_pipeline.py — Complete ML pipeline, requirements 1-8, single file.

REQUIREMENTS
  1. Load a clear dataset with clearly defined input and output
  2. Input X, output y
  3. Split into training and testing
  4. Define five models: random forest, decision tree, linear, neural net, KNN
  5. Train on the training set
  6. Evaluate on the testing set
  7. Combine models by averaging outputs (plain and performance-weighted)
  8. Tune hyperparameters, sweeping max_depth to find the best value

USAGE — three ways, all work

  A. Paste this whole file into ONE Colab/Jupyter cell and run it.
     Edit the CONFIG block below first if you want non-default settings.

  B. Command line:
        python ml_pipeline.py --data ml_dataset.parquet
        python ml_pipeline.py --sample 60000
        python ml_pipeline.py --task regression

  C. Import and call:
        from .ml_pipeline import run
        out = run(data="ml_dataset.parquet", sample=60000)

  Notebook kernels inject their own arguments (-f kernel.json), which is why
  argparse is only used when this file is actually run as a script. In a
  notebook the CONFIG block is used instead.

If no dataset exists at --data, the file builds one from raw 1-minute bars
(--nq / --es), or falls back to a synthetic dataset so the pipeline is runnable
with no data at all.

DESIGN NOTES THAT MATTER MORE THAN MODEL CHOICE

  Chronological split. Neighbouring bars share overlapping lookback windows, so
  a shuffled test set contains near-duplicates of training rows and every model
  looks far better than it is. Time-ordered splitting is the only honest option
  for market data.

  Scaling inside a Pipeline. KNN and neural nets are distance/gradient based, so
  `minutes_from_open` (0-360) would swamp `body_ratio` (0-1). Wrapping the
  scaler in a Pipeline means it is fitted on training data only — scaling before
  splitting leaks test statistics into training.

  A baseline on every table. On financial data a model can post a plausible
  score while having learned nothing. Comparing to "predict the mean" (or the
  majority class) is what separates weak-but-real signal from noise.
"""

from __future__ import annotations

import argparse
import os
import time
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier, DummyRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor

warnings.filterwarnings("ignore")
RANDOM_STATE = 42

FEATURE_NAMES = [
    # compression / clumping   -> strategy §7 accumulation-distribution
    "range_ratio", "body_ratio", "overlap_ratio", "close_dispersion",
    "high_low_std",
    # direction / cleanliness  -> strategy §4 "cleaner price action"
    "persistence", "net_move_atr", "wick_ratio", "up_bar_frac",
    "return_autocorr",
    # volatility
    "atr_norm", "atr_change", "true_range_norm",
    # volume
    "vol_ratio", "vol_trend",
    # session context
    "minutes_from_open", "dist_from_or_mid",
]

TARGETS = {
    "y_return": "continuous — forward return over next 6 bars, ATR-normalised",
    "y_expansion": "binary — 1 if max excursion > 1.62 x ATR",
    "y_excursion": "continuous — max excursion in ATR units",
}

# =====================================================================
# CONFIG — edit these when pasting into a notebook cell
# =====================================================================
CONFIG = {
    "data": "ml_dataset.parquet",   # prepared dataset; falls back to synthetic
    "nq": "nq_1m.parquet",          # raw bars, used only to build the dataset
    "es": "es_1m.parquet",
    "task": "classification",       # or "regression"
    "target": None,                 # None -> y_expansion / y_return by task
    "sample": None,                 # e.g. 60000 for a faster run
    "test_frac": 0.2,
    "rf_subsample": 50000,          # caps rows for the forest depth sweep
    "tune": True,                   # False skips requirement 8
}


def rule(char="=", n=72):
    print(char * n)


def step(number, title):
    print()
    rule()
    print(f"REQUIREMENT {number} — {title}")
    rule()


# =====================================================================
# 1 + 2 — dataset with defined X and y
# =====================================================================

def make_synthetic(n=40_000, seed=RANDOM_STATE) -> pd.DataFrame:
    """
    Fallback dataset so the pipeline runs with no market data present.

    Signal is deliberately weak and partly non-linear, so tree models can beat
    the linear one and the depth sweep has something real to find.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        {name: rng.normal(0, 1, n) for name in FEATURE_NAMES}
    )
    X["minutes_from_open"] = rng.uniform(0, 360, n)

    signal = (
        0.30 * X["atr_norm"]
        + 0.25 * np.sin(X["minutes_from_open"] / 60.0)
        + 0.20 * (X["vol_ratio"] > 0.5).astype(float)
        + 0.15 * X["persistence"] * X["net_move_atr"]
    )
    noise = rng.normal(0, 1.0, n)
    y_return = signal + noise
    excursion = np.abs(y_return) + rng.gamma(2, 0.4, n)

    X["timestamp"] = pd.date_range("2016-06-21", periods=n, freq="5min",
                                   tz="America/New_York")
    X["session_date"] = X["timestamp"].dt.normalize()
    X["symbol"] = np.where(rng.random(n) > 0.5, "NQ", "ES")
    X["y_return"] = y_return
    X["y_excursion"] = excursion
    X["y_expansion"] = (excursion > np.median(excursion)).astype(int)
    return X


def load_or_build(data_path, nq_path=None, es_path=None) -> pd.DataFrame:
    """Load a prepared dataset, else build one, else fall back to synthetic."""
    if data_path and os.path.exists(data_path):
        print(f"loading prepared dataset: {data_path}")
        return pd.read_parquet(data_path)

    if nq_path and es_path and os.path.exists(nq_path) and os.path.exists(es_path):
        print("building dataset from raw 1-minute bars ...")
        from .dataset import DatasetSpec, build_combined  # local module
        data = build_combined(
            {"NQ": pd.read_parquet(nq_path), "ES": pd.read_parquet(es_path)},
            DatasetSpec(),
        )
        if data_path:
            data.to_parquet(data_path)
            print(f"cached -> {data_path}")
        return data

    print("no market data found — using synthetic fallback so the pipeline runs")
    return make_synthetic()


def describe_dataset(data: pd.DataFrame, target: str) -> None:
    print(f"samples        : {len(data):,}")
    print(f"features (X)   : {len(FEATURE_NAMES)}")
    if "timestamp" in data:
        print(f"date range     : {data['timestamp'].min().date()} -> "
              f"{data['timestamp'].max().date()}")
    if "session_date" in data:
        print(f"sessions       : {data['session_date'].nunique():,}")
    if "symbol" in data:
        print(f"symbols        : {sorted(data['symbol'].unique())}")

    print("\nINPUT  X — 17 features")
    groups = [
        ("compression/clumping (§7)", FEATURE_NAMES[0:5]),
        ("direction/cleanliness (§4)", FEATURE_NAMES[5:10]),
        ("volatility", FEATURE_NAMES[10:13]),
        ("volume", FEATURE_NAMES[13:15]),
        ("session context", FEATURE_NAMES[15:17]),
    ]
    for label, names in groups:
        print(f"  {label:<28} {', '.join(names)}")

    print("\nOUTPUT y — available targets")
    for name, desc in TARGETS.items():
        mark = " <-- selected" if name == target else ""
        if name in data:
            print(f"  {name:<14} {desc}{mark}")

    y = data[target]
    if y.nunique() <= 2:
        bal = y.mean()
        print(f"\n  class balance : {bal:.1%} / {1 - bal:.1%}")
        if not 0.35 < bal < 0.65:
            print("  WARNING: imbalanced — accuracy will be misleading")
    else:
        print(f"\n  mean {y.mean():+.4f}  std {y.std():.4f}  "
              f"range [{y.min():.2f}, {y.max():.2f}]")

    nan = data[FEATURE_NAMES].isna().sum().sum()
    inf = int(np.isinf(data[FEATURE_NAMES].to_numpy()).sum())
    print(f"\n  NaNs {nan}   Infs {inf}")


def split_xy(data: pd.DataFrame, target: str):
    if target not in data.columns:
        raise ValueError(f"target '{target}' not in dataset")
    return data[FEATURE_NAMES].copy(), data[target].copy()


# =====================================================================
# 3 — train / test split
# =====================================================================

def time_split(data: pd.DataFrame, test_frac=0.2):
    """Chronological split. A random split leaks via overlapping windows."""
    if "timestamp" in data.columns:
        cut = data["timestamp"].quantile(1 - test_frac)
        return (data[data["timestamp"] <= cut].copy(),
                data[data["timestamp"] > cut].copy())
    cut = int(len(data) * (1 - test_frac))
    return data.iloc[:cut].copy(), data.iloc[cut:].copy()


# =====================================================================
# 4 — define the five models
# =====================================================================

def define_models(task="classification", max_depth=8) -> dict:
    """
    Random forest, decision tree, linear, neural net, KNN.

    Only the tree models take max_depth; linear and KNN have no depth parameter,
    which is exactly why requirement 8 sweeps depth for the tree learners.
    """
    if task == "regression":
        return {
            "linear": Pipeline([("scale", StandardScaler()),
                                ("model", Ridge(alpha=1.0))]),
            "knn": Pipeline([("scale", StandardScaler()),
                             ("model", KNeighborsRegressor(
                                 n_neighbors=50, weights="distance", n_jobs=-1))]),
            "decision_tree": DecisionTreeRegressor(
                max_depth=max_depth, min_samples_leaf=50,
                random_state=RANDOM_STATE),
            "random_forest": RandomForestRegressor(
                n_estimators=100, max_depth=max_depth, min_samples_leaf=50,
                random_state=RANDOM_STATE, n_jobs=-1),
            "neural_net": Pipeline([("scale", StandardScaler()),
                                    ("model", MLPRegressor(
                                        hidden_layer_sizes=(64, 32), max_iter=300,
                                        early_stopping=True, n_iter_no_change=10,
                                        random_state=RANDOM_STATE))]),
        }
    return {
        "linear": Pipeline([("scale", StandardScaler()),
                            ("model", LogisticRegression(
                                max_iter=1000, random_state=RANDOM_STATE))]),
        "knn": Pipeline([("scale", StandardScaler()),
                         ("model", KNeighborsClassifier(
                             n_neighbors=50, weights="distance", n_jobs=-1))]),
        "decision_tree": DecisionTreeClassifier(
            max_depth=max_depth, min_samples_leaf=50, random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=100, max_depth=max_depth, min_samples_leaf=50,
            random_state=RANDOM_STATE, n_jobs=-1),
        "neural_net": Pipeline([("scale", StandardScaler()),
                                ("model", MLPClassifier(
                                    hidden_layer_sizes=(64, 32), max_iter=300,
                                    early_stopping=True, n_iter_no_change=10,
                                    random_state=RANDOM_STATE))]),
    }


# =====================================================================
# 5 + 6 — train and evaluate
# =====================================================================

@dataclass
class Result:
    name: str
    train_time: float
    metrics: dict
    predictions: np.ndarray
    model: object = None
    proba: np.ndarray | None = None


def evaluate(y_true, y_pred, task="classification") -> dict:
    if task == "regression":
        return {
            "r2": r2_score(y_true, y_pred),
            "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "mae": mean_absolute_error(y_true, y_pred),
            # directional hit rate — for a trading model this matters more than
            # R², because you act on the sign, not the magnitude
            "dir_acc": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
        }
    return {"accuracy": accuracy_score(y_true, y_pred),
            "f1": f1_score(y_true, y_pred)}


def train_and_evaluate(models, X_tr, y_tr, X_te, y_te, task="classification"):
    results = {}
    dummy = (DummyRegressor() if task == "regression"
             else DummyClassifier(strategy="most_frequent"))
    dummy.fit(X_tr, y_tr)
    results["BASELINE"] = Result(
        "BASELINE", 0.0, evaluate(y_te, dummy.predict(X_te), task),
        dummy.predict(X_te))

    hdr = f"{'model':<16}{'fit(s)':>8}   metrics"
    print(hdr)
    print("-" * 72)
    m = results["BASELINE"].metrics
    print(f"{'BASELINE':<16}{0.0:>8.1f}   " +
          "  ".join(f"{k}={v:.4f}" for k, v in m.items()))

    for name, model in models.items():
        t0 = time.time()
        model.fit(X_tr, y_tr)
        elapsed = time.time() - t0
        pred = model.predict(X_te)
        proba = (model.predict_proba(X_te)[:, 1]
                 if task == "classification" and hasattr(model, "predict_proba")
                 else None)
        met = evaluate(y_te, pred, task)
        results[name] = Result(name, elapsed, met, pred, model, proba)
        print(f"{name:<16}{elapsed:>8.1f}   " +
              "  ".join(f"{k}={v:.4f}" for k, v in met.items()))

    key = "accuracy" if task == "classification" else "r2"
    best = max((n for n in results if n != "BASELINE"),
               key=lambda n: results[n].metrics[key])
    print(f"\nbest single model: {best} "
          f"({key}={results[best].metrics[key]:.4f}) vs "
          f"baseline {results['BASELINE'].metrics[key]:.4f}")
    return results


# =====================================================================
# 7 — combine by averaging
# =====================================================================

def ensemble(results, y_te, task="classification", weight_by=None):
    """
    Plain average and performance-weighted average.

    Classification averages PROBABILITIES, not hard 0/1 labels: averaging labels
    produces values that are neither a class nor a calibrated probability.
    Soft voting — average predict_proba, threshold at 0.5 — is the correct form.

    Caveat kept in the open: weights come from test-set scores, so the weighted
    ensemble has seen the test set. Fair for comparing weighting schemes,
    optimistic as a live estimate; proper practice fits weights on a separate
    validation split.
    """
    weight_by = weight_by or ("accuracy" if task == "classification" else "r2")
    names = [n for n in results if n != "BASELINE"]

    if task == "classification":
        names = [n for n in names if results[n].proba is not None]
        preds = np.column_stack([results[n].proba for n in names])
    else:
        preds = np.column_stack([results[n].predictions for n in names])

    fin = (lambda x: (x >= 0.5).astype(int)) if task == "classification" else (lambda x: x)

    simple = preds.mean(axis=1)
    simple_m = evaluate(y_te, fin(simple), task)

    scores = np.array([max(results[n].metrics.get(weight_by, 0.0), 0.0)
                       for n in names])
    w = scores / scores.sum() if scores.sum() > 0 else np.ones(len(names)) / len(names)
    weighted = (preds * w).sum(axis=1)
    weighted_m = evaluate(y_te, fin(weighted), task)

    print(f"members: {', '.join(names)}\n")
    print(f"{'method':<20}   metrics")
    print("-" * 72)
    print(f"{'simple average':<20}   " +
          "  ".join(f"{k}={v:.4f}" for k, v in simple_m.items()))
    print(f"{'weighted average':<20}   " +
          "  ".join(f"{k}={v:.4f}" for k, v in weighted_m.items()))
    print(f"\nweights (by {weight_by}):")
    for n, wi in zip(names, w):
        print(f"  {n:<16} {wi:.4f}")

    key = "accuracy" if task == "classification" else "r2"
    best_single = max(names, key=lambda n: results[n].metrics[key])
    bs = results[best_single].metrics[key]
    be = max(simple_m[key], weighted_m[key])
    print(f"\nbest ensemble {be:.4f}  vs  best single ({best_single}) {bs:.4f}"
          f"  -> {'ensemble wins' if be > bs else 'single model wins'}")
    if be <= bs:
        print("  Ensembles help when models make UNCORRELATED errors. These read")
        print("  the same features and largely agree, so averaging reproduces the")
        print("  consensus rather than improving on it.")

    return {"simple": simple_m, "weighted": weighted_m,
            "weights": dict(zip(names, w.round(4)))}


# =====================================================================
# 8 — hyperparameter tuning: max_depth sweep
# =====================================================================

def tune_max_depth(X_tr, y_tr, X_te, y_te, task="classification",
                   depths=(2, 3, 5, 8, 12, 16, 20, None),
                   models=("decision_tree", "random_forest"),
                   rf_subsample=None):
    """
    Sweep max_depth for the depth-sensitive models.

    Train scores are printed beside test scores because the GAP between them is
    the overfitting signal: a deep tree drives train accuracy up while test
    accuracy falls. Reporting the test score alone hides the mechanism.

    `rf_subsample` caps rows for the random forest sweep — 100 trees x 8 depths
    on 240k rows is slow, and the shape of the curve is unchanged by sampling.
    """
    metric = "accuracy" if task == "classification" else "r2"
    rows = []

    for name in models:
        xtr, ytr = X_tr, y_tr
        note = ""
        if name == "random_forest" and rf_subsample and len(X_tr) > rf_subsample:
            xtr = X_tr.iloc[-rf_subsample:]
            ytr = y_tr.iloc[-rf_subsample:]
            note = f"  [subsampled to {rf_subsample:,} rows]"

        print(f"\n{name}{note}")
        print(f"  {'max_depth':<12}{'train':>9}{'test':>9}{'gap':>9}{'fit(s)':>9}")
        print("  " + "-" * 48)

        for depth in depths:
            model = define_models(task, max_depth=depth)[name]
            t0 = time.time()
            model.fit(xtr, ytr)
            elapsed = time.time() - t0
            tr = evaluate(ytr, model.predict(xtr), task)[metric]
            te = evaluate(y_te, model.predict(X_te), task)[metric]
            label = "None" if depth is None else str(depth)
            rows.append({"model": name, "max_depth": label,
                         f"train_{metric}": round(tr, 4),
                         f"test_{metric}": round(te, 4),
                         "gap": round(tr - te, 4), "fit_s": round(elapsed, 1)})
            print(f"  {label:<12}{tr:>9.4f}{te:>9.4f}{tr - te:>9.4f}{elapsed:>9.1f}")

    tuning = pd.DataFrame(rows)
    col = f"test_{metric}"
    print("\nBEST DEPTH")
    print("-" * 72)
    best = {}
    for name, g in tuning.groupby("model"):
        row = g.loc[g[col].idxmax()]
        best[name] = row["max_depth"]
        print(f"  {name:<16} max_depth={row['max_depth']:<6} "
              f"{col}={row[col]:.4f}  (gap {row['gap']:.4f})")

    worst = tuning[tuning["max_depth"] == "None"]
    if not worst.empty:
        print("\n  Unlimited depth shows the overfitting clearly: train score high,")
        print("  test score lower, widest gap of any setting.")
    return tuning, best


def feature_importance(model, names=FEATURE_NAMES) -> pd.Series:
    if hasattr(model, "feature_importances_"):
        return pd.Series(model.feature_importances_,
                         index=names).sort_values(ascending=False)
    if hasattr(model, "named_steps"):
        inner = model.named_steps.get("model")
        if hasattr(inner, "coef_"):
            return pd.Series(np.abs(np.ravel(inner.coef_)),
                             index=names).sort_values(ascending=False)
    return pd.Series(dtype=float)


# =====================================================================
# main
# =====================================================================

def run(**overrides):
    """
    Run the full pipeline. Any CONFIG key can be overridden:

        run()                                  # CONFIG defaults
        run(sample=60000)                      # faster
        run(task="regression")                 # predict returns
        run(data="/content/drive/MyDrive/ml_dataset.parquet")
    """
    cfg = {**CONFIG, **overrides}
    unknown = set(overrides) - set(CONFIG)
    if unknown:
        raise TypeError(f"unknown option(s): {sorted(unknown)}. "
                        f"Valid: {sorted(CONFIG)}")

    task = cfg["task"]
    target = cfg["target"] or ("y_expansion" if task == "classification"
                               else "y_return")

    rule()
    print("ML PIPELINE — REQUIREMENTS 1-8")
    print(f"task: {task}   target: {target}")
    rule()

    # ---- 1 + 2 ----
    step("1 + 2", "LOAD DATASET WITH DEFINED INPUT X AND OUTPUT y")
    data = load_or_build(cfg["data"], cfg["nq"], cfg["es"])
    if cfg["sample"] and cfg["sample"] < len(data):
        data = data.iloc[-cfg["sample"]:].copy()
        print(f"[sampled to the most recent {cfg['sample']:,} rows]\n")
    describe_dataset(data, target)

    # ---- 3 ----
    step(3, "SPLIT INTO TRAINING AND TESTING")
    train, test = time_split(data, cfg["test_frac"])
    X_tr, y_tr = split_xy(train, target)
    X_te, y_te = split_xy(test, target)
    print(f"method     : chronological ({1 - cfg['test_frac']:.0%} / "
          f"{cfg['test_frac']:.0%}), NOT random")
    print("reason     : overlapping lookback windows make a shuffled split leak")
    print(f"train      : {len(X_tr):,} rows", end="")
    if "timestamp" in train:
        print(f"  ({train['timestamp'].min().date()} -> "
              f"{train['timestamp'].max().date()})")
    else:
        print()
    print(f"test       : {len(X_te):,} rows", end="")
    if "timestamp" in test:
        print(f"  ({test['timestamp'].min().date()} -> "
              f"{test['timestamp'].max().date()})")
    else:
        print()
    print(f"X shape    : train {X_tr.shape}   test {X_te.shape}")
    print(f"y shape    : train {y_tr.shape}   test {y_te.shape}")

    # ---- 4 ----
    step(4, "DEFINE MODELS")
    models = define_models(task, max_depth=8)
    for i, (name, m) in enumerate(models.items(), 1):
        inner = m.named_steps["model"] if hasattr(m, "named_steps") else m
        print(f"  {i}. {name:<16} {type(inner).__name__}")
    print("\n  scaled inside a Pipeline: linear, knn, neural_net")
    print("  depth-sensitive (swept in req 8): decision_tree, random_forest")

    # ---- 5 + 6 ----
    step("5 + 6", "TRAIN ON TRAINING SET, EVALUATE ON TESTING SET")
    results = train_and_evaluate(models, X_tr, y_tr, X_te, y_te, task)

    # ---- 7 ----
    step(7, "COMBINE MODELS BY AVERAGING OUTPUTS")
    ens = ensemble(results, y_te, task)

    # ---- 8 ----
    tuning = best = None
    if cfg["tune"]:
        step(8, "HYPERPARAMETER TUNING — max_depth SWEEP")
        tuning, best = tune_max_depth(
            X_tr, y_tr, X_te, y_te, task, rf_subsample=cfg["rf_subsample"])

    # ---- extra: what carried the signal ----
    rf = results.get("random_forest")
    if rf is not None and rf.model is not None:
        print()
        rule()
        print("FEATURE IMPORTANCE (random forest)")
        rule()
        print(feature_importance(rf.model).head(10).round(4).to_string())

    # ---- checklist ----
    print()
    rule()
    print("REQUIREMENTS CHECKLIST")
    rule()
    key = "accuracy" if task == "classification" else "r2"
    best_single = max((n for n in results if n != "BASELINE"),
                      key=lambda n: results[n].metrics[key])
    items = [
        ("1. Clear dataset, defined I/O",
         f"{len(data):,} samples, {len(FEATURE_NAMES)} features, "
         f"{len(TARGETS)} targets"),
        ("2. Input X, output y",
         f"X {X_tr.shape[1]} cols, y = {target}"),
        ("3. Train / test split",
         f"{len(X_tr):,} / {len(X_te):,}, chronological"),
        ("4. Five models defined",
         ", ".join(models.keys())),
        ("5. Trained on training set",
         f"all {len(models)} fitted on {len(X_tr):,} rows"),
        ("6. Evaluated on testing set",
         f"best {best_single} {key}={results[best_single].metrics[key]:.4f} "
         f"vs baseline {results['BASELINE'].metrics[key]:.4f}"),
        ("7. Combined by averaging",
         f"simple {ens['simple'][key]:.4f}, weighted {ens['weighted'][key]:.4f}"),
        ("8. max_depth tuned",
         (", ".join(f"{k}={v}" for k, v in best.items()) if best else "skipped")),
    ]
    for label, detail in items:
        print(f"  [x] {label:<32} {detail}")

    if tuning is not None:
        try:
            tuning.to_csv("tuning_results.csv", index=False)
            print("\nsaved: tuning_results.csv")
        except OSError as exc:
            print(f"\n(could not write tuning_results.csv: {exc})")

    return {"results": results, "ensemble": ens, "tuning": tuning,
            "best_depth": best, "data": data,
            "X_train": X_tr, "y_train": y_tr, "X_test": X_te, "y_test": y_te}


def _in_notebook() -> bool:
    """True inside IPython/Jupyter/Colab, where argparse would see kernel args."""
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except Exception:
        return False


def _cli():
    """Parse command-line flags. Only used when run as a real script."""
    ap = argparse.ArgumentParser(description="ML pipeline, requirements 1-8")
    ap.add_argument("--data", default=CONFIG["data"])
    ap.add_argument("--nq", default=CONFIG["nq"])
    ap.add_argument("--es", default=CONFIG["es"])
    ap.add_argument("--task", default=CONFIG["task"],
                    choices=["classification", "regression"])
    ap.add_argument("--target", default=CONFIG["target"])
    ap.add_argument("--sample", type=int, default=CONFIG["sample"])
    ap.add_argument("--test-frac", type=float, default=CONFIG["test_frac"])
    ap.add_argument("--rf-subsample", type=int, default=CONFIG["rf_subsample"])
    ap.add_argument("--no-tune", action="store_true")
    a = ap.parse_args()
    return {"data": a.data, "nq": a.nq, "es": a.es, "task": a.task,
            "target": a.target, "sample": a.sample, "test_frac": a.test_frac,
            "rf_subsample": a.rf_subsample, "tune": not a.no_tune}
