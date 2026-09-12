"""
Data pipeline (roadmap step 3).

Source: Coin Metrics Community Network Data, file csv/btc.csv in
https://github.com/coinmetrics/data (CC BY-NC 4.0). Daily series; PriceUSD is
the Coin Metrics reference rate as of 00:00 UTC of the following day, i.e. the close of the
UTC day, so the day-t return and the day-t exchange flows cover the same interval.
Exchange flows (FlowInExNtv, FlowOutExNtv) give the on-chain exchange net-flow
covariate; realised volatility is computed from the price series itself.

Timing convention (Section 7 of the methodology): every covariate z_t is
measurable at the close of day t; Xlag[t] = (1, z_{t-1}) is the regressor that
drives the transition INTO day t, so that a forecast for day t made at the close
of t-1 uses only F_{t-1}.
"""
from __future__ import annotations

import os
import urllib.request
from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd

COINMETRICS_URL = "https://raw.githubusercontent.com/coinmetrics/data/master/csv/btc.csv"
DEFAULT_RAW = os.path.join(os.path.dirname(__file__), "..", "..", "data", "raw", "coinmetrics_btc.csv")


def download_coinmetrics(path: str = DEFAULT_RAW, force: bool = False) -> str:
    path = os.path.abspath(path)
    if force or not os.path.exists(path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        urllib.request.urlretrieve(COINMETRICS_URL, path)
    return path


def load_btc(path: str = DEFAULT_RAW, start: str = "2014-01-01", end: str | None = None) -> pd.DataFrame:
    """Daily frame with close, simple and log returns, and raw covariate inputs."""
    df = pd.read_csv(download_coinmetrics(path), parse_dates=["time"]).set_index("time").sort_index()
    df = df.loc[start:end]
    out = pd.DataFrame(index=df.index)
    out["close"] = df["PriceUSD"]
    out["flow_in"] = df["FlowInExNtv"]
    out["flow_out"] = df["FlowOutExNtv"]
    out["supply_ex"] = df["SplyExNtv"]
    out["volume_usd"] = df["volume_reported_spot_usd_1d"]
    out = out.dropna(subset=["close"])
    out["ret"] = out["close"].pct_change()
    out["y"] = np.log(out["close"]).diff()
    return out.dropna(subset=["y"])


# ----------------------------------------------------------------------------
# Covariates (all measurable at the close of day t)
# ----------------------------------------------------------------------------
def covariate_logrv(df: pd.DataFrame, window: int = 10) -> pd.Series:
    """z_t = ln sqrt( mean_{s=t-window+1..t} y_s^2 ), Section 6.7."""
    rv = np.sqrt((df["y"] ** 2).rolling(window, min_periods=window).mean())
    return np.log(rv).rename(f"logrv{window}")


def covariate_exflow(df: pd.DataFrame, window: int = 30) -> pd.Series:
    """Exchange net-flow intensity: (inflow - outflow) / rolling mean of total flow.

    Positive values mean coins are moving onto exchanges (potential sell pressure).
    Scaling by total flow removes the secular growth in on-chain activity.
    """
    net = df["flow_in"] - df["flow_out"]
    tot = (df["flow_in"] + df["flow_out"]).rolling(window, min_periods=window).mean()
    return (net / tot).rename(f"exflow{window}")


COVARIATES = {"logrv": covariate_logrv, "exflow": covariate_exflow}


def build_design(df: pd.DataFrame, covariates: list[str] | None, **kw) -> tuple[pd.DataFrame, list[str]]:
    """Attach covariate columns z_t and their lagged versions z_{t-1}. Drops the warm-up rows."""
    names = []
    out = df.copy()
    for name in covariates or []:
        s = COVARIATES[name](out, **kw.get(name, {}))
        out[s.name] = s
        out[s.name + "_lag"] = s.shift(1)
        names.append(s.name)
    return out.dropna(), names


@dataclass
class Window:
    train_start: int
    train_end: int   # exclusive
    test_end: int    # exclusive


def walk_forward_windows(n: int, train_len: int, test_len: int, first_test: int | None = None) -> Iterator[Window]:
    """Rolling windows: train on [b-train_len, b), test on [b, b+test_len)."""
    if train_len < 1 or test_len < 1:
        raise ValueError("train_len and test_len must be positive integers")
    b = train_len if first_test is None else max(first_test, train_len)
    while b < n:
        yield Window(b - train_len, b, min(b + test_len, n))
        b += test_len


class Standardizer:
    """Train-window standardisation of covariates, Section 6.7."""

    def __init__(self):
        self.mean = None
        self.sd = None

    def fit(self, Z: np.ndarray) -> "Standardizer":
        self.mean = Z.mean(axis=0)
        self.sd = Z.std(axis=0) + 1e-12
        return self

    def transform(self, Z: np.ndarray) -> np.ndarray:
        return (Z - self.mean) / self.sd


def design_matrix(df: pd.DataFrame, names: list[str], std: Standardizer | None) -> np.ndarray:
    """Xlag (T, 1 + len(names)) with an intercept and lagged, standardised covariates."""
    T = len(df)
    if not names:
        return np.ones((T, 1))
    Z = df[[n + "_lag" for n in names]].to_numpy(float)
    if std is not None:
        Z = std.transform(Z)
    return np.column_stack([np.ones(T), Z])
