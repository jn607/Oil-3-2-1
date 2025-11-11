"""Utility functions for the oil analysis notebook.

Provides a minimal but functional implementation of the functions used by
`oil_analysis.ipynb`:
 - load_price_history(csv_path, today)
 - compute_crack(prices)
 - compute_percentile(series, value)
 - plot_crack_time_series(prices, today, outpath)
 - build_scenario_table(crude, rbob, ulsd)
 - write_market_note(prices, today, note_path, citation_refs)
 - write_fundamentals_pdf(pdf_path, citation_refs)

This module expects the input CSV to contain columns: date, crude, rbob, ulsd
where `date` is parseable by pandas, `crude` is USD per barrel and `rbob`/`ulsd`
are USD per gallon.
"""
from pathlib import Path
from typing import Iterable, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime


def _as_path(p):
    return Path(p) if not isinstance(p, Path) else p


def load_price_history(csv_path, today: pd.Timestamp = None) -> pd.DataFrame:
    """Load a price CSV and return a cleaned DataFrame.

    Expected columns: date, crude, rbob, ulsd
    - date will be parsed as datetime
    - rows will be sorted by date
    - basic forward/backfill will be applied for small gaps
    """
    csv_path = _as_path(csv_path)
    df = pd.read_csv(csv_path, parse_dates=["date"])
    required = {"date", "crude", "rbob", "ulsd"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV at {csv_path} missing required columns: {required - set(df.columns)}")

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)
    # Cast to numeric
    df["crude"] = pd.to_numeric(df["crude"], errors="coerce")
    df["rbob"] = pd.to_numeric(df["rbob"], errors="coerce")
    df["ulsd"] = pd.to_numeric(df["ulsd"], errors="coerce")

    # Fill small gaps linearly, then forward/backfill
    df["crude"] = df["crude"].interpolate(limit=7).ffill().bfill()
    df["rbob"] = df["rbob"].interpolate(limit=7).ffill().bfill()
    df["ulsd"] = df["ulsd"].interpolate(limit=7).ffill().bfill()

    if today is not None:
        # Ensure today is a Timestamp
        today = pd.to_datetime(today)
        # If the latest date is before today, append a copy of the latest row with today's date
        if df["date"].max() < today:
            last = df.iloc[[-1]].copy()
            last.loc[:, "date"] = today
            df = pd.concat([df, last], ignore_index=True)
    return df


def compute_crack(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute a 3-2-1 crack spread in USD per barrel and return a new DataFrame.

    Formula used (typical, on a $/bbl basis):
      crack = (2 * RBOB + 1 * ULSD) * 42 - 3 * crude
    where RBOB and ULSD are provided in USD per gallon and 42 is gallons per barrel.
    """
    df = prices.copy()
    if not {"crude", "rbob", "ulsd", "date"}.issubset(df.columns):
        raise ValueError("DataFrame must contain date, crude, rbob, ulsd columns")

    df["crack"] = (2 * df["rbob"] + 1 * df["ulsd"]) * 42 - 3 * df["crude"]
    return df


def compute_percentile(series: Iterable[float], value: float) -> float:
    """Return the percentile (0-100) of `value` within `series`.

    If the series contains NaNs they are ignored.
    """
    arr = np.asarray(pd.Series(series).dropna())
    if arr.size == 0:
        return np.nan
    # Percentile as percentage of values <= value
    pct = (arr <= value).sum() / arr.size * 100.0
    return float(pct)


def plot_crack_time_series(prices: pd.DataFrame, today, outpath: Path | str):
    """Plot crack time series with percentile bands and annotate today's value.

    Saves a PNG to `outpath`.
    """
    outpath = _as_path(outpath)
    df = prices.copy()
    if "date" not in df.columns or "crack" not in df.columns:
        raise ValueError("prices must include 'date' and 'crack' columns")

    df = df.sort_values("date").reset_index(drop=True)
    df = df.dropna(subset=["crack"]).copy()

    p10 = df["crack"].quantile(0.10)
    p25 = df["crack"].quantile(0.25)
    p50 = df["crack"].quantile(0.50)
    p75 = df["crack"].quantile(0.75)
    p90 = df["crack"].quantile(0.90)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(df["date"], df["crack"], color="#1f77b4", lw=1.5, label="3-2-1 crack")
    # percentile bands (shaded)
    ax.fill_between(df["date"], p10, p90, color="#c6d9f1", alpha=0.3, label="10-90 pct")
    ax.fill_between(df["date"], p25, p75, color="#9ec5f7", alpha=0.35, label="25-75 pct")
    ax.axhline(p50, color="#2ca02c", linestyle="--", linewidth=1, label="Median")

    # annotate today's value if present
    today = pd.to_datetime(today)
    today_row = df.loc[df["date"] == df["date"].max()]
    if not today_row.empty:
        x = today_row["date"].iloc[0]
        y = today_row["crack"].iloc[0]
        ax.scatter([x], [y], color="red", zorder=5)
        ax.annotate(f"{y:.1f} USD/bbl", xy=(x, y), xytext=(5, 5), textcoords="offset points", fontsize=9)

    ax.set_ylabel("USD per barrel")
    ax.set_xlabel("Date")
    ax.set_title("3-2-1 Crack Spread")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    plt.close(fig)


def build_scenario_table(crude: float, rbob: float, ulsd: float) -> pd.DataFrame:
    """Build a simple scenario table for ±$1/bbl crude moves and ±$0.05/gal product moves.

    Returns a DataFrame with columns: crude, rbob, ulsd, crack
    """
    scenarios = []
    crude_moves = [-1.0, 0.0, 1.0]
    prod_moves = [-0.05, 0.0, 0.05]
    for dc in crude_moves:
        for dp in prod_moves:
            c = crude + dc
            r = rbob + dp
            u = ulsd + dp
            crack = (2 * r + u) * 42 - 3 * c
            scenarios.append({"crude": c, "rbob": r, "ulsd": u, "crack": crack})
    df = pd.DataFrame(scenarios)
    # sort for readability
    df = df.sort_values(["crack"], ascending=False).reset_index(drop=True)
    return df


def write_market_note(prices: pd.DataFrame, today, note_path: Path | str, citation_refs: Tuple[str, ...] = ()): 
    """Write a short market note (Markdown) summarising the latest crack and context.

    The note is deliberately concise and intended as a starting point for editing.
    """
    note_path = _as_path(note_path)
    df = prices.copy()
    df = compute_crack(df)
    latest = df.iloc[-1]
    pct = compute_percentile(df["crack"], latest["crack"]) if not pd.isna(latest["crack"]) else float('nan')
    mean = df["crack"].mean()

    lines = []
    lines.append(f"# Market note — 3‑2‑1 crack overview ({pd.to_datetime(latest['date']).date()})\n")
    lines.append(f"Today's 3‑2‑1 crack: **{latest['crack']:.1f} USD/bbl** (percentile: **{pct:.1f}%** vs history).\n")
    lines.append(f"3‑year average (approx): **{mean:.1f} USD/bbl**. This note is automatically generated and should be edited before distribution.\n")
    if citation_refs:
        lines.append("\nReferences:\n")
        for r in citation_refs:
            # If a citation is provided as (label, url) produce a markdown link
            if isinstance(r, (list, tuple)) and len(r) >= 2:
                label, url = r[0], r[1]
                lines.append(f"- [{label}]({url})\n")
            else:
                # fallback: write the raw locator/string
                lines.append(f"- {r}\n")

    note_path.parent.mkdir(parents=True, exist_ok=True)
    note_path.write_text("\n".join(lines), encoding="utf-8")


def write_fundamentals_pdf(pdf_path: Path | str, citation_refs: Iterable[str] = ()): 
    """Create a simple one-page PDF with citation references using matplotlib text.

    This avoids extra PDF dependencies and produces a readable placeholder PDF.
    """
    pdf_path = _as_path(pdf_path)
    fig, ax = plt.subplots(figsize=(8.5, 11))
    ax.axis("off")
    title = "Oil fundamentals — references"
    ax.text(0.5, 0.95, title, ha="center", va="top", fontsize=18)

    y = 0.9
    for i, r in enumerate(citation_refs, start=1):
        ax.text(0.02, y - i * 0.04, f"{i}. {r}", ha="left", va="top", fontsize=10)

    fig.tight_layout()
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, dpi=150)
    plt.close(fig)
