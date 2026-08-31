"""Daily Bluesky post: one chart of the lead-day-1 leaderboard.

Optional and token-gated (``BSKY_HANDLE`` + ``BSKY_APP_PASSWORD``, or ``~/.bsky_handle`` and
``~/.bsky_app_password``). With ``--dry-run`` nothing is sent: the chart is written to
``data/raw/bluesky_preview.png`` — under ``raw/``, which is gitignored and excluded from the dataset
push, so a preview never ends up in a data commit — and the post text and alt text are returned.

What is plotted: mean absolute error of the daily maximum temperature at lead day 1, from the 00Z
cycle, bilinear interpolation, averaged over all stations (``station_id == "ALL"``), with the
bootstrap 95 % interval as the error bar. Groups below :data:`castcheck.verify.MIN_N` samples are
drawn in grey and labelled, per METHODOLOGY §4; if no group in the 90-day window reaches that many
samples the next wider window is used and the title says so.
"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR, display_name
from ..verify import MIN_N, PERSISTENCE_ID

SITE = "https://castcheck.zifanzhang.com"
PREVIEW_PATH = DATA_DIR / "raw" / "bluesky_preview.png"
#: Bluesky's post limit is 300 *graphemes*; we stay under it by construction and assert it in tests.
MAX_POST_CHARS = 300

ACCENT = "#1f6feb"
LOW_N_COLOUR = "#9aa4b2"
BASELINE_COLOUR = "#8b949e"
TEXT_COLOUR = "#24292f"
MUTED_COLOUR = "#57606a"

#: Windows tried in order; the first with at least one group of n >= MIN_N wins.
WINDOW_ORDER = ("90d", "30d", "all")
WINDOW_LABEL = {"90d": "last 90 days", "30d": "last 30 days", "365d": "last 365 days", "all": "all data"}

LEAD_DAY = 1
INIT_HOUR = 0
VARIABLE = "tmax"
METHOD = "bilinear"


def c_to_f_delta(celsius: float) -> float:
    """A temperature *difference* in °C expressed in °F (no 32 offset)."""
    return float(celsius) * 9.0 / 5.0


def load_scores() -> pd.DataFrame:
    path = DATA_DIR / "scores" / "latest.parquet"
    return pd.read_parquet(path) if path.exists() else pd.DataFrame()


def select_leaderboard(scores: pd.DataFrame, windows: tuple[str, ...] = WINDOW_ORDER
                       ) -> tuple[pd.DataFrame, str] | None:
    """The rows to plot and the window they came from, or None when nothing is worth posting.

    Selection is fixed (all stations, lead day 1, 00Z, tmax, bilinear); only the window falls back,
    so an early-life dataset still gets a chart instead of an empty one.
    """
    if scores is None or scores.empty:
        return None
    base = scores[(scores["station_id"] == "ALL") & (scores["lead_day"] == LEAD_DAY)
                  & (scores["init_hour"] == INIT_HOUR) & (scores["variable"] == VARIABLE)
                  & (scores["method"] == METHOD)]
    for window in windows:
        sel = base[base["window"] == window].copy()
        sel = sel[sel["n"] > 0].sort_values("mae")
        if len(sel) and (sel["n"] >= MIN_N).any():
            return sel.reset_index(drop=True), window
    return None


def _data_through(sel: pd.DataFrame) -> str:
    ends = pd.to_datetime(sel["period_end"], errors="coerce").dropna()
    return ends.max().strftime("%Y-%m-%d") if len(ends) else datetime.now(UTC).strftime("%Y-%m-%d")


def build_chart(sel: pd.DataFrame, window: str) -> tuple[bytes, str]:
    """Render the leaderboard as a 1200×675 PNG (social-card ratio). Returns (png, alt text)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [display_name(m) for m in sel["model_id"]]
    mae_f = [c_to_f_delta(v) for v in sel["mae"]]
    lo = [max(0.0, c_to_f_delta(m - lo_)) for m, lo_ in zip(sel["mae"], sel["mae_ci_low"])]
    hi = [max(0.0, c_to_f_delta(hi_ - m)) for m, hi_ in zip(sel["mae"], sel["mae_ci_high"])]
    through = _data_through(sel)

    fig, ax = plt.subplots(figsize=(1200 / 160, 675 / 160), dpi=160)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    y = range(len(sel))
    colours, edges = [], []
    for mid, n in zip(sel["model_id"], sel["n"]):
        if mid == PERSISTENCE_ID:
            colours.append("white")
            edges.append(BASELINE_COLOUR)
        elif n < MIN_N:
            colours.append(LOW_N_COLOUR)
            edges.append("none")
        else:
            colours.append(ACCENT)
            edges.append("none")
    bars = ax.barh(list(y), mae_f, xerr=[lo, hi], color=colours, edgecolor=edges,
                   linewidth=1.2, capsize=2, height=0.68,
                   error_kw={"ecolor": MUTED_COLOUR, "elinewidth": 1.0})
    for bar, mid in zip(bars, sel["model_id"]):
        if mid == PERSISTENCE_ID:  # the baseline is a reference line, not a competitor
            bar.set_linestyle("--")
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9, color=TEXT_COLOUR)
    ax.invert_yaxis()

    right = max((m + h for m, h in zip(mae_f, hi)), default=1.0)
    ax.set_xlim(0, right * 1.22)
    for i, (m, h, n) in enumerate(zip(mae_f, hi, sel["n"])):
        note = f"n={int(n)}" + ("  (n<30)" if n < MIN_N else "")
        ax.text(m + h + right * 0.02, i, note, va="center", fontsize=8,
                color=MUTED_COLOUR if n >= MIN_N else LOW_N_COLOUR)

    ax.set_xlabel("mean absolute error of daily maximum temperature, °F", fontsize=9, color=TEXT_COLOUR)
    ax.set_title(f"Lead day 1 · 00Z · {WINDOW_LABEL.get(window, window)} · 23 U.S. stations",
                 fontsize=12, color=TEXT_COLOUR, loc="left", pad=22)
    ax.text(0, 1.045, "raw model output, no post-processing · bootstrap 95% CI · "
                      f"data through {through}", transform=ax.transAxes, fontsize=8, color=MUTED_COLOUR)
    ax.tick_params(axis="x", labelsize=8, colors=MUTED_COLOUR)
    ax.grid(axis="x", color="#e6e8eb", linewidth=0.8)
    ax.set_axisbelow(True)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.spines["bottom"].set_color("#d0d7de")
    fig.text(0.985, 0.022, "castcheck.zifanzhang.com", ha="right", fontsize=8, color=MUTED_COLOUR)
    fig.tight_layout(rect=(0.0, 0.05, 0.99, 1.0))

    buf = BytesIO()
    fig.savefig(buf, format="png", facecolor="white")
    plt.close(fig)

    parts = ", ".join(f"{lab} {v:.1f} (n={int(n)})" for lab, v, n in zip(labels, mae_f, sel["n"]))
    alt = (f"Horizontal bar chart, {len(sel)} models, of the mean absolute error in °F of the daily "
           f"maximum temperature forecast at lead day 1 from the 00Z cycle over the "
           f"{WINDOW_LABEL.get(window, window)}, averaged over 23 U.S. airport stations, with "
           f"bootstrap 95% confidence intervals; data through {through}. Values: {parts}.")
    return buf.getvalue(), alt


def post_text(sel: pd.DataFrame, window: str) -> str:
    """The post body: the result first, then the caveat, then the link. At most 300 characters."""
    models = sel[sel["model_id"] != PERSISTENCE_ID]  # the baseline is drawn, never announced as best
    usable = models[models["n"] >= MIN_N]
    best = (usable if len(usable) else models if len(models) else sel).iloc[0]
    mae_f = c_to_f_delta(best["mae"])
    text = (f"Best raw Tmax forecast at lead day 1 over the {WINDOW_LABEL.get(window, window)}: "
            f"{display_name(best['model_id'])} ({mae_f:.1f} °F MAE, n={int(best['n'])})\n"
            f"Raw 0.25° model output, no post-processing, scored against NWS climate reports.\n"
            f"{SITE}")
    if len(text) > MAX_POST_CHARS:  # pragma: no cover - guarded by tests on real names
        head, _, tail = text.rpartition("\n")
        text = head[: MAX_POST_CHARS - len(tail) - 2] + "…\n" + tail
    return text


def post_daily(dry_run: bool = False) -> str:
    """Build the chart and either post it or write the preview and report what would be posted."""
    handle = os.environ.get("BSKY_HANDLE") or _read_secret("~/.bsky_handle")
    pw = os.environ.get("BSKY_APP_PASSWORD") or _read_secret("~/.bsky_app_password")

    picked = select_leaderboard(load_scores())
    if picked is None:
        return "skipped: not enough data"
    sel, window = picked
    png, alt = build_chart(sel, window)
    text = post_text(sel, window)

    if dry_run or not (handle and pw):
        PREVIEW_PATH.parent.mkdir(parents=True, exist_ok=True)
        PREVIEW_PATH.write_bytes(png)
        why = "dry-run" if dry_run else "no Bluesky credentials"
        return f"{why}: wrote {PREVIEW_PATH} ({len(png)} bytes)\n--- text ---\n{text}\n--- alt ---\n{alt}"

    from atproto import Client

    c = Client()
    c.login(handle, pw)
    c.send_image(text=text, image=png, image_alt=alt)
    return f"posted as {handle} ({len(text)} chars, window={window})"


def _read_secret(path: str) -> str | None:
    p = Path(path).expanduser()
    try:
        return p.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None
