"""Daily Bluesky post: one chart of yesterday's lead-day-1 errors (optional; needs BSKY_HANDLE + BSKY_APP_PASSWORD)."""

from __future__ import annotations

import os
from datetime import UTC, datetime
from io import BytesIO

from ..config import DATA_DIR

SITE = "https://castcheck.zifanzhang.com"


def _chart_png() -> tuple[bytes, str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd

    scores = pd.read_parquet(DATA_DIR / "scores" / "latest.parquet")
    sel = scores[(scores.station_id == "ALL") & (scores.window == "90d") & (scores.method == "bilinear")
                 & (scores.init_hour == 0) & (scores.lead_day == 1) & (scores.variable == "tmax")]
    sel = sel.sort_values("mae")
    fig, ax = plt.subplots(figsize=(7, 3.6), dpi=160)
    ax.barh(sel.model_id, sel.mae * 9 / 5, xerr=[(sel.mae - sel.mae_ci_low) * 9 / 5, (sel.mae_ci_high - sel.mae) * 9 / 5],
            color="#2563eb", alpha=0.9, capsize=2)
    ax.invert_yaxis()
    ax.set_xlabel("MAE of daily max temperature, °F (lead day 1, 00Z, last 90 days, 23 U.S. stations)")
    ax.set_title("CastCheck — raw model output vs NWS climate reports", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    buf = BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    best = sel.iloc[0]
    alt = (f"Bar chart of 90-day mean absolute error of daily maximum temperature at lead day 1 for {len(sel)} models; "
           f"best {best.model_id} at {best.mae * 9 / 5:.1f} °F.")
    return buf.getvalue(), alt


def post_daily(dry_run: bool = False) -> str:
    handle, pw = os.environ.get("BSKY_HANDLE"), os.environ.get("BSKY_APP_PASSWORD")
    png, alt = _chart_png()
    text = (f"Daily check {datetime.now(UTC):%Y-%m-%d}: 90-day MAE of raw 2 m Tmax forecasts, lead day 1, "
            f"23 U.S. stations vs NWS climate reports. Raw model output, no post-processing. {SITE}")
    if dry_run or not (handle and pw):
        out = DATA_DIR / "scores" / "bluesky_preview.png"
        out.write_bytes(png)
        return f"dry-run: wrote {out}; text={text!r}"
    from atproto import Client

    c = Client()
    c.login(handle, pw)
    c.send_image(text=text, image=png, image_alt=alt)
    return "posted"
