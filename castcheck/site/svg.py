"""Server-side inline SVG: every chart on the site is generated here, at build time.

Nothing on this site needs JavaScript to be complete (``assets/chart.js`` only adds hover
read-outs).  The figures therefore have to be plain markup, and they have to work in both colour
schemes — so every colour is a CSS custom property defined in ``assets/site.css`` and resolved by
the browser at paint time (inline SVG participates in the document cascade, so ``var(--ink)``
works inside these fragments and flips automatically in dark mode).

Every figure carries an ``<svg role="img" aria-label="…"><title>…</title>`` pair, and callers are
expected to put an equivalent ``<table>`` next to it (usually inside ``<details>``), because a
picture is never the only place a number appears.

All temperature values passed in are already in the display unit (°F); conversion happens in
``build.py``.
"""

from __future__ import annotations

import math
from html import escape

__all__ = [
    "availability_row",
    "bias_class",
    "histogram",
    "line_chart",
    "sparkline",
    "us_map",
]

# Diverging fill for a bias value, warm = too warm (positive), cool = too cold (negative).
# The classes are defined in site.css and are always paired with a text sign (+/−) so that the
# colour is redundant, never load-bearing.
_BIAS_STEPS = ((2.0, 3), (1.0, 2), (0.25, 1))


def bias_class(bias_f: float | None, significant: bool = True) -> str:
    """CSS class for a bias value in °F: ``b-warm-3`` … ``b-cool-3``, or ``b-null``."""
    if bias_f is None or (isinstance(bias_f, float) and math.isnan(bias_f)):
        return "b-null"
    if not significant:
        return "b-flat"
    a = abs(float(bias_f))
    level = 0
    for thr, lv in _BIAS_STEPS:
        if a >= thr:
            level = lv
            break
    if level == 0:
        return "b-flat"
    return f"b-{'warm' if bias_f > 0 else 'cool'}-{level}"


def _fmt(x: float, digits: int = 1) -> str:
    return f"{x:.{digits}f}".rstrip("0").rstrip(".") or "0"


def _nice(v: float) -> float:
    """A round-ish axis maximum at or above ``v``."""
    if v <= 0:
        return 1.0
    exp = math.floor(math.log10(v))
    base = 10.0**exp
    for m in (1, 2, 2.5, 5, 10):
        if v <= m * base:
            return m * base
    return 10 * base


def _open(width: float, height: float, label: str, cls: str = "fig") -> list[str]:
    return [
        f'<svg class="{cls}" viewBox="0 0 {_fmt(width)} {_fmt(height)}" role="img" '
        f'preserveAspectRatio="xMidYMid meet" aria-label="{escape(label)}">',
        f"<title>{escape(label)}</title>",
    ]


def _text(x, y, s, *, anchor="start", cls="lbl", extra="") -> str:
    return (
        f'<text x="{_fmt(x)}" y="{_fmt(y)}" text-anchor="{anchor}" class="{cls}"{extra}>'
        f"{escape(str(s))}</text>"
    )


# ------------------------------------------------------------------------------------------
# daily error time series
# ------------------------------------------------------------------------------------------

def line_chart(
    dates: list[str],
    values: list[float | None],
    *,
    label: str,
    unit: str = "°F",
    width: float = 760.0,
    height: float = 230.0,
    ma_window: int = 7,
) -> str:
    """Daily signed error with a zero line and a centred ``ma_window``-day mean.

    ``values`` may contain ``None`` for days with no score; the line breaks there rather than
    interpolating across the gap (METHODOLOGY: missing is explicit, never smoothed away).
    """
    n = len(values)
    if n == 0:
        return f'<p class="empty">{escape(label)}: no scored days yet.</p>'

    left, right, top, bottom = 44.0, 12.0, 12.0, 30.0
    iw, ih = width - left - right, height - top - bottom
    finite = [v for v in values if v is not None]
    m = _nice(max((abs(v) for v in finite), default=1.0))

    def px(i: int) -> float:
        return left + (iw / 2 if n < 2 else i * iw / (n - 1))

    def py(v: float) -> float:
        return top + ih / 2 - (v / m) * (ih / 2)

    out = _open(width, height, label)
    for t in (m, m / 2, 0.0, -m / 2, -m):
        y = py(t)
        cls = "zero" if t == 0 else "grid"
        out.append(f'<line class="{cls}" x1="{_fmt(left)}" x2="{_fmt(width - right)}" '
                   f'y1="{_fmt(y)}" y2="{_fmt(y)}"/>')
        out.append(_text(left - 6, y + 3.5, ("+" if t > 0 else "") + _fmt(t), anchor="end"))

    # raw series
    seg: list[str] = []
    started = False
    for i, v in enumerate(values):
        if v is None:
            started = False
            continue
        seg.append(("L" if started else "M") + f"{_fmt(px(i), 1)} {_fmt(py(v), 1)}")
        started = True
    if seg:
        out.append(f'<path class="series" d="{" ".join(seg)}"/>')

    # centred moving average, only where the window is full
    if ma_window > 1 and n >= ma_window:
        half = ma_window // 2
        ma_pts: list[str] = []
        started = False
        for i in range(n):
            lo, hi = i - half, i + half + 1
            if lo < 0 or hi > n:
                started = False
                continue
            win = [v for v in values[lo:hi] if v is not None]
            if len(win) < ma_window:
                started = False
                continue
            avg = sum(win) / len(win)
            ma_pts.append(("L" if started else "M") + f"{_fmt(px(i), 1)} {_fmt(py(avg), 1)}")
            started = True
        if ma_pts:
            out.append(f'<path class="series-ma" d="{" ".join(ma_pts)}"/>')

    for i, v in enumerate(values):
        if v is None:
            continue
        out.append(
            f'<circle class="pt" cx="{_fmt(px(i), 1)}" cy="{_fmt(py(v), 1)}" r="1.8">'
            f"<title>{escape(dates[i])}: {'+' if v > 0 else ''}{_fmt(v, 2)} {unit}</title></circle>"
        )

    if dates:
        out.append(_text(left, height - 8, dates[0]))
        if n > 1:
            out.append(_text(width - right, height - 8, dates[-1], anchor="end"))
    out.append(_text(width - right, top + 10, f"{unit}, forecast − observed", anchor="end",
                     cls="lbl dim"))
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------------------------------
# error distribution
# ------------------------------------------------------------------------------------------

def histogram(
    values: list[float],
    *,
    label: str,
    unit: str = "°F",
    bins: int = 20,
    width: float = 760.0,
    height: float = 240.0,
) -> tuple[str, list[dict]]:
    """Signed-error histogram with the P50 and P90 of |error| marked.

    Returns ``(svg, rows)`` where ``rows`` is the same data as a table (bin edges and counts) so
    the caller can publish the numbers next to the picture.
    """
    vals = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not vals:
        return f'<p class="empty">{escape(label)}: no scored days yet.</p>', []

    m = _nice(max(abs(v) for v in vals)) or 1.0
    lo, hi = -m, m
    step = (hi - lo) / bins
    counts = [0] * bins
    for v in vals:
        k = int((min(max(v, lo), hi - 1e-12) - lo) / step)
        counts[min(max(k, 0), bins - 1)] += 1
    top_count = max(counts) or 1

    absol = sorted(abs(v) for v in vals)

    def pct(p: float) -> float:
        if not absol:
            return 0.0
        k = (len(absol) - 1) * p
        f, c = math.floor(k), math.ceil(k)
        return absol[f] + (absol[c] - absol[f]) * (k - f)

    p50, p90 = pct(0.5), pct(0.9)

    left, right, top, bottom = 40.0, 12.0, 28.0, 30.0
    iw, ih = width - left - right, height - top - bottom

    def px(v: float) -> float:
        return left + (v - lo) / (hi - lo) * iw

    out = _open(width, height, label)
    out.append(f'<line class="grid" x1="{_fmt(left)}" x2="{_fmt(width - right)}" '
               f'y1="{_fmt(top + ih)}" y2="{_fmt(top + ih)}"/>')
    bw = iw / bins
    for i, c in enumerate(counts):
        h = ih * c / top_count
        x = left + i * bw
        centre = lo + (i + 0.5) * step
        cls = "bar-warm" if centre > 0 else "bar-cool"
        out.append(
            f'<rect class="{cls}" x="{_fmt(x + 0.4, 2)}" y="{_fmt(top + ih - h, 2)}" '
            f'width="{_fmt(max(bw - 0.8, 0.5), 2)}" height="{_fmt(h, 2)}">'
            f"<title>{_fmt(lo + i * step, 2)} to {_fmt(lo + (i + 1) * step, 2)} {unit}: "
            f"{c} day{'' if c == 1 else 's'}</title></rect>"
        )
    out.append(f'<line class="zero" x1="{_fmt(px(0))}" x2="{_fmt(px(0))}" '
               f'y1="{_fmt(top)}" y2="{_fmt(top + ih)}"/>')
    for v in (p50, p90):
        for sgn in (1, -1):
            x = px(sgn * v)
            out.append(f'<line class="mark" x1="{_fmt(x)}" x2="{_fmt(x)}" y1="{_fmt(top)}" '
                       f'y2="{_fmt(top + ih)}"/>')
    # One caption line instead of two labels that would collide when P50 and P90 are close.
    out.append(_text(width - right, top - 10,
                     f"dashed: P50 |e| {_fmt(p50, 2)} · P90 |e| {_fmt(p90, 2)} {unit}",
                     anchor="end", cls="lbl dim"))
    out.append(_text(left, height - 8, f"{_fmt(lo, 1)}"))
    out.append(_text(px(0), height - 8, "0", anchor="middle"))
    out.append(_text(width - right, height - 8, f"+{_fmt(hi, 1)} {unit}", anchor="end"))
    out.append(_text(left, top - 10, f"n = {len(vals)} days", cls="lbl dim"))
    out.append("</svg>")

    rows = [
        {"from": f"{lo + i * step:+.2f}", "to": f"{lo + (i + 1) * step:+.2f}", "n": c}
        for i, c in enumerate(counts)
    ]
    return "".join(out), rows


# ------------------------------------------------------------------------------------------
# small multiples
# ------------------------------------------------------------------------------------------

def sparkline(
    values: list[float | None],
    *,
    label: str,
    width: float = 96.0,
    height: float = 28.0,
    vmax: float | None = None,
) -> str:
    """A tiny MAE-versus-lead line drawn on a shared vertical scale (``vmax``)."""
    finite = [v for v in values if v is not None]
    if not finite:
        return f'<span class="spark-empty" aria-label="{escape(label)}: no data">—</span>'
    hi = vmax if vmax and vmax > 0 else max(finite)
    hi = hi or 1.0
    n = len(values)
    pad = 3.0

    def px(i: int) -> float:
        return pad + (0 if n < 2 else i * (width - 2 * pad) / (n - 1))

    def py(v: float) -> float:
        return height - pad - (v / hi) * (height - 2 * pad)

    out = _open(width, height, label, cls="spark")
    seg, started = [], False
    for i, v in enumerate(values):
        if v is None:
            started = False
            continue
        seg.append(("L" if started else "M") + f"{_fmt(px(i), 1)} {_fmt(py(v), 1)}")
        started = True
    out.append(f'<path class="series" d="{" ".join(seg)}"/>')
    last = next((i for i in range(n - 1, -1, -1) if values[i] is not None), None)
    if last is not None:
        out.append(f'<circle class="pt" cx="{_fmt(px(last), 1)}" '
                   f'cy="{_fmt(py(values[last]), 1)}" r="1.8"/>')
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------------------------------
# availability
# ------------------------------------------------------------------------------------------

def availability_row(flags: list[str], *, label: str, width: float = 300.0,
                     height: float = 16.0) -> str:
    """A GitHub-Status-style bar: one slot per day, ``yes`` / ``part`` / ``no``."""
    n = len(flags)
    if n == 0:
        return ""
    out = _open(width, height, label, cls="uptime")
    w = width / n
    for i, f in enumerate(flags):
        out.append(
            f'<rect class="u-{f}" x="{_fmt(i * w, 3)}" y="0" '
            f'width="{_fmt(max(w - 0.35, 0.4), 3)}" height="{_fmt(height)}" rx="1"/>'
        )
    out.append("</svg>")
    return "".join(out)


# ------------------------------------------------------------------------------------------
# station map
# ------------------------------------------------------------------------------------------

_LON0, _LAT0, _LAT1, _LAT2 = -96.0, 37.5, 29.5, 45.5


def _albers(lon: float, lat: float) -> tuple[float, float]:
    """Albers equal-area conic for the contiguous United States (metres-free, unit sphere)."""
    p1, p2 = math.radians(_LAT1), math.radians(_LAT2)
    n = 0.5 * (math.sin(p1) + math.sin(p2))
    c = math.cos(p1) ** 2 + 2 * n * math.sin(p1)
    theta = n * (math.radians(lon) - math.radians(_LON0))
    rho = math.sqrt(max(c - 2 * n * math.sin(math.radians(lat)), 0.0)) / n
    rho0 = math.sqrt(max(c - 2 * n * math.sin(math.radians(_LAT0)), 0.0)) / n
    return rho * math.sin(theta), rho0 - rho * math.cos(theta)


def us_map(points: list[dict], *, label: str, width: float = 760.0,
           height: float = 430.0) -> str:
    """Stations on an Albers grid, radius = sample size, fill = bias class.

    There is no coastline: CastCheck ships no third-party boundary file, so the frame is an
    honest latitude/longitude graticule rather than a traced outline.  Every point repeats its
    numbers in the table underneath.
    """
    if not points:
        return f'<p class="empty">{escape(label)}: no stations scored yet.</p>'

    lons = [-125.0, -115.0, -105.0, -95.0, -85.0, -75.0, -65.0]
    lats = [25.0, 30.0, 35.0, 40.0, 45.0, 50.0]
    corners = [_albers(lo, la) for lo in lons for la in lats]
    xs = [c[0] for c in corners]
    ys = [c[1] for c in corners]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    pad = 30.0
    sx = (width - 2 * pad) / max(x1 - x0, 1e-9)
    sy = (height - 2 * pad) / max(y1 - y0, 1e-9)
    s = min(sx, sy)
    ox = pad + ((width - 2 * pad) - s * (x1 - x0)) / 2
    oy = pad + ((height - 2 * pad) - s * (y1 - y0)) / 2

    def proj(lon: float, lat: float) -> tuple[float, float]:
        # Albers y grows northward; SVG y grows downward, so the axis is flipped here.
        x, y = _albers(lon, lat)
        return ox + (x - x0) * s, oy + (y1 - y) * s

    out = _open(width, height, label, cls="fig map")
    for lo in lons:
        d = " ".join(
            ("M" if i == 0 else "L") + "%.1f %.1f" % proj(lo, la)
            for i, la in enumerate([25.0 + 0.5 * k for k in range(51)])
        )
        out.append(f'<path class="grid" d="{d}" fill="none"/>')
    for la in lats:
        d = " ".join(
            ("M" if i == 0 else "L") + "%.1f %.1f" % proj(lo, la)
            for i, lo in enumerate([-125.0 + 1.0 * k for k in range(61)])
        )
        out.append(f'<path class="grid" d="{d}" fill="none"/>')
    for lo in lons:
        x, y = proj(lo, 24.6)
        out.append(_text(x, y, f"{abs(int(lo))}°W", anchor="middle", cls="lbl dim"))
    for la in lats:
        x, y = proj(-126.5, la)
        out.append(_text(x, y + 3, f"{int(la)}°N", anchor="end", cls="lbl dim"))

    nmax = max((p.get("n") or 0) for p in points) or 1
    for p in points:
        if p.get("lat") is None or p.get("lon") is None:
            continue
        x, y = proj(float(p["lon"]), float(p["lat"]))
        r = 4.0 + 7.0 * math.sqrt((p.get("n") or 0) / nmax)
        cls = p.get("bias_class", "b-null")
        sign = p.get("sign", "")
        out.append(
            f'<a href="{escape(p["href"])}"><circle class="dot {cls}" cx="{_fmt(x, 1)}" '
            f'cy="{_fmt(y, 1)}" r="{_fmt(r, 1)}"><title>{escape(p["id"])} {escape(p["name"])}: '
            f'bias {escape(str(p.get("bias", "—")))} °F, n = {p.get("n", 0)}</title></circle>'
            f'{_text(x, y - r - 3, p["id"], anchor="middle", cls="lbl map-lbl")}</a>'
        )
        if sign:
            out.append(_text(x, y + 3, sign, anchor="middle", cls="lbl map-sign"))
    out.append("</svg>")
    return "".join(out)
