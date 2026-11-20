"""GTA VI trophy earn-rate forecaster (gta6-trophy-rarity-forecast-01).

Forecasts, for each of the 55 GTA VI PS5 trophies, five quantiles of its
Exophase earn rate at the case's target, from the case's history of reads.

Model (fit at runtime from the case file alone, no network, deterministic):
  * Time is log-time since the game's release, x = ln(1 + hours since
    2026-11-19T00:00Z). Rates move fastest just after launch and slow
    down, so a trend in log-time extrapolates far more sanely than one in
    linear time.
  * Per trophy, a weighted least-squares line of logit(rate) on x is fit
    over the reads since the page first actually changed (the "active"
    reads). The frozen launch-night reads carry no trend, so they are
    left out of the fit; the line is anchored on the last read and its
    slope is shrunk toward the median slope across trophies, then the
    extrapolated move is capped. This is what carries the forecast over
    the final's 4.5-day horizon: persistence (the house baseline) ignores
    the trend, and a 24-hour-only fit (the house curve) throws away the
    earlier evolution.
  * The spread is honest about a 4.5-day horizon: fit residual, a
    random-walk term growing with the horizon, and an allowance for an
    Exophase refresh jump (the practice proves a refresh can move every
    rate at once). Pinball loss rewards quantiles that are really
    believed, so a too-narrow band is worse than a wide one.
  * When the history is entirely frozen (the practice cases), there is
    no trend to fit. The fallback keeps the last rate as the median but
    widens the spread a lot: over a refresh the rates move by factors of
    several, and a narrow band scores terribly (the house baseline's
    practice loss ~1.51 comes from exactly that).

Only the Python standard library is used. Reads $MAKEFUN_INPUT_DIR
(else /input) and writes to $MAKEFUN_OUTPUT_DIR (else /output).
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a
# sibling module would import (this entry is a single file).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0, 0.674, 1.645]
RELEASE = datetime(2026, 11, 19, 0, 0, 0, tzinfo=timezone.utc)

# --- model constants (chosen from the launch-night dynamics, not tuned
# to any answer: the frozen page, the 04:30 refresh to 3,868 players,
# and the rates the public page showed a day later) ---
SHRINK_SLOPE = 0.15        # light pull of each slope toward the
                           # cross-trophy median; trophies are bimodal
                           # (early ones rise, late ones fall), so a
                           # strong pull toward the median would bias
                           # the risers down
MAX_MOVE_LOGIT = 3.0       # cap on |median logit - last logit|; a 4.5-day
                           # horizon can legitimately move a rate a lot
SIGMA_RW = 0.60            # random-walk spread, logit per sqrt(day)
SIGMA_REFRESH = 0.45       # allowance for one Exophase refresh jump
RESID_FLOOR = 0.25         # minimum fit-residual spread
RESID_CAP = 1.50           # maximum fit-residual spread
FLAT_SPREAD = 2.00         # spread when the history is frozen (no trend)
FLAT_MEDIAN_SHIFT = -0.70  # frozen history: rates more often fall than
                           # rise over a refresh (dilution), so the
                           # median logit is nudged down


def clamp(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp(pct) / 100.0
    return math.log(p / (1 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, returning int for whole numbers."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value == int(value) else value


def parse_time(s):
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def hours_since_release(t):
    return max(0.0, (t - RELEASE).total_seconds() / 3600.0)


def log_time(t):
    return math.log1p(hours_since_release(t))


def wls(xs, ys, ws):
    """Weighted least squares y = a + b*x. Returns (a, b, resid_sd)."""
    sw = sum(ws)
    if sw <= 0:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    a = my - b * mx
    resid = [y - (a + b * x) for x, y in zip(xs, ys)]
    var = sum(w * r * r for w, r in zip(ws, resid)) / sw
    return (a, b, math.sqrt(var))


def forecast_case(case):
    if case["levels"] != LEVELS:
        raise ValueError("levels are not the contest's five")
    history = case["history"]
    if not history:
        raise ValueError("no history")
    trophies = case["trophies"]
    n = len(trophies)
    horizon_days = case["horizon_minutes"] / 1440.0
    target = parse_time(case["target"])

    # Per-read log-time and rate matrix (None where a read lacks a trophy).
    xs = []
    rates = []
    for read in history:
        xs.append(log_time(parse_time(read["at"])))
        rates.append(read["rates"])
    x_last = xs[-1]
    x_tgt = log_time(target)

    # First read where the page actually changed vs the previous read.
    # The frozen launch-night reads carry no trend signal.
    first_active = None
    for j in range(1, len(history)):
        prev, cur = rates[j - 1], rates[j]
        changed = False
        for i in range(n):
            a, b = prev[i], cur[i]
            if a is None or b is None:
                if a is not b:
                    changed = True
                    break
            elif a != b:
                changed = True
                break
        if changed:
            first_active = j
            break
    active = list(range(first_active, len(history))) if first_active is not None else []

    # Per-trophy trend fit over the active reads (recency-weighted).
    fits = []
    slopes = []
    for i in range(n):
        pts = []
        for j in active:
            r = rates[j][i]
            if r is not None:
                pts.append((xs[j], logit(r), 1.0 + 0.5 * j))
        if len(pts) >= 3 and len({p[0] for p in pts}) >= 2:
            got = wls([p[0] for p in pts], [p[1] for p in pts], [p[2] for p in pts])
            if got is not None:
                fits.append(got)
                slopes.append(got[1])
                continue
        fits.append(None)
    slopes_sorted = sorted(slopes)
    b_global = slopes_sorted[len(slopes_sorted) // 2] if slopes_sorted else 0.0

    spread_rw = SIGMA_RW * math.sqrt(horizon_days)
    rows = []
    for i, trophy in enumerate(trophies):
        last_rate = None
        for j in range(len(history) - 1, -1, -1):
            if rates[j][i] is not None:
                last_rate = rates[j][i]
                break
        if last_rate is None:
            last_rate = 0.01
        y_last = logit(last_rate)
        fit = fits[i]
        if fit is not None:
            _a, b, resid = fit
            b_shrunk = (1.0 - SHRINK_SLOPE) * b + SHRINK_SLOPE * b_global
            move = b_shrunk * (x_tgt - x_last)
            move = max(-MAX_MOVE_LOGIT, min(MAX_MOVE_LOGIT, move))
            median_logit = y_last + move
            s_fit = min(RESID_CAP, max(RESID_FLOOR, resid))
            spread = math.sqrt(s_fit * s_fit + spread_rw * spread_rw
                               + SIGMA_REFRESH * SIGMA_REFRESH)
        else:
            # Frozen history: no trend. Nudge the median down a little
            # (dilution usually wins over a refresh) and spread wide.
            median_logit = y_last + FLAT_MEDIAN_SHIFT
            spread = math.sqrt(FLAT_SPREAD * FLAT_SPREAD
                               + spread_rw * spread_rw)
        median_logit = max(math.log(0.01 / 99.99), min(math.log(99.99 / 0.01), median_logit))
        qs = []
        for z in Z:
            q = sigmoid(median_logit + z * spread)
            qs.append(round4(clamp(q)))
        # Keep the row non-decreasing after rounding.
        for k in range(1, 5):
            if qs[k] < qs[k - 1]:
                qs[k] = qs[k - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        if len(out.encode("utf-8")) > 65536:
            raise ValueError(f"{path.name}: forecast over 65,536 bytes")
        (OUTPUT_DIR / f"{path.stem}.out").write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
