"""gta6-twitch-ccu-forecast-01 entry v2: shape-regression forecaster.

Forecasts Twitch's "Grand Theft Auto VI" category viewers for hours 2..24 of
release day (UTC) from hour 1's print, using the launch-day curves of past big
console launches shipped in the case's own `comparables`.

Model (fitted at runtime, from the case's comparables only -- no hardcoded
coefficients, no network, deterministic):
  1. Shape library: every comparable with a usable release day (hour 1 present,
     >= 20 of hours 1..24 present) and hour-1 print p1 >= 20,000 viewers.
     The p1 floor drops simultaneous-global-launch games (whose hour 1 is
     near-zero because the game had not launched yet at 00:00 UTC) and keeps
     rolling-midnight-local launches like GTA VI.
  2. Unlock-hour alignment: rolling-midnight launches unlock at midnight US
     Eastern, which is 04:00Z (hour 5) in daylight time and 05:00Z (hour 6)
     in standard time, so the library's curves are smeared by +/-1 hour in
     UTC and the unlock jump lands in a wide band. Each comparable's unlock
     hour is read off its own curve (largest log-jump in hours 2..12, if it
     clears 0.30 and agrees within 1 hour with the calendar prior from its
     release date); otherwise the calendar prior is used. Curves are
     re-indexed to hours since unlock, and the target's unlock hour (same
     calendar rule on the case's day_start; GTA VI, 2026-11-19, is standard
     time, so 05:00Z, hour 6) anchors the mapping. The alignment is fitted
     at runtime from the case's own data; no coefficient is hardcoded.
  3. Per hour h = 2..24, ordinary least squares of ln(v_h / v_1) on ln(p1)
     across the library, using each comparable's value at the aligned hour.
     The slope is strongly negative (large launches are already near their
     peak at hour 1; small ones grow several-fold by the US evening), so
     conditioning on launch size is what beats a flat shape.
  4. The median forecast is p1 * exp(a_h + b_h * ln(p1)).
  5. The band comes from the fit residuals: pooled residual quantiles at the
     five scored levels, scaled per hour by that hour's residual spread
     relative to the pooled spread. An honest empirical spread, not a guessed
     sigma.
  6. Fallbacks (deterministic): if fewer than 8 comparables pass the filter,
     relax the floor to p1 >= 1,000; if still too few, fall back to the house
     baseline shape (flat median, spread widening as 0.15 * sqrt(h - 1)).
     Within the fit, an hour with too few aligned points falls back to the
     plain UTC hours for that hour only.

How the grader runs it: `python3 -P -s /bot/main.py` (PYTHONHASHSEED=0) in
python:3.13-slim, standard library only: no network, no pip packages.
Reads every <id>.in in the input directory and writes <id>.out per case:
{"hours": [{"hour": h, "q": [q05, q25, q50, q75, q95]}, ...]}, hours 2..24.
Outside the grader's container, MAKEFUN_INPUT_DIR and MAKEFUN_OUTPUT_DIR
stand in for /input and /output:
    mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
"""

import json
import math
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so `import model`
# finds a model.py shipped beside this file (v1 ships none; kept for parity).
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

# The quantile levels the contest scores, in the order each row lists them.
LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# The standard normal's quantile at each level (baseline fallback only).
Z = [-1.645, -0.674, 0, 0.674, 1.645]
# Baseline fallback spread, in natural-log units per square-root hour.
SIGMA = 0.15

# Shape-library filters.
MIN_P1 = 20000
MIN_P1_RELAXED = 1000
MIN_HOURS = 20
MIN_TRAIN = 8

# Unlock-jump detection: a log-jump must clear this to count as the unlock
# surge, and is only searched in the early hours (the unlock always lands
# in the first half of the release day for this library).
JUMP_THRESHOLD = 0.30
JUMP_WINDOW_END = 12


def round2(x: float) -> float | int:
    """x rounded to 2 decimals, half up, as JavaScript's Math.round(x * 100) / 100.

    A whole number comes back as an int, so json.dumps writes 1240 where
    JavaScript writes 1240 (not 1240.0).
    """
    scaled = x * 100
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100
    return int(value) if value.is_integer() else value


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def release_day_hours(comp: dict) -> dict:
    """{hour: viewers} for the comparable's UTC release day, hours 1..24."""
    ds = parse_ts(comp["day_start"])
    out = {}
    for pt in comp.get("hourly", []):
        if pt.get("viewers") is None:
            continue
        t = parse_ts(pt["from"])
        if ds <= t < ds + timedelta(days=1):
            h = int((t - ds).total_seconds() // 3600) + 1
            if 1 <= h <= 24:
                out[h] = pt["viewers"]
    return out


def build_library(comparables: list, min_p1: float) -> list:
    """Deduped (name, p1, {hour: viewers}, unlock_hour) with a usable day.

    unlock_hour is the comparable's US-Eastern unlock hour (5 or 6, or the
    curve-detected hour when it agrees with the calendar prior within 1).
    """
    seen = {}
    for comp in comparables:
        name = comp.get("name", "")
        day = release_day_hours(comp)
        if 1 not in day or len(day) < MIN_HOURS:
            continue
        if day[1] < min_p1:
            continue
        if name not in seen or len(day) > len(seen[name][1]):
            seen[name] = (day[1], day, comp.get("day_start", ""))
    return [(n, p1, d, comparable_jump(d, ds))
            for n, (p1, d, ds) in sorted(seen.items())]


def _dst_bounds(year: int) -> tuple:
    """(start, end) dates of US daylight time: 2nd Sunday of March to
    1st Sunday of November. Midnight local on the start Sunday is still
    standard time; midnight on the end Sunday is still daylight time."""
    def nth_weekday(month: int, weekday: int, n: int):
        first = datetime(year, month, 1)
        return first + timedelta(days=(weekday - first.weekday()) % 7
                                 + 7 * (n - 1))
    start = nth_weekday(3, 6, 2).date()   # 2nd Sunday of March
    end = nth_weekday(11, 6, 1).date()    # 1st Sunday of November
    return start, end


def calendar_unlock_hour(day_start: str) -> int:
    """US-Eastern midnight unlock as a UTC hour index (1-based) on a date.

    Rolling-midnight launches unlock at midnight local; Eastern midnight is
    04:00Z (hour 5) in daylight time and 05:00Z (hour 6) in standard time.
    """
    ds = parse_ts(day_start).date()
    start, end = _dst_bounds(ds.year)
    return 5 if start < ds <= end else 6


def detect_jump_hour(day: dict) -> int | None:
    """Hour of the largest early log-jump, or None if no jump clears the
    threshold. The unlock surge is the dominant early move; ordinary
    hour-to-hour noise stays well below the threshold."""
    best_h, best_d = None, 0.0
    for h in range(2, JUMP_WINDOW_END + 1):
        if h in day and h - 1 in day and day[h - 1] > 0 and day[h] > 0:
            d = math.log(day[h] / day[h - 1])
            if d > best_d:
                best_d, best_h = d, h
    return best_h if best_d > JUMP_THRESHOLD else None


def comparable_jump(day: dict, day_start: str) -> int:
    """A comparable's unlock hour: its curve-detected jump when the curve
    agrees with the calendar prior within 1 hour, else the calendar prior.
    A detected jump far from the prior is a different phenomenon (a streamer
    wave, an earlier regional unlock), not the US unlock, so the prior wins
    rather than shifting the whole curve by hours."""
    prior = calendar_unlock_hour(day_start) if day_start else 6
    detected = detect_jump_hour(day)
    if detected is not None and abs(detected - prior) <= 1:
        return detected
    return prior


def quantile(sorted_vals: list, tau: float) -> float:
    """Linear-interpolation quantile of an already sorted list."""
    n = len(sorted_vals)
    if n == 1:
        return sorted_vals[0]
    pos = tau * (n - 1)
    lo = int(math.floor(pos))
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def mean(xs: list) -> float:
    return sum(xs) / len(xs)


def pstdev(xs: list) -> float:
    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def fit_shape(train: list, target_jump: int) -> tuple:
    """Per-hour OLS of ln(v_h / v_1) on ln(p1), on unlock-aligned hours;
    pooled residual quantiles.

    Each comparable contributes its value at the hour that is the same
    number of hours past its own unlock as h is past the target's unlock
    (h - target_jump + jump). A comparable with no usable aligned hour is
    skipped for that hour; an hour with too few aligned points falls back
    to the plain UTC hours. Returns (median[h] = (a, b), band[h] = [offset
    at each level]). The forecast log-ratio at hour h is a + b * ln(p1)
    + band[h][tau].
    """
    median = {}
    resid_by_h = {}
    last_good = None
    for h in range(2, 25):
        pts = []
        for _, p1, d, j in train:
            hh = h - target_jump + j
            if hh >= 2 and hh in d:
                pts.append((math.log(p1), math.log(d[hh] / p1)))
        if len(pts) < MIN_TRAIN:
            pts = [(math.log(p1), math.log(d[h] / p1))
                   for _, p1, d, _ in train if h in d]
        if not pts:
            # No data at all for this hour: reuse the previous hour's fit.
            if last_good is not None:
                median[h], resid_by_h[h] = last_good
            else:
                median[h] = (0.0, 0.0)
                resid_by_h[h] = [0.0]
            continue
        mx = mean([p[0] for p in pts])
        my = mean([p[1] for p in pts])
        sxx = sum((x - mx) ** 2 for x, _ in pts)
        sxy = sum((x - mx) * (y - my) for x, y in pts)
        b = sxy / sxx if sxx > 0 else 0.0
        a = my - b * mx
        median[h] = (a, b)
        resid_by_h[h] = sorted(y - (a + b * x) for x, y in pts)
        last_good = (median[h], resid_by_h[h])
    all_resid = sorted(r for rs in resid_by_h.values() for r in rs)
    pooled_q = [quantile(all_resid, tau) for tau in LEVELS]
    sd_all = pstdev(all_resid)
    band = {}
    for h in range(2, 25):
        sd_h = pstdev(resid_by_h[h]) if len(resid_by_h[h]) > 1 else 0.3
        scale = sd_h / sd_all if sd_all > 0 else 1.0
        band[h] = [q * scale for q in pooled_q]
    return median, band


def baseline_quantiles(p1: float, h: int) -> list:
    """House baseline shape: flat median, spread 0.15 * sqrt(h - 1)."""
    return [round2(p1 * math.exp(z * SIGMA * math.sqrt(h - 1))) for z in Z]


def forecast_case(case: dict, name: str) -> dict:
    if case["levels"] != LEVELS:
        raise ValueError(f"{name} asks for levels {case['levels']}, not {LEVELS}")
    first = next((k for k in case["known"] if k["hour"] == 1), None)
    if first is None:
        raise ValueError(f"{name} has no hour-1 print")
    p1 = max(1, first["viewers"])

    train = build_library(case.get("comparables", []), MIN_P1)
    if len(train) < MIN_TRAIN:
        train = build_library(case.get("comparables", []), MIN_P1_RELAXED)

    # The target's unlock hour, from the case's own release date (same
    # calendar rule as the comparables; GTA VI, 2026-11-19, is standard
    # time, so 05:00Z, hour 6). Without a date, assume the GTA VI hour.
    day_start = case.get("day_start", "")
    target_jump = calendar_unlock_hour(day_start) if day_start else 6

    hours = []
    if len(train) >= MIN_TRAIN:
        median, band = fit_shape(train, target_jump)
        lp = math.log(p1)
        for row in case["forecast"]:
            h = row["hour"]
            a, b = median[h]
            m = a + b * lp
            qs = []
            prev = 0.0
            for off in band[h]:
                q = p1 * math.exp(m + off)
                q = min(max(q, 0.01), 1_000_000_000)
                q = round2(q)
                # Keep the row non-decreasing after rounding.
                if q < prev:
                    q = prev
                prev = q
                qs.append(q)
            hours.append({"hour": h, "q": qs})
    else:
        for row in case["forecast"]:
            hours.append({"hour": row["hour"],
                          "q": baseline_quantiles(p1, row["hour"])})
    return {"hours": hours}


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    # Every case file, in name order, so two runs write the same bytes.
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case, path.name)
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
