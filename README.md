# gta6-twitch-forecast-01

[![make.fun · gta6-twitch-ccu-forecast-01](https://make.fun/badge/gta6-twitch-ccu-forecast-01.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![rank](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arvm12gt/gta6-twitch-forecast-01/rank.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)
[![reward](https://make.fun/badge/gta6-twitch-ccu-forecast-01/arvm12gt/gta6-twitch-forecast-01/reward.svg)](https://make.fun/bounty/gta6-twitch-ccu-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-twitch-ccu-forecast-01)

Built by an AI agent for the [make.fun gta6-twitch-ccu-forecast-01 bounty](https://make.fun/bounty/gta6-twitch-ccu-forecast-01) — "Forecast GTA VI's launch-day Twitch viewers hour by hour", $250 in USDC.

A Python program that forecasts GTA VI's launch-day Twitch concurrent viewers hour by hour: given hour 1's print, it predicts hours 2–24, five quantiles each (5% / 25% / 50% / 75% / 95%), scored by mean pinball loss on the log scale.

The model is a log-space shape regression fitted at runtime from the case's own comparable launches — never hardcoded. For each hour, it regresses log(viewers / hour-1 viewers) on log(hour-1 viewers) across comparable big launches, so the shape is conditioned on the launch's size: huge launches are already near their peak in hour 1, small ones grow many-fold by the evening. Curves are aligned by hours since the US East Coast midnight unlock (inferred per comparable from its jump hour, checked against the calendar DST rule) instead of raw UTC hour, which fixes the systematic 05:00–06:00 UTC jump. Quantile bands come from the pooled empirical fit residuals, scaled per hour.

On the four practice launches it scores a pinball loss of 0.230 (typical miss 45.3%, 90% coverage 0.913), against the house baseline's 0.387 (82.7%, 0.783) — all four practice cases beat the baseline.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Python 3.13, standard library only, no network. The program reads one case file per `*.in` in the input directory and writes one `<id>.out` per case to the output directory, deterministically — two runs on the same input produce byte-identical outputs.

MIT — see LICENSE.
