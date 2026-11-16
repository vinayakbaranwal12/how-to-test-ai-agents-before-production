"""Cost of a pass-rate gate: episodes per build times price per episode.

The two-sample gate pays for both sides, so episodes per build is 2 x runs per side.
The default price is the measured per-episode cost from the 1,300-episode live test:
(2,784,918 input tokens x $2.50 + 187,006 output tokens x $15.00) / 1,000,000 / 1,300 episodes.
That test used short tasks against a mock API, so treat the price as a floor.

Usage:
  python3 cost.py --runs-per-side 126 371 889 --builds-per-day 10
"""
import argparse
from decimal import ROUND_HALF_UP, Decimal

DEFAULT_PRICE = round((2784918 * 2.50 + 187006 * 15.00) / 1_000_000 / 1300, 4)

ap = argparse.ArgumentParser()
ap.add_argument("--runs-per-side", type=int, nargs="+", default=[126, 371, 889])
ap.add_argument("--usd-per-episode", type=float, default=DEFAULT_PRICE)
ap.add_argument("--builds-per-day", type=int, default=10)
args = ap.parse_args()

print(f"usd_per_episode={args.usd_per_episode} builds_per_day={args.builds_per_day}")
print("runs_per_side  episodes_per_build  usd_per_build  usd_per_day")
for n in args.runs_per_side:
    episodes = 2 * n
    exact = Decimal(episodes) * Decimal(str(args.usd_per_episode))
    usd = exact.quantize(Decimal("0.01"), ROUND_HALF_UP)
    per_day = (exact * args.builds_per_day).quantize(Decimal("0.01"), ROUND_HALF_UP)
    print(f"{n:<14} {episodes:<19} {usd:<14} {per_day}")
