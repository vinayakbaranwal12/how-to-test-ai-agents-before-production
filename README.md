# How to Test AI Agents Before Production: evidence

This folder holds the scripts and results behind the DigitalOcean Community article "How to Test AI Agents Before Production". The agent in `gates_sim.py` is a mock with fixed success probabilities. It is not a language model, so the results show how each test gate behaves statistically, not how a model behaves.

DigitalOcean Community article: link added on publication.

## Headline Results

At a 95% baseline, the two sample gate needs 889 runs per side for a drop to 92%, 371 runs per side for a drop to 90%, and 126 runs per side for a drop to 85%.

In 10,000 simulated builds of a mock agent whose success rate falls from 95% to 85%, a single run outcome check failed 15.5% of regressed builds. The 126 run gate failed 80.2% of them.

## Requirements

- Python 3.9.6
- numpy 2.0.2
- matplotlib, only for `make_charts.py`

## Reproduce the article's numbers

Run every command from this folder. Each command prints the output block shown in the article, and the same text is saved in `results/`.

```bash
python3 harness/runcount.py --baseline 0.95 --drops 0.92 0.90 0.85 --alpha 0.05 --power 0.80
python3 harness/runcount.py --baseline 0.95 --drops 0.92 0.90 0.85 --alpha 0.05 --power 0.90
python3 harness/runcount.py --baseline 0.95 --drops 0.90 0.85 --alpha 0.05 --power 0.80 --judge-error 0.05
python3 harness/runcount.py --baseline 0.95 --drops 0.90 0.85 --alpha 0.05 --power 0.80 --judge-error 0.10
python3 harness/cost.py --runs-per-side 126 371 889 --builds-per-day 10
python3 harness/gates_sim.py
python3 harness/rules_check.py data/trace_pass.json
python3 harness/rules_check.py data/trace_fail.json
```

`rules_check.py` exits with status 1 when it finds violations, which is expected for `data/trace_fail.json`. Both trace files are hand-written samples.

## Files

- `harness/runcount.py`: exact run counts for a one-sample and a two-sample pass-rate gate, with an optional judge-error model.
- `harness/gates_sim.py`: compares an exact-match gate, a single-run outcome check, and a pass-rate gate on a mock agent. It uses 10,000 builds and seed 20261029.
- `harness/cost.py`: prices the run counts at the measured cost per episode.
- `harness/rules_check.py`: checks rules that must hold on every run, such as an allow list and step and token budgets.
- `harness/make_charts.py`: draws the two charts in `results/`.
- `data/`: two sample traces for the rule checker.
- `results/`: saved output of each command and the two chart images.

## Limits

- The agent is a mock, so nothing here measures model behavior or model drift.
- The cost per episode is a floor taken from a separate 1,300-episode test, and it excludes judge and simulator calls.
