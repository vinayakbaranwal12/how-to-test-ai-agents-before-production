"""Compare three test gates against one hidden regression, using a MOCK agent.

The mock agent is a Bernoulli process with a fixed success probability. It is
not a language model. It isolates the statistics of each gate from model
behavior, so the only thing that varies between gates is how they decide.

Gates:
  exact_match   : one run, passes only if the answer equals one golden string
  outcome_check : one run, passes if the task outcome is correct
  pass_rate     : n runs per arm, one-sided Fisher exact test against a
                  baseline measured with n runs, fails at alpha

To test a real agent, replace run_batch() with a function that runs your agent
and returns (success, matched_golden) arrays of shape (builds, runs).

Usage:
  python3 gates_sim.py --builds 10000 --seed 20261029 --runs 126 --runs-small-drop 371
"""
import argparse

import numpy as np

from runcount import fisher_cutoffs, two_sample_power


def run_batch(rng, p_success, match_prob, builds, runs):
    """Mock agent. A correct run matches the golden wording with probability match_prob."""
    success = rng.random((builds, runs)) < p_success
    matched = success & (rng.random((builds, runs)) < match_prob)
    return success, matched


def gate_rates(rng, p_healthy, p_reg, args):
    out = {}
    for label, p in (("healthy", p_healthy), ("regressed", p_reg)):
        s, m = run_batch(rng, p, args.match, args.builds, 1)
        out[("exact_match", label)] = float((~m[:, 0]).mean())
        out[("outcome_check", label)] = float((~s[:, 0]).mean())
    return out


def pass_rate_fail_rate(rng, p_healthy, p_candidate, n, args, cutoffs):
    base, _ = run_batch(rng, p_healthy, args.match, args.builds, n)
    cand, _ = run_batch(rng, p_candidate, args.match, args.builds, n)
    a = base.sum(axis=1)
    b = cand.sum(axis=1)
    return float((b <= cutoffs[a + b]).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--healthy", type=float, default=0.95)
    ap.add_argument("--regressed", type=float, default=0.85)
    ap.add_argument("--small-drop", type=float, default=0.90)
    ap.add_argument("--match", type=float, default=0.40, help="P(correct answer equals the golden string)")
    ap.add_argument("--runs", type=int, default=126, help="runs per arm sized for healthy -> regressed")
    ap.add_argument("--runs-small-drop", type=int, default=371, help="runs per arm sized for healthy -> small-drop")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--usd-per-episode", type=float, default=0.0075, help="measured cost floor per episode")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    single = gate_rates(rng, args.healthy, args.regressed, args)
    cut_n = fisher_cutoffs(args.runs, args.alpha)
    cut_big = fisher_cutoffs(args.runs_small_drop, args.alpha)
    pr_false = pass_rate_fail_rate(rng, args.healthy, args.healthy, args.runs, args, cut_n)
    pr_detect = pass_rate_fail_rate(rng, args.healthy, args.regressed, args.runs, args, cut_n)
    pr_small_at_n = pass_rate_fail_rate(rng, args.healthy, args.small_drop, args.runs, args, cut_n)
    pr_small_false = pass_rate_fail_rate(rng, args.healthy, args.healthy, args.runs_small_drop, args, cut_big)
    pr_small_at_big = pass_rate_fail_rate(rng, args.healthy, args.small_drop, args.runs_small_drop, args, cut_big)

    an_detect, an_false = two_sample_power(args.healthy, args.regressed, args.runs, args.alpha)
    an_small_n, _ = two_sample_power(args.healthy, args.small_drop, args.runs, args.alpha)
    an_small_big, an_small_false = two_sample_power(args.healthy, args.small_drop, args.runs_small_drop, args.alpha)

    print(f"builds={args.builds} seed={args.seed} healthy={args.healthy} regressed={args.regressed} match={args.match} alpha={args.alpha}")
    header = f"{'gate':<28}{'drop':<11}{'runs/build':<12}{'usd/build':<11}{'fails_healthy':<15}{'fails_regressed':<17}{'exact_math'}"
    print(header)

    def row(gate, drop, runs, fh, fr, math_fr):
        print(f"{gate:<28}{drop:<11}{runs:<12}{runs * args.usd_per_episode:<11.4f}{fh:<15.4f}{fr:<17.4f}{math_fr}")

    row("exact_match", "0.95-0.85", 1, single[("exact_match", "healthy")], single[("exact_match", "regressed")],
        f"{1 - args.regressed * args.match:.4f}")
    row("outcome_check", "0.95-0.85", 1, single[("outcome_check", "healthy")], single[("outcome_check", "regressed")],
        f"{1 - args.regressed:.4f}")
    row(f"pass_rate n={args.runs}/arm", "0.95-0.85", 2 * args.runs, pr_false, pr_detect, f"{an_detect:.4f}")
    row(f"pass_rate n={args.runs}/arm", "0.95-0.90", 2 * args.runs, pr_false, pr_small_at_n, f"{an_small_n:.4f}")
    row(f"pass_rate n={args.runs_small_drop}/arm", "0.95-0.90", 2 * args.runs_small_drop, pr_small_false, pr_small_at_big,
        f"{an_small_big:.4f}")


if __name__ == "__main__":
    main()
