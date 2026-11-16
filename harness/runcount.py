"""Run-count calculator for pass-rate gates on non-deterministic agents.

Two gates are computed with exact binomial and hypergeometric arithmetic.
No sampling is involved, so every figure is reproducible byte for byte.

  one-sample : the baseline pass rate is known. Fail the build when the
               number of successes in n runs is at or below a cutoff c, where
               c is the largest value that keeps the false-alarm rate at or
               below alpha when nothing has changed.
  two-sample : the baseline is itself measured with n runs. Fail the build
               when a one-sided Fisher exact test says the candidate is worse
               at significance alpha.

Usage:
  python3 runcount.py --baseline 0.95 --drops 0.92 0.90 0.85 --alpha 0.05 --power 0.80
"""
import argparse
import math

import numpy as np

Z = {0.05: 1.6448536269514722, 0.025: 1.959963984540054}
ZB = {0.80: 0.8416212335729143, 0.90: 1.2815515655446004}


def binom_pmf(n, p):
    k = np.arange(n + 1)
    lf = np.array([math.lgamma(i + 1) for i in range(n + 1)])
    logp = lf[n] - lf[k] - lf[n - k] + k * math.log(p) + (n - k) * math.log1p(-p)
    return np.exp(logp)


def one_sample_power(p0, p1, n, alpha):
    pmf0 = binom_pmf(n, p0)
    pmf1 = binom_pmf(n, p1)
    cdf0 = np.cumsum(pmf0)
    ok = np.where(cdf0 <= alpha)[0]
    if len(ok) == 0:
        return -1, 0.0, 0.0
    c = int(ok[-1])
    return c, float(cdf0[c]), float(np.cumsum(pmf1)[c])


def one_sample_n(p0, p1, alpha, power, nmax=1500):
    """Return (first n that reaches power, first n from which power never dips below target up to nmax)."""
    first = None
    powers = []
    for n in range(5, nmax + 1):
        _, _, pw = one_sample_power(p0, p1, n, alpha)
        powers.append(pw)
        if first is None and pw >= power:
            first = n
    stable = None
    for i in range(len(powers) - 1, -1, -1):
        if powers[i] < power:
            stable = 5 + i + 1
            break
    if stable is None:
        stable = 5
    return first, stable


def fisher_cutoffs(n, alpha):
    """For each total success count s over 2n runs, the largest candidate success
    count b with one-sided Fisher p-value <= alpha (-1 when none)."""
    N = 2 * n
    lf = np.concatenate(([0.0], np.cumsum(np.log(np.arange(1, N + 1)))))
    t = np.full(N + 1, -1, dtype=int)
    for s in range(N + 1):
        lo, hi = max(0, s - n), min(n, s)
        b = np.arange(lo, hi + 1)
        logpmf = (lf[s] - lf[b] - lf[s - b]) + (lf[N - s] - lf[n - b] - lf[N - s - n + b]) - (lf[N] - lf[n] - lf[N - n])
        cdf = np.cumsum(np.exp(logpmf))
        ok = np.where(cdf <= alpha + 1e-12)[0]
        if len(ok):
            t[s] = int(b[ok[-1]])
    return t


def two_sample_power(p0, p1, n, alpha):
    t = fisher_cutoffs(n, alpha)
    P0 = binom_pmf(n, p0)
    P1 = binom_pmf(n, p1)
    a = np.arange(n + 1)[:, None]
    b = np.arange(n + 1)[None, :]
    reject = b <= t[a + b]
    return float((P0[:, None] * P1[None, :] * reject).sum()), float((P0[:, None] * P0[None, :] * reject).sum())


def normal_two_sample_n(p0, p1, alpha, power):
    q0, q1 = 1 - p0, 1 - p1
    pbar = (p0 + p1) / 2
    num = Z[alpha] * math.sqrt(2 * pbar * (1 - pbar)) + ZB[power] * math.sqrt(p0 * q0 + p1 * q1)
    return math.ceil((num / (p0 - p1)) ** 2)


def two_sample_n(p0, p1, alpha, power):
    approx = normal_two_sample_n(p0, p1, alpha, power)
    lo, hi = max(5, approx - 20), approx + 400
    first = None
    for n in range(lo, hi + 1):
        pw, _ = two_sample_power(p0, p1, n, alpha)
        if pw >= power:
            first = n
            break
    return approx, first


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", type=float, default=0.95)
    ap.add_argument("--drops", type=float, nargs="+", default=[0.92, 0.90, 0.85])
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--power", type=float, default=0.80)
    ap.add_argument("--judge-error", type=float, default=0.0,
                    help="symmetric chance the judge flips a verdict; the gate then sees p*(1-e)+(1-p)*e")
    args = ap.parse_args()

    if args.judge_error > 0:
        e = args.judge_error

        def observed(p):
            return round(p * (1 - e) + (1 - p) * e, 6)

        print(f"judge_error={e} true baseline={args.baseline} observed baseline={observed(args.baseline)}")
        print("drop_to  observed_drop_to  exact_n_per_arm")
        for p1 in args.drops:
            _, exact = two_sample_n(observed(args.baseline), observed(p1), args.alpha, args.power)
            print(f"{p1:<8} {observed(p1):<17} {exact}")
        return

    print(f"baseline={args.baseline} alpha={args.alpha} power={args.power}")
    print("one-sample gate: baseline known, n runs per build")
    print("drop_to  first_n  stable_n  cutoff  false_alarm  power_at_stable_n")
    for p1 in args.drops:
        first, stable = one_sample_n(args.baseline, p1, args.alpha, args.power)
        c, fa, pw = one_sample_power(args.baseline, p1, stable, args.alpha)
        print(f"{p1:<8} {first:<8} {stable:<9} {c:<7} {fa:<12.4f} {pw:.4f}")
    print("two-sample gate: baseline measured too, n runs per arm")
    print("drop_to  normal_approx_n  exact_n  power_at_exact_n  false_alarm")
    for p1 in args.drops:
        approx, exact = two_sample_n(args.baseline, p1, args.alpha, args.power)
        pw, fa = two_sample_power(args.baseline, p1, exact, args.alpha)
        print(f"{p1:<8} {approx:<16} {exact:<8} {pw:<17.4f} {fa:.4f}")


if __name__ == "__main__":
    main()
