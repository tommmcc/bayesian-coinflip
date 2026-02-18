#!/usr/bin/env python3
"""
Bayesian Coin Flip (Beta–Binomial) — zero dependencies.

Modelling an unknown coin bias p = P(Heads).

Prior:     p ~ Beta(alpha, beta)
Data:      h heads, t tails
Posterior: p | data ~ Beta(alpha + h, beta + t)

Print:
- the real H/T sequence (tied to the simulated flips)
- posterior mean and MAP
- a 95% credible interval (via Monte Carlo sampling)
- P(p > 0.5) and P(|p - 0.5| < eps)

No NumPy. No external libraries.
"""

import argparse
import random
import time
from typing import List, Tuple


def simulate_flips(n: int, true_p: float, rng: random.Random) -> Tuple[List[bool], int, int]:
    flips = []
    heads = 0
    for _ in range(n):
        is_head = rng.random() < true_p
        flips.append(is_head)
        heads += 1 if is_head else 0
    tails = n - heads
    return flips, heads, tails


def show_flips(flips: List[bool], delay: float = 0.0, wrap: int = 80) -> None:
    """Print the real flip outcomes as H/T."""
    print("Flips (real sequence):")

    line_len = 0
    for f in flips:
        ch = "H" if f else "T"
        print(ch, end="", flush=True)
        line_len += 1

        if line_len >= wrap:
            print()
            line_len = 0

        if delay > 0:
            time.sleep(delay)

    if line_len != 0:
        print()
    print()


def beta_mean(a: float, b: float) -> float:
    return a / (a + b)


def beta_map(a: float, b: float) -> float:
    # Mode exists only if a>1 and b>1; otherwise fallback to mean.
    if a > 1 and b > 1:
        return (a - 1) / (a + b - 2)
    return beta_mean(a, b)


def quantile(sorted_vals: List[float], q: float) -> float:
    """
    Compute quantile with linear interpolation.
    sorted_vals must be sorted ascending.
    q in [0,1].
    """
    if not sorted_vals:
        raise ValueError("quantile() received empty list.")
    if q <= 0:
        return sorted_vals[0]
    if q >= 1:
        return sorted_vals[-1]

    n = len(sorted_vals)
    pos = (n - 1) * q
    lo = int(pos)
    hi = min(lo + 1, n - 1)
    frac = pos - lo
    return sorted_vals[lo] * (1 - frac) + sorted_vals[hi] * frac


def summarize_posterior(a_post: float, b_post: float, eps: float, samples: int, rng: random.Random):
    """
    Monte Carlo sampling from Beta(a_post, b_post) using random.betavariate.
    Then estimate:
      - 95% credible interval
      - P(p > 0.5)
      - P(|p - 0.5| < eps)
    """
    draws = [rng.betavariate(a_post, b_post) for _ in range(samples)]
    draws.sort()

    ci_low = quantile(draws, 0.025)
    ci_high = quantile(draws, 0.975)

    # For probabilities, we can reuse the sorted draws (but easiest: count)
    gt_half = 0
    near_fair = 0
    for x in draws:
        if x > 0.5:
            gt_half += 1
        if abs(x - 0.5) < eps:
            near_fair += 1

    return {
        "mean": beta_mean(a_post, b_post),
        "map": beta_map(a_post, b_post),
        "ci_low": ci_low,
        "ci_high": ci_high,
        "p_gt_half": gt_half / samples,
        "p_near_fair": near_fair / samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Bayesian coin flip (Beta–Binomial) — zero dependencies.")
    parser.add_argument("--n", type=int, default=50, help="Number of flips to simulate")
    parser.add_argument("--true-p", type=float, default=0.6, help="True P(Heads) (simulation only)")
    parser.add_argument("--alpha", type=float, default=1.0, help="Prior alpha")
    parser.add_argument("--beta", type=float, default=1.0, help="Prior beta")
    parser.add_argument("--eps", type=float, default=0.05, help="Fairness band for P(|p-0.5|<eps)")
    parser.add_argument("--samples", type=int, default=20000, help="Monte Carlo samples (posterior)")
    parser.add_argument("--seed", type=int, default=None, help="Set for reproducible runs (omit for randomness)")
    parser.add_argument("--delay", type=float, default=0.0, help="Delay between printing flips (seconds)")
    args = parser.parse_args()

    # Validation
    if args.n <= 0:
        raise ValueError("--n must be positive.")
    if not (0.0 < args.true_p < 1.0):
        raise ValueError("--true-p must be between 0 and 1 (exclusive).")
    if args.alpha <= 0 or args.beta <= 0:
        raise ValueError("--alpha and --beta must be > 0.")
    if args.samples < 5000:
        raise ValueError("--samples should be >= 5000 for stable-ish results.")
    if args.delay < 0:
        raise ValueError("--delay must be >= 0.")

    rng = random.Random(args.seed)

    flips, heads, tails = simulate_flips(args.n, args.true_p, rng)
    show_flips(flips, delay=args.delay)

    # Bayesian update (conjugate posterior)
    a_post = args.alpha + heads
    b_post = args.beta + tails

    summary = summarize_posterior(a_post, b_post, args.eps, args.samples, rng)

    print("=== Coin Bias Estimate ===")
    print(f"Flips:  {args.n}")
    print(f"Heads:  {heads}")
    print(f"Tails:  {tails}\n")

    print("Plain English:")
    print(f"- Best estimate of P(Heads): {summary['mean']:.3f}  (~{summary['mean']*100:.1f}%)")
    print(f"- 95% uncertainty range:     {summary['ci_low']:.3f} to {summary['ci_high']:.3f}")
    print(f"- Chance coin favors heads:  {summary['p_gt_half']:.3f}   (P(Heads) > 0.5)")
    print(f"- Chance roughly fair:       {summary['p_near_fair']:.3f}   (within ±{args.eps:g} of 0.5)\n")

    print("Technical:")
    print(f"- Prior Beta({args.alpha:g}, {args.beta:g}) → Posterior Beta({a_post:g}, {b_post:g})")
    print(f"- Posterior MAP: {summary['map']:.3f}")
    print(f"- Seed: {args.seed if args.seed is not None else '(none)'}")


if __name__ == "__main__":
    main()
