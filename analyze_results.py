"""
analyze_results.py
==================
Aggregate participant CSV files and compute statistics for the paper rebuttal.

Outputs:
  - Per-condition accuracy (mean ± std across participants)
  - Binomial test: each condition vs. chance level (1/n_classes)
  - Pairwise comparisons: TSM vs. each other condition (paired t-test / Wilcoxon)
  - Reaction time statistics
  - Summary table ready to paste into the rebuttal

Usage:
  python analyze_results.py --results_dir ./collected_results --n_classes 10

collected_results/ should contain one CSV per participant
(the files participants download from the web study).
"""

import os, argparse, glob
import numpy as np
import pandas as pd
from scipy import stats

def load_all(results_dir):
    dfs = []
    for f in sorted(glob.glob(os.path.join(results_dir, '*.csv'))):
        try:
            df = pd.read_csv(f)
            pid = df['participant_id'].iloc[0] if 'participant_id' in df.columns else os.path.basename(f)
            df['_file'] = pid
            dfs.append(df)
        except Exception as e:
            print(f"  skip {f}: {e}")
    if not dfs:
        raise FileNotFoundError(f"No CSV files found in {results_dir}")
    return pd.concat(dfs, ignore_index=True)


def per_participant_acc(df):
    """Returns DataFrame: participant × condition → accuracy."""
    return (df.groupby(['_file', 'condition'])['correct']
              .mean()
              .unstack('condition'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--results_dir', default='./collected_results')
    ap.add_argument('--n_classes',   type=int, default=10)
    ap.add_argument('--alpha',       type=float, default=0.05)
    args = ap.parse_args()

    print(f"Loading results from: {args.results_dir}")
    df = load_all(args.results_dir)
    n_participants = df['_file'].nunique()
    conditions = sorted(df['condition'].unique())
    chance = 1.0 / args.n_classes

    print(f"  Participants : {n_participants}")
    print(f"  Conditions   : {conditions}")
    print(f"  Total trials : {len(df)}")
    print(f"  Chance level : {chance*100:.1f}%\n")

    ppa = per_participant_acc(df)

    sep = "=" * 68
    lines = [sep,
             f"User Study Results  (ImageNet-1K, {args.n_classes} classes, N={n_participants})",
             f"Chance level = {chance*100:.1f}%",
             sep]

    # ── per-condition summary ─────────────────────────────────────────────────
    lines.append(f"\n{'Condition':<14}  {'Acc (%)':>8}  {'±SD':>6}  {'n_trials':>8}  "
                 f"{'vs chance':>12}  {'p-value':>9}")
    lines.append("-" * 68)

    cond_accs = {}
    for cond in conditions:
        sub = df[df['condition'] == cond]
        n_trials = len(sub)
        n_correct = sub['correct'].sum()
        acc_mean = sub['correct'].mean()
        # Per-participant accuracies for SD
        pp = ppa[cond].dropna() if cond in ppa.columns else pd.Series([acc_mean])
        acc_sd = pp.std(ddof=1) if len(pp) > 1 else 0.0
        cond_accs[cond] = pp.values

        # Binomial test: is accuracy different from chance?
        binom = stats.binomtest(int(n_correct), n_trials, chance)
        p_val = binom.pvalue
        sig = '***' if p_val < 0.001 else ('**' if p_val < 0.01 else ('*' if p_val < 0.05 else 'ns'))
        direction = 'LOWER' if acc_mean < chance else 'higher'

        lines.append(f"  {cond:<12}  {acc_mean*100:>8.2f}  {acc_sd*100:>6.2f}  "
                     f"{n_trials:>8}  {direction:>12}  {p_val:>9.4f} {sig}")

    # ── pairwise comparisons with TSM ─────────────────────────────────────────
    if 'TSM' in cond_accs:
        lines.append(f"\nPairwise (TSM vs. other conditions)  — paired Wilcoxon signed-rank test:")
        lines.append("-" * 68)
        tsm_vals = cond_accs['TSM']
        for cond in conditions:
            if cond == 'TSM': continue
            other_vals = cond_accs[cond]
            # Align by participant (inner join)
            shared = ppa[['TSM', cond]].dropna() if cond in ppa.columns else None
            if shared is not None and len(shared) >= 5:
                stat, p = stats.wilcoxon(shared['TSM'], shared[cond])
                effect_d = (shared['TSM'].mean() - shared[cond].mean()) / shared[['TSM', cond]].stack().std()
                sig = '***' if p < 0.001 else ('**' if p < 0.01 else ('*' if p < 0.05 else 'ns'))
                lines.append(f"  TSM vs {cond:<10}:  W={stat:.1f}  p={p:.4f} {sig}  "
                             f"Cohen's d={effect_d:.3f}  "
                             f"(TSM Δ={( shared['TSM'].mean()-shared[cond].mean())*100:+.2f}%)")
            else:
                lines.append(f"  TSM vs {cond:<10}:  (insufficient paired data, n={len(shared) if shared is not None else 0})")

    # ── reaction time ─────────────────────────────────────────────────────────
    if 'rt_ms' in df.columns:
        lines.append(f"\nReaction time (ms)  [median (IQR)]:")
        lines.append("-" * 68)
        for cond in conditions:
            rts = df[df['condition'] == cond]['rt_ms'].dropna()
            q1, med, q3 = np.percentile(rts, [25, 50, 75])
            lines.append(f"  {cond:<14}:  {med:.0f}  ({q1:.0f}–{q3:.0f})")

    lines.append(f"\n{sep}")
    lines.append("Paste into rebuttal:")
    lines.append(f"  Chance = {chance*100:.1f}%   N = {n_participants} participants")
    for cond in conditions:
        sub = df[df['condition'] == cond]
        acc = sub['correct'].mean() * 100
        pp  = ppa[cond].dropna() if cond in ppa.columns else pd.Series([acc/100])
        sd  = pp.std(ddof=1) * 100 if len(pp) > 1 else 0.0
        lines.append(f"  {cond}: {acc:.2f}% ± {sd:.2f}%")

    output = "\n".join(lines)
    print(output)

    out_path = os.path.join(args.results_dir, 'analysis_summary.txt')
    os.makedirs(args.results_dir, exist_ok=True)
    with open(out_path, 'w') as f:
        f.write(output + "\n")
    print(f"\n→ Saved: {out_path}")


if __name__ == '__main__':
    main()
