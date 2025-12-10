"""
Compare Results Across Multiple Runs

This script analyzes output.txt to compare TEMPORAL vs Baseline across different seeds.

Usage:
  python test_comparison.py
"""

import re
import numpy as np
from collections import defaultdict


def parse_output_file(filename="output.txt"):
    """Parse output.txt to extract results"""
    try:
        with open(filename, 'r') as f:
            content = f.read()
    except FileNotFoundError:
        print(f"❌ {filename} not found!")
        print("Run training first with: python run_colab.py")
        return None

    # Parse runs
    runs = []
    run_blocks = re.split(r"NEW RUN:", content)[1:]  # Skip text before first run

    for block in run_blocks:
        run_info = {'temporal': {}, 'baseline': {}}

        # Extract seed
        seed_match = re.search(r"SEED: (\d+)", block)
        if seed_match:
            run_info['seed'] = int(seed_match.group(1))

        # Extract TEMPORAL results
        temporal_match = re.search(
            r"TEMPORAL MODEL TRAINING RESULTS.*?Final Results:\s+Train Loss: ([\d.]+)\s+Eval Loss: ([\d.]+)\s+Perplexity: ([\d.]+)",
            block, re.DOTALL
        )
        if temporal_match:
            run_info['temporal'] = {
                'train_loss': float(temporal_match.group(1)),
                'eval_loss': float(temporal_match.group(2)),
                'perplexity': float(temporal_match.group(3))
            }

        # Extract BASELINE results
        baseline_match = re.search(
            r"BASELINE MODEL TRAINING RESULTS.*?Final Results:\s+Train Loss: ([\d.]+)\s+Eval Loss: ([\d.]+)\s+Perplexity: ([\d.]+)",
            block, re.DOTALL
        )
        if baseline_match:
            run_info['baseline'] = {
                'train_loss': float(baseline_match.group(1)),
                'eval_loss': float(baseline_match.group(2)),
                'perplexity': float(baseline_match.group(3))
            }

        # Only add runs that have both models
        if run_info['temporal'] and run_info['baseline']:
            runs.append(run_info)

    return runs


def analyze_runs(runs):
    """Analyze and compare runs"""
    if not runs:
        print("No complete runs found in output.txt")
        return

    print("\n" + "="*80)
    print("MULTI-RUN COMPARISON ANALYSIS")
    print("="*80)
    print(f"\nTotal completed runs: {len(runs)}\n")

    # Collect metrics
    temporal_ppls = []
    baseline_ppls = []
    improvements = []

    print("=" * 80)
    print("INDIVIDUAL RUN RESULTS")
    print("=" * 80 + "\n")

    for i, run in enumerate(runs, 1):
        seed = run.get('seed', 'Unknown')
        temporal_ppl = run['temporal']['perplexity']
        baseline_ppl = run['baseline']['perplexity']
        improvement = ((baseline_ppl - temporal_ppl) / baseline_ppl) * 100

        temporal_ppls.append(temporal_ppl)
        baseline_ppls.append(baseline_ppl)
        improvements.append(improvement)

        print(f"Run {i} (Seed: {seed}):")
        print(f"  TEMPORAL:  Perplexity = {temporal_ppl:.4f}")
        print(f"  BASELINE:  Perplexity = {baseline_ppl:.4f}")
        print(f"  Improvement: {improvement:+.2f}%")
        print(f"  Winner: {'✅ TEMPORAL' if temporal_ppl < baseline_ppl else '❌ BASELINE'}\n")

    # Statistical analysis
    print("=" * 80)
    print("AGGREGATE STATISTICS")
    print("=" * 80 + "\n")

    print("TEMPORAL Model:")
    print(f"  Mean Perplexity: {np.mean(temporal_ppls):.4f} ± {np.std(temporal_ppls):.4f}")
    print(f"  Min/Max: {np.min(temporal_ppls):.4f} / {np.max(temporal_ppls):.4f}")

    print("\nBASELINE Model:")
    print(f"  Mean Perplexity: {np.mean(baseline_ppls):.4f} ± {np.std(baseline_ppls):.4f}")
    print(f"  Min/Max: {np.min(baseline_ppls):.4f} / {np.max(baseline_ppls):.4f}")

    print("\nImprovement:")
    print(f"  Mean: {np.mean(improvements):+.2f}% ± {np.std(improvements):.2f}%")
    print(f"  Min/Max: {np.min(improvements):+.2f}% / {np.max(improvements):+.2f}%")

    wins = sum(1 for imp in improvements if imp > 0)
    print(f"\nTEMPORAL wins: {wins}/{len(runs)} runs ({wins/len(runs)*100:.1f}%)")

    # Statistical significance (t-test)
    if len(temporal_ppls) >= 3:
        from scipy import stats
        t_stat, p_value = stats.ttest_rel(temporal_ppls, baseline_ppls)
        print(f"\nPaired t-test:")
        print(f"  t-statistic: {t_stat:.4f}")
        print(f"  p-value: {p_value:.4f}")
        if p_value < 0.05:
            print(f"  ✅ Statistically significant (p < 0.05)!")
        else:
            print(f"  ⚠️  Not statistically significant (need more runs)")

    # Verdict
    print("\n" + "=" * 80)
    print("CONCLUSION")
    print("=" * 80 + "\n")

    avg_improvement = np.mean(improvements)
    if avg_improvement > 0.5 and wins >= len(runs) * 0.7:
        print("✅ STRONG EVIDENCE: TEMPORAL consistently outperforms Baseline")
        print(f"   Average improvement: {avg_improvement:.2f}%")
        print(f"   Wins {wins}/{len(runs)} runs")
    elif avg_improvement > 0:
        print("✅ POSITIVE RESULT: TEMPORAL shows improvement over Baseline")
        print(f"   Average improvement: {avg_improvement:.2f}%")
        print("   Consider running more seeds for statistical significance")
    else:
        print("⚠️  NO IMPROVEMENT: TEMPORAL not beating Baseline")
        print("   Possible issues: model needs more training, hyperparameters, etc.")

    print()

    # Save analysis
    with open("analysis_summary.txt", 'w') as f:
        f.write("TEMPORAL vs BASELINE: Multi-Run Analysis\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total runs: {len(runs)}\n\n")

        f.write("TEMPORAL:\n")
        f.write(f"  Mean Perplexity: {np.mean(temporal_ppls):.4f} ± {np.std(temporal_ppls):.4f}\n\n")

        f.write("BASELINE:\n")
        f.write(f"  Mean Perplexity: {np.mean(baseline_ppls):.4f} ± {np.std(baseline_ppls):.4f}\n\n")

        f.write(f"Average Improvement: {np.mean(improvements):+.2f}%\n")
        f.write(f"TEMPORAL Wins: {wins}/{len(runs)} ({wins/len(runs)*100:.1f}%)\n\n")

        if avg_improvement > 0:
            f.write("✅ TEMPORAL outperforms Baseline\n")
        else:
            f.write("⚠️  No clear advantage for TEMPORAL\n")

    print("Analysis saved to: analysis_summary.txt\n")


def main():
    print("\n" + "="*80)
    print("TEMPORAL MULTI-RUN COMPARISON")
    print("="*80)
    print("\nThis script compares TEMPORAL vs Baseline across multiple seed runs.\n")

    runs = parse_output_file()

    if runs is None or len(runs) == 0:
        print("\n❌ No runs found in output.txt")
        print("\nTo generate runs:")
        print("  python run_colab.py --seed 42")
        print("  python run_colab.py --seed 123")
        print("  python run_colab.py --seed 777")
        print("\nThen run this script again.\n")
        return

    analyze_runs(runs)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
