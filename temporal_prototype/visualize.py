"""
Visualization script for TEMPORAL analysis.
Plots time embedding evolution, frequency correlations, and performance comparisons.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
import argparse
from scipy.stats import pearsonr

from config import Config


def plot_time_evolution(time_stats_history, output_dir):
    """
    Plot how time embeddings evolve during training.
    Shows mean, max time magnitudes over epochs.
    """
    if not time_stats_history:
        print("No time statistics history available")
        return

    epochs = [entry['epoch'] for entry in time_stats_history]
    mean_magnitudes = [entry['stats']['mean_time_magnitude'] for entry in time_stats_history]
    max_magnitudes = [entry['stats']['max_time_magnitude'] for entry in time_stats_history]

    plt.figure(figsize=(10, 6))
    plt.plot(epochs, mean_magnitudes, label='Mean Time Magnitude', marker='o')
    plt.plot(epochs, max_magnitudes, label='Max Time Magnitude', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('Time Embedding Magnitude')
    plt.title('Time Embedding Evolution During Training')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'time_evolution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_frequency_vs_time(time_stats, output_dir):
    """
    Plot correlation between token frequency and time values.
    Shows whether frequently used tokens develop higher time values.
    """
    usage_counts = time_stats['usage_counts']
    time_magnitudes = time_stats['time_magnitudes']

    # Filter out zero counts
    mask = usage_counts > 0
    usage_counts = usage_counts[mask]
    time_magnitudes = time_magnitudes[mask]

    # Compute correlation
    correlation, p_value = pearsonr(usage_counts, time_magnitudes)

    plt.figure(figsize=(10, 6))
    plt.scatter(usage_counts, time_magnitudes, alpha=0.5, s=20)
    plt.xlabel('Token Usage Count')
    plt.ylabel('Time Embedding Magnitude')
    plt.title(f'Token Frequency vs Time Value\nCorrelation: {correlation:.4f} (p={p_value:.4e})')
    plt.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(usage_counts, time_magnitudes, 1)
    p = np.poly1d(z)
    plt.plot(usage_counts, p(usage_counts), "r--", alpha=0.8, label='Trend line')
    plt.legend()

    output_path = os.path.join(output_dir, 'frequency_vs_time.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_time_distribution(time_stats, output_dir):
    """
    Plot distribution of time embedding magnitudes across vocabulary.
    Shows histogram and cumulative distribution.
    """
    time_magnitudes = time_stats['time_magnitudes']

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Histogram
    ax1.hist(time_magnitudes, bins=50, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Time Embedding Magnitude')
    ax1.set_ylabel('Count')
    ax1.set_title('Distribution of Time Values')
    ax1.grid(True, alpha=0.3)

    # Cumulative distribution
    sorted_times = np.sort(time_magnitudes)
    cumulative = np.arange(1, len(sorted_times) + 1) / len(sorted_times)
    ax2.plot(sorted_times, cumulative)
    ax2.set_xlabel('Time Embedding Magnitude')
    ax2.set_ylabel('Cumulative Proportion')
    ax2.set_title('Cumulative Distribution of Time Values')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'time_distribution.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_perplexity_comparison(temporal_logs, baseline_logs, output_dir):
    """
    Compare perplexity between TEMPORAL and Baseline models during training.
    """
    # Extract validation perplexities
    temporal_ppls = temporal_logs.get('val_perplexities', [])
    baseline_ppls = baseline_logs.get('val_perplexities', [])

    if not temporal_ppls or not baseline_ppls:
        print("Insufficient data for perplexity comparison")
        return

    temporal_steps = [entry['step'] for entry in temporal_ppls]
    temporal_values = [entry['perplexity'] for entry in temporal_ppls]

    baseline_steps = [entry['step'] for entry in baseline_ppls]
    baseline_values = [entry['perplexity'] for entry in baseline_ppls]

    plt.figure(figsize=(10, 6))
    plt.plot(temporal_steps, temporal_values, label='TEMPORAL', marker='o', markersize=4)
    plt.plot(baseline_steps, baseline_values, label='Baseline', marker='s', markersize=4)
    plt.xlabel('Training Step')
    plt.ylabel('Validation Perplexity')
    plt.title('Perplexity Comparison: TEMPORAL vs Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'perplexity_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_training_loss_comparison(temporal_logs, baseline_logs, output_dir):
    """
    Compare training loss curves.
    """
    temporal_losses = temporal_logs.get('train_losses', [])
    baseline_losses = baseline_logs.get('train_losses', [])

    if not temporal_losses or not baseline_losses:
        print("Insufficient data for loss comparison")
        return

    temporal_steps = [entry['step'] for entry in temporal_losses]
    temporal_values = [entry['loss'] for entry in temporal_losses]

    baseline_steps = [entry['step'] for entry in baseline_losses]
    baseline_values = [entry['loss'] for entry in baseline_losses]

    plt.figure(figsize=(10, 6))
    plt.plot(temporal_steps, temporal_values, label='TEMPORAL', alpha=0.7)
    plt.plot(baseline_steps, baseline_values, label='Baseline', alpha=0.7)
    plt.xlabel('Training Step')
    plt.ylabel('Training Loss')
    plt.title('Training Loss: TEMPORAL vs Baseline')
    plt.legend()
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'loss_comparison.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_token_category_analysis(evaluation_results, output_dir):
    """
    Plot accuracy and confidence for different token time categories.
    """
    if 'temporal' not in evaluation_results:
        print("No TEMPORAL evaluation results available")
        return

    token_analysis = evaluation_results['temporal'].get('token_analysis')
    if not token_analysis:
        print("No token analysis data available")
        return

    categories = list(token_analysis.keys())
    accuracies = [token_analysis[cat]['accuracy'] for cat in categories]
    confidences = [token_analysis[cat]['avg_confidence'] for cat in categories]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Accuracy by category
    ax1.bar(categories, accuracies, alpha=0.7, edgecolor='black')
    ax1.set_xlabel('Time Value Category')
    ax1.set_ylabel('Prediction Accuracy')
    ax1.set_title('Prediction Accuracy by Token Time Category')
    ax1.grid(True, alpha=0.3, axis='y')

    # Confidence by category
    ax2.bar(categories, confidences, alpha=0.7, edgecolor='black', color='orange')
    ax2.set_xlabel('Time Value Category')
    ax2.set_ylabel('Average Confidence')
    ax2.set_title('Prediction Confidence by Token Time Category')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    output_path = os.path.join(output_dir, 'token_category_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def plot_time_dimensions(time_stats, output_dir):
    """
    Plot first 4 dimensions of time embeddings to show their evolution.
    """
    time_embeddings = time_stats.get('time_values_dim0')
    usage_counts = time_stats.get('usage_counts')

    if time_embeddings is None:
        print("Time embedding dimension data not available")
        return

    # Filter tokens that were actually used
    mask = usage_counts > 0
    used_counts = usage_counts[mask]
    time_dim0 = time_embeddings[mask]

    plt.figure(figsize=(10, 6))
    plt.scatter(used_counts, time_dim0, alpha=0.5, s=20)
    plt.xlabel('Token Usage Count')
    plt.ylabel('Time Embedding Dimension 0 (Usage Count)')
    plt.title('Time Embedding Dimension 0 vs Actual Usage')
    plt.grid(True, alpha=0.3)

    output_path = os.path.join(output_dir, 'time_dim0_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def create_summary_figure(temporal_logs, baseline_logs, evaluation_results, output_dir):
    """
    Create a comprehensive summary figure with multiple subplots.
    """
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # 1. Perplexity comparison
    ax1 = fig.add_subplot(gs[0, 0])
    if temporal_logs.get('val_perplexities') and baseline_logs.get('val_perplexities'):
        temporal_ppls = temporal_logs['val_perplexities']
        baseline_ppls = baseline_logs['val_perplexities']
        ax1.plot([e['step'] for e in temporal_ppls], [e['perplexity'] for e in temporal_ppls],
                label='TEMPORAL', marker='o', markersize=3)
        ax1.plot([e['step'] for e in baseline_ppls], [e['perplexity'] for e in baseline_ppls],
                label='Baseline', marker='s', markersize=3)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Perplexity')
        ax1.set_title('Validation Perplexity')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

    # 2. Time evolution
    ax2 = fig.add_subplot(gs[0, 1])
    if temporal_logs.get('time_stats_history'):
        history = temporal_logs['time_stats_history']
        epochs = [e['epoch'] for e in history]
        means = [e['stats']['mean_time_magnitude'] for e in history]
        ax2.plot(epochs, means, marker='o')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Mean Time Magnitude')
        ax2.set_title('Time Embedding Evolution')
        ax2.grid(True, alpha=0.3)

    # 3. Final perplexity bar chart
    ax3 = fig.add_subplot(gs[0, 2])
    if 'temporal' in evaluation_results and 'baseline' in evaluation_results:
        models = ['TEMPORAL', 'Baseline']
        ppls = [
            evaluation_results['temporal']['test_perplexity'],
            evaluation_results['baseline']['test_perplexity']
        ]
        bars = ax3.bar(models, ppls, alpha=0.7, edgecolor='black')
        bars[0].set_color('blue')
        bars[1].set_color('orange')
        ax3.set_ylabel('Test Perplexity')
        ax3.set_title('Final Test Perplexity')
        ax3.grid(True, alpha=0.3, axis='y')

    # 4. Token category accuracy
    ax4 = fig.add_subplot(gs[1, 0])
    if 'temporal' in evaluation_results and evaluation_results['temporal'].get('token_analysis'):
        analysis = evaluation_results['temporal']['token_analysis']
        categories = list(analysis.keys())
        accuracies = [analysis[cat]['accuracy'] for cat in categories]
        ax4.bar(categories, accuracies, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Time Category')
        ax4.set_ylabel('Accuracy')
        ax4.set_title('Accuracy by Time Category')
        ax4.grid(True, alpha=0.3, axis='y')

    # 5. Token category confidence
    ax5 = fig.add_subplot(gs[1, 1])
    if 'temporal' in evaluation_results and evaluation_results['temporal'].get('token_analysis'):
        analysis = evaluation_results['temporal']['token_analysis']
        categories = list(analysis.keys())
        confidences = [analysis[cat]['avg_confidence'] for cat in categories]
        ax5.bar(categories, confidences, alpha=0.7, edgecolor='black', color='green')
        ax5.set_xlabel('Time Category')
        ax5.set_ylabel('Confidence')
        ax5.set_title('Confidence by Time Category')
        ax5.grid(True, alpha=0.3, axis='y')

    # 6. Summary statistics
    ax6 = fig.add_subplot(gs[1, 2])
    ax6.axis('off')
    summary_text = "Summary Statistics\n" + "="*30 + "\n\n"

    if 'temporal' in evaluation_results:
        t_ppl = evaluation_results['temporal']['test_perplexity']
        summary_text += f"TEMPORAL PPL: {t_ppl:.4f}\n"

    if 'baseline' in evaluation_results:
        b_ppl = evaluation_results['baseline']['test_perplexity']
        summary_text += f"BASELINE PPL: {b_ppl:.4f}\n\n"

    if 'temporal' in evaluation_results and 'baseline' in evaluation_results:
        improvement = ((b_ppl - t_ppl) / b_ppl) * 100
        if improvement > 0:
            summary_text += f"TEMPORAL better by:\n{improvement:.2f}%\n\n"
        else:
            summary_text += f"BASELINE better by:\n{-improvement:.2f}%\n\n"

    if 'temporal' in evaluation_results:
        if 'time_frequency_correlation' in evaluation_results['temporal']:
            corr = evaluation_results['temporal']['time_frequency_correlation']
            summary_text += f"Time-Freq Corr: {corr:.4f}\n"

        if 'confidence_time_correlation' in evaluation_results['temporal']:
            corr = evaluation_results['temporal']['confidence_time_correlation']
            summary_text += f"Conf-Time Corr: {corr:.4f}\n"

    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.suptitle('TEMPORAL Architecture Analysis Summary', fontsize=16, fontweight='bold')

    output_path = os.path.join(output_dir, 'summary_analysis.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Visualize TEMPORAL training and evaluation results")
    parser.add_argument('--log-dir', type=str, default='logs', help='Directory containing training logs')
    parser.add_argument('--output-dir', type=str, default='outputs', help='Directory for evaluation results')
    args = parser.parse_args()

    config = Config()

    # Create output directory for plots
    plots_dir = os.path.join(args.output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)

    print("\n" + "="*60)
    print("TEMPORAL Visualization")
    print("="*60 + "\n")

    # Load training logs
    temporal_log_path = os.path.join(args.log_dir, 'temporal_logs.json')
    baseline_log_path = os.path.join(args.log_dir, 'baseline_logs.json')

    temporal_logs = {}
    baseline_logs = {}

    if os.path.exists(temporal_log_path):
        with open(temporal_log_path, 'r') as f:
            temporal_logs = json.load(f)
        print(f"Loaded TEMPORAL logs from {temporal_log_path}")

    if os.path.exists(baseline_log_path):
        with open(baseline_log_path, 'r') as f:
            baseline_logs = json.load(f)
        print(f"Loaded BASELINE logs from {baseline_log_path}")

    # Load evaluation results
    eval_results_path = os.path.join(args.output_dir, 'evaluation_results.json')
    evaluation_results = {}

    if os.path.exists(eval_results_path):
        with open(eval_results_path, 'r') as f:
            evaluation_results = json.load(f)
        print(f"Loaded evaluation results from {eval_results_path}")

    # Generate plots
    print("\nGenerating visualizations...")

    # Time evolution
    if temporal_logs.get('time_stats_history'):
        plot_time_evolution(temporal_logs['time_stats_history'], plots_dir)

    # Get final time statistics from last epoch
    if temporal_logs.get('time_stats_history'):
        final_time_stats = temporal_logs['time_stats_history'][-1]['stats']

        plot_frequency_vs_time(final_time_stats, plots_dir)
        plot_time_distribution(final_time_stats, plots_dir)
        plot_time_dimensions(final_time_stats, plots_dir)

    # Comparison plots
    if temporal_logs and baseline_logs:
        plot_perplexity_comparison(temporal_logs, baseline_logs, plots_dir)
        plot_training_loss_comparison(temporal_logs, baseline_logs, plots_dir)

    # Token category analysis
    if evaluation_results:
        plot_token_category_analysis(evaluation_results, plots_dir)

    # Summary figure
    if temporal_logs and baseline_logs and evaluation_results:
        create_summary_figure(temporal_logs, baseline_logs, evaluation_results, plots_dir)

    print(f"\nAll visualizations saved to: {plots_dir}")


if __name__ == "__main__":
    main()
