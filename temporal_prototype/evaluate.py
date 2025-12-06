"""
Evaluation script for TEMPORAL and Baseline models.
Analyzes and compares model performance with TEMPORAL-specific metrics.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import argparse
from collections import defaultdict

from config import Config
from model import TemporalTransformer, BaselineTransformer
from train import TextDataset, load_wikitext_data


class ModelEvaluator:
    """Evaluator for analyzing model performance"""

    def __init__(self, model, config, test_dataset, model_name="model"):
        self.model = model
        self.config = config
        self.test_dataset = test_dataset
        self.model_name = model_name

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

        self.criterion = nn.CrossEntropyLoss(reduction='none')

    def evaluate_perplexity(self):
        """Compute perplexity on test set"""
        from torch.utils.data import DataLoader

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        total_loss = 0
        total_tokens = 0

        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(self.device), y.to(self.device)

                if isinstance(self.model, TemporalTransformer):
                    logits = self.model(x, update_time=False)
                else:
                    logits = self.model(x)

                loss = F.cross_entropy(
                    logits.view(-1, self.config.vocab_size),
                    y.view(-1)
                )

                total_loss += loss.item() * y.numel()
                total_tokens += y.numel()

        avg_loss = total_loss / total_tokens
        perplexity = np.exp(avg_loss)

        return avg_loss, perplexity

    def analyze_token_predictions(self, num_samples=100):
        """
        Analyze prediction quality for tokens with different time values.
        TEMPORAL-specific metric.
        """
        if not isinstance(self.model, TemporalTransformer):
            return None

        from torch.utils.data import DataLoader

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False
        )

        # Get time statistics
        time_stats = self.model.get_time_statistics()
        time_magnitudes = time_stats['time_magnitudes']

        # Categorize tokens by time magnitude
        time_thresholds = np.percentile(time_magnitudes, [25, 50, 75])
        token_categories = {
            'very_low': [],
            'low': [],
            'medium': [],
            'high': []
        }

        for token_id, mag in enumerate(time_magnitudes):
            if mag < time_thresholds[0]:
                token_categories['very_low'].append(token_id)
            elif mag < time_thresholds[1]:
                token_categories['low'].append(token_id)
            elif mag < time_thresholds[2]:
                token_categories['medium'].append(token_id)
            else:
                token_categories['high'].append(token_id)

        # Analyze predictions for each category
        category_stats = defaultdict(lambda: {'correct': 0, 'total': 0, 'confidence': []})

        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if i >= num_samples:
                    break

                x, y = x.to(self.device), y.to(self.device)
                logits, confidence = self.model(x, update_time=False, return_confidence=True)

                predictions = logits.argmax(dim=-1)

                # Check each token
                for pos in range(y.shape[1]):
                    true_token = y[0, pos].item()
                    pred_token = predictions[0, pos].item()
                    conf = confidence[0, pos].item()

                    # Determine category
                    for cat_name, tokens in token_categories.items():
                        if true_token in tokens:
                            category_stats[cat_name]['total'] += 1
                            if pred_token == true_token:
                                category_stats[cat_name]['correct'] += 1
                            category_stats[cat_name]['confidence'].append(conf)

        # Compute accuracy and average confidence for each category
        results = {}
        for cat_name, stats in category_stats.items():
            if stats['total'] > 0:
                accuracy = stats['correct'] / stats['total']
                avg_confidence = np.mean(stats['confidence'])
                results[cat_name] = {
                    'accuracy': accuracy,
                    'avg_confidence': avg_confidence,
                    'count': stats['total']
                }

        return results

    def analyze_time_frequency_correlation(self):
        """
        Analyze correlation between token frequency and time values.
        TEMPORAL-specific metric.
        """
        if not isinstance(self.model, TemporalTransformer):
            return None

        time_stats = self.model.get_time_statistics()
        usage_counts = time_stats['usage_counts']
        time_magnitudes = time_stats['time_magnitudes']

        # Compute correlation
        correlation = np.corrcoef(usage_counts, time_magnitudes)[0, 1]

        # Find top frequent tokens
        top_indices = np.argsort(usage_counts)[-20:]
        top_frequencies = usage_counts[top_indices]
        top_time_values = time_magnitudes[top_indices]

        return {
            'correlation': correlation,
            'top_tokens': {
                'indices': top_indices.tolist(),
                'frequencies': top_frequencies.tolist(),
                'time_magnitudes': top_time_values.tolist()
            }
        }

    def analyze_confidence_time_correlation(self, num_samples=100):
        """
        Analyze correlation between prediction confidence and time magnitude.
        TEMPORAL-specific metric.
        """
        if not isinstance(self.model, TemporalTransformer):
            return None

        from torch.utils.data import DataLoader

        test_loader = DataLoader(
            self.test_dataset,
            batch_size=1,
            shuffle=False
        )

        time_stats = self.model.get_time_statistics()
        time_magnitudes = time_stats['time_magnitudes']

        confidences = []
        token_times = []

        with torch.no_grad():
            for i, (x, y) in enumerate(test_loader):
                if i >= num_samples:
                    break

                x, y = x.to(self.device), y.to(self.device)
                logits, confidence = self.model(x, update_time=False, return_confidence=True)

                for pos in range(y.shape[1]):
                    true_token = y[0, pos].item()
                    conf = confidence[0, pos].item()
                    time_mag = time_magnitudes[true_token]

                    confidences.append(conf)
                    token_times.append(time_mag)

        # Compute correlation
        correlation = np.corrcoef(token_times, confidences)[0, 1]

        return {
            'correlation': correlation,
            'mean_confidence': np.mean(confidences),
            'std_confidence': np.std(confidences)
        }

    def full_evaluation(self):
        """Run all evaluation metrics"""
        print(f"\n{'='*60}")
        print(f"Evaluating {self.model_name}")
        print(f"{'='*60}\n")

        # Standard metrics
        test_loss, test_ppl = self.evaluate_perplexity()
        print(f"Test Loss: {test_loss:.4f}")
        print(f"Test Perplexity: {test_ppl:.4f}")

        results = {
            'model_name': self.model_name,
            'test_loss': test_loss,
            'test_perplexity': test_ppl
        }

        # TEMPORAL-specific metrics
        if isinstance(self.model, TemporalTransformer):
            print("\n--- TEMPORAL-Specific Metrics ---")

            # Time statistics
            time_stats = self.model.get_time_statistics()
            print(f"\nTime Embedding Statistics:")
            print(f"  Mean Magnitude: {time_stats['mean_time_magnitude']:.4f}")
            print(f"  Max Magnitude: {time_stats['max_time_magnitude']:.4f}")
            print(f"  Min Magnitude: {time_stats['min_time_magnitude']:.4f}")

            results['time_stats'] = {
                'mean_magnitude': time_stats['mean_time_magnitude'],
                'max_magnitude': time_stats['max_time_magnitude'],
                'min_magnitude': time_stats['min_time_magnitude']
            }

            # Time-frequency correlation
            print("\nAnalyzing time-frequency correlation...")
            time_freq_corr = self.analyze_time_frequency_correlation()
            if time_freq_corr:
                print(f"  Correlation: {time_freq_corr['correlation']:.4f}")
                results['time_frequency_correlation'] = time_freq_corr['correlation']

            # Token prediction analysis by time value
            print("\nAnalyzing predictions by token time value...")
            token_analysis = self.analyze_token_predictions()
            if token_analysis:
                print("\n  Token Category Analysis:")
                for cat, stats in token_analysis.items():
                    print(f"    {cat.upper()}: Accuracy={stats['accuracy']:.4f}, "
                          f"Confidence={stats['avg_confidence']:.4f}, "
                          f"Count={stats['count']}")
                results['token_analysis'] = token_analysis

            # Confidence-time correlation
            print("\nAnalyzing confidence-time correlation...")
            conf_time_corr = self.analyze_confidence_time_correlation()
            if conf_time_corr:
                print(f"  Correlation: {conf_time_corr['correlation']:.4f}")
                print(f"  Mean Confidence: {conf_time_corr['mean_confidence']:.4f}")
                results['confidence_time_correlation'] = conf_time_corr['correlation']

        print(f"\n{'='*60}\n")

        return results


def load_model(model_type, config, checkpoint_path):
    """Load a trained model from checkpoint"""
    if model_type == 'temporal':
        model = TemporalTransformer(config)
    else:
        model = BaselineTransformer(config)

    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])

    return model


def main():
    parser = argparse.ArgumentParser(description="Evaluate TEMPORAL and Baseline models")
    parser.add_argument('--model', type=str, choices=['temporal', 'baseline', 'both'], default='both',
                        help='Which model to evaluate')
    parser.add_argument('--checkpoint-dir', type=str, default='checkpoints',
                        help='Directory containing model checkpoints')
    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Load test data
    _, _, test_tokens, vocab = load_wikitext_data(config)
    test_dataset = TextDataset(test_tokens, config.max_seq_length)

    print(f"Test dataset size: {len(test_dataset)}")

    all_results = {}

    # Evaluate TEMPORAL model
    if args.model in ['temporal', 'both']:
        temporal_checkpoint = os.path.join(args.checkpoint_dir, 'temporal_epoch_final.pt')
        if os.path.exists(temporal_checkpoint):
            temporal_model = load_model('temporal', config, temporal_checkpoint)
            temporal_evaluator = ModelEvaluator(temporal_model, config, test_dataset, "TEMPORAL")
            temporal_results = temporal_evaluator.full_evaluation()
            all_results['temporal'] = temporal_results
        else:
            print(f"TEMPORAL checkpoint not found: {temporal_checkpoint}")

    # Evaluate Baseline model
    if args.model in ['baseline', 'both']:
        baseline_checkpoint = os.path.join(args.checkpoint_dir, 'baseline_epoch_final.pt')
        if os.path.exists(baseline_checkpoint):
            baseline_model = load_model('baseline', config, baseline_checkpoint)
            baseline_evaluator = ModelEvaluator(baseline_model, config, test_dataset, "BASELINE")
            baseline_results = baseline_evaluator.full_evaluation()
            all_results['baseline'] = baseline_results
        else:
            print(f"BASELINE checkpoint not found: {baseline_checkpoint}")

    # Compare results
    if args.model == 'both' and 'temporal' in all_results and 'baseline' in all_results:
        print("\n" + "="*60)
        print("COMPARISON SUMMARY")
        print("="*60)
        print(f"\nPerplexity:")
        print(f"  TEMPORAL: {all_results['temporal']['test_perplexity']:.4f}")
        print(f"  BASELINE: {all_results['baseline']['test_perplexity']:.4f}")

        ppl_diff = all_results['baseline']['test_perplexity'] - all_results['temporal']['test_perplexity']
        ppl_improvement = (ppl_diff / all_results['baseline']['test_perplexity']) * 100

        if ppl_diff > 0:
            print(f"  → TEMPORAL is BETTER by {ppl_improvement:.2f}%")
        else:
            print(f"  → BASELINE is better by {-ppl_improvement:.2f}%")

    # Save results
    os.makedirs(config.output_dir, exist_ok=True)
    results_path = os.path.join(config.output_dir, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == "__main__":
    main()
