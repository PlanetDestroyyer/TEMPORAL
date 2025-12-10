#!/usr/bin/env python3
"""
Train TEMPORAL Only (No Baseline Comparison)

Quick validation script that trains only TEMPORAL model
to validate the architecture works at scale.

Usage:
  python train_temporal_only.py --config scaled --seed 42
  python train_temporal_only.py --config colab --seed 123
"""

import os
import sys
import argparse
import subprocess
from datetime import datetime


def print_section(title):
    """Print a formatted section header"""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def run_cmd(cmd):
    """Run a shell command and handle errors"""
    print(f"Running: {cmd}\n")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"\n❌ Command failed with exit code {result.returncode}")
        sys.exit(1)
    print()


def check_environment():
    """Check Python/PyTorch environment"""
    print_section("ENVIRONMENT CHECK")

    import sys
    print(f"✓ Python: {sys.version}")

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
        print(f"✓ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("❌ PyTorch not installed!")
        print("Run: pip install torch")
        sys.exit(1)

    try:
        import transformers
        print(f"✓ Transformers: {transformers.__version__}")
    except ImportError:
        print("❌ Transformers not installed!")
        print("Run: pip install transformers datasets")
        sys.exit(1)


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description='Train TEMPORAL model only (skip baseline)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--config', type=str, default='scaled',
                        choices=['production', 'colab', 'scaled', 'debug'],
                        help='Configuration preset (default: scaled)')
    args = parser.parse_args()

    seed = args.seed
    config_name = args.config
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Open output file for appending
    output_file = "output.txt"

    def log_and_print(msg):
        """Print to console and write to file"""
        print(msg)
        with open(output_file, 'a') as f:
            f.write(msg + '\n')

    # Write header to output file
    with open(output_file, 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write(f"TEMPORAL ONLY TRAINING (NO BASELINE)\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"CONFIG: {config_name}\n")
        f.write(f"SEED: {seed}\n")
        f.write("="*80 + "\n\n")

    print_section("TEMPORAL: Architecture Validation (TEMPORAL Only)")
    log_and_print(f"Run timestamp: {timestamp}")
    log_and_print(f"Configuration: {config_name}")
    log_and_print(f"Random seed: {seed}\n")
    log_and_print("Training TEMPORAL model only (no baseline comparison).\n")
    log_and_print("Features:")
    log_and_print("  ✓ Self-learning time embeddings (gradient-based)")
    log_and_print("  ✓ Large model: 12 layers, 384-dim (190M params)")
    log_and_print("  ✓ WikiText-2 dataset, 10 epochs")
    log_and_print("  ✓ SOTA components: RMSNorm, SwiGLU, Flash Attention")
    log_and_print("  ✓ Mixed precision training (BF16)")
    log_and_print(f"  ✓ Outputs saved to: {output_file}")

    # Estimate time based on config
    if config_name == 'scaled':
        log_and_print("\nEstimated time: ~1.5-2 hours on P100 GPU (TEMPORAL only)\n")
    elif config_name == 'debug':
        log_and_print("\nEstimated time: 2-5 minutes\n")
    else:
        log_and_print("\nEstimated time: ~30 minutes on P100 GPU\n")

    # Check environment
    check_environment()

    # Train TEMPORAL model ONLY
    print_section("Training TEMPORAL Model")
    log_and_print("Training large-scale TEMPORAL architecture:")
    log_and_print("  - 12 layers (2x colab)")
    log_and_print("  - 384-dim embeddings (+50%)")
    log_and_print("  - 10 epochs (5x colab)")
    log_and_print("  - WikiText-2 full dataset")
    log_and_print("  - 360k sample-epochs total\n")

    run_cmd(f"python train.py --config {config_name} --model-type temporal --seed {seed}")

    # Done
    print_section("Training Complete!")
    log_and_print("✅ TEMPORAL model training complete!")
    log_and_print("\nNext steps:")
    log_and_print("1. Check final perplexity in output above")
    log_and_print(f"2. Run inference test: python test_inference_learning.py --config {config_name}")
    log_and_print(f"3. Analyze embeddings: python test_time_patterns.py --config {config_name}")
    log_and_print("\nTo validate the model works:")
    log_and_print(f"  python validate_model.py --config {config_name}\n")


if __name__ == "__main__":
    main()
