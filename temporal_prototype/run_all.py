"""
Complete TEMPORAL Pipeline - Single Script
Runs training, evaluation, and visualization automatically.

Perfect for Kaggle/Colab: Just run this one file!
"""

import os
import sys
import subprocess

def print_header(title):
    """Print a nice header"""
    print("\n" + "="*70)
    print(f"  {title}")
    print("="*70 + "\n")

def run_command(cmd, description):
    """Run a command and handle errors"""
    print(f"▶ {description}...")
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"❌ Failed: {description}")
        sys.exit(1)
    print(f"✅ Completed: {description}\n")

def main():
    print_header("TEMPORAL: Complete Pipeline")
    print("This will train, evaluate, and visualize the TEMPORAL architecture.")
    print("Estimated time: 30-60 minutes on CPU, 10-20 minutes on GPU\n")

    # Check if we're in the right directory
    if not os.path.exists('config.py'):
        print("❌ Error: Please run this from the temporal_prototype directory")
        sys.exit(1)

    # Step 1: Syntax check
    print_header("Step 1/5: Syntax Validation")
    run_command("python check_syntax.py", "Validating Python syntax")

    # Step 2: Run tests
    print_header("Step 2/5: Unit Tests")
    print("⚠️  This requires PyTorch. If it fails, dependencies may be missing.")
    print("Run: pip install -r requirements.txt\n")
    try:
        run_command("python test_implementation.py", "Running unit tests")
    except:
        print("⚠️  Tests skipped (dependencies may be missing)")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            sys.exit(1)

    # Step 3: Training
    print_header("Step 3/5: Training Both Models")
    print("Training TEMPORAL and Baseline models...")
    print("This is the longest step - grab a coffee! ☕\n")
    run_command("python train.py --model both", "Training models")

    # Step 4: Evaluation
    print_header("Step 4/5: Evaluation & Analysis")
    run_command("python evaluate.py --model both", "Evaluating models")

    # Step 5: Visualization
    print_header("Step 5/5: Generating Visualizations")
    run_command("python visualize.py", "Creating plots")

    # Success!
    print_header("✅ COMPLETE!")
    print("All steps finished successfully!\n")
    print("📊 Results:")
    print(f"  - Checkpoints: checkpoints/")
    print(f"  - Logs: logs/")
    print(f"  - Results: outputs/evaluation_results.json")
    print(f"  - Plots: outputs/plots/\n")

    print("🎨 View your results:")
    print(f"  - Main summary: outputs/plots/summary_analysis.png")
    print(f"  - Time evolution: outputs/plots/time_evolution.png")
    print(f"  - Comparison: outputs/plots/perplexity_comparison.png\n")

    print("Next steps:")
    print("  - Review outputs/evaluation_results.json for detailed metrics")
    print("  - Check outputs/plots/ for all visualizations")
    print("  - Read ANALYSIS.md for interpretation guidance\n")

if __name__ == "__main__":
    main()
