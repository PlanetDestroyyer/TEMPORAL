"""
Test Inference-Time Learning

This test validates TEMPORAL's unique capability: improving predictions
on repeated text through inference-time updates.

The hypothesis: When the same text is shown multiple times with update_time=True,
the model should get better at predicting it (lower perplexity).

Usage:
  python test_inference_learning.py --config colab
  python test_inference_learning.py --config scaled

Note: The --config argument must match the config used during training!
"""

import torch
import numpy as np
import argparse
from transformers import AutoTokenizer
from model import TemporalTransformer
from config import get_config


def compute_perplexity(model, input_ids, update_time=False):
    """Compute perplexity for a given input"""
    model.eval()
    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids, update_time=update_time)
        loss = outputs['loss'].item()
        perplexity = np.exp(loss)
    return perplexity


def test_inference_learning(config_name='colab'):
    """Test if model improves on repeated text"""
    print("\n" + "="*80)
    print("TEST: INFERENCE-TIME LEARNING")
    print("="*80)
    print("\nThis test validates TEMPORAL's unique capability:")
    print("Does the model get better at predicting text after seeing it multiple times?\n")

    # Load model
    config = get_config(config_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading TEMPORAL model (config: {config_name})...")

    # Determine checkpoint path based on config
    checkpoint_path = f'checkpoints/temporal_production/final.pt'

    # Try to load checkpoint - first attempt with saved config
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Check if checkpoint has config saved
        if 'config' in checkpoint:
            saved_config_dict = checkpoint['config']
            print(f"✓ Found saved config in checkpoint")

            # Recreate config from saved dict
            config_obj = get_config(config_name)
            for key, value in saved_config_dict.items():
                if hasattr(config_obj, key):
                    setattr(config_obj, key, value)
            config = config_obj

        # Create model with correct config
        model = TemporalTransformer(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}\n")

    except FileNotFoundError:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Make sure to train the model first with: python run_colab.py\n")
        return
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print(f"\nTrying to create model with {config_name} config and load checkpoint...")

        try:
            # Create model with specified config
            model = TemporalTransformer(config).to(device)

            # Try to load with strict=False to see what matches
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            print(f"⚠️  Loaded checkpoint with strict=False - some weights may not match")
            print(f"This may happen if checkpoint was saved with different config\n")
        except Exception as e2:
            print(f"❌ Complete failure to load checkpoint: {e2}")
            print("\nTip: Make sure checkpoint config matches test config")
            print(f"Current test config: {config_name}")
            print(f"Try running with: python test_inference_learning.py --config production\n")
            return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Test text
    test_text = """
    The concept of time embeddings in neural networks is fascinating.
    Time embeddings allow models to track their experience with specific tokens.
    This experiential learning enables continual adaptation during inference.
    """

    print(f"Test text: {test_text[:100]}...\n")

    # Tokenize
    input_ids = tokenizer.encode(test_text, return_tensors='pt').to(device)

    # Test 1: Baseline (no updates)
    print("Test 1: Without time updates")
    print("-" * 40)
    ppl_baseline = compute_perplexity(model, input_ids, update_time=False)
    print(f"Perplexity: {ppl_baseline:.4f}\n")

    # Test 2: With time updates over multiple passes
    print("Test 2: With time updates (5 passes)")
    print("-" * 40)
    perplexities = []

    for i in range(5):
        ppl = compute_perplexity(model, input_ids, update_time=True)
        perplexities.append(ppl)
        improvement = ((perplexities[0] - ppl) / perplexities[0]) * 100 if i > 0 else 0
        print(f"Pass {i+1}: Perplexity={ppl:.4f} (Improvement: {improvement:+.2f}%)")

    # Analysis
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)

    initial_ppl = perplexities[0]
    final_ppl = perplexities[-1]
    improvement = ((initial_ppl - final_ppl) / initial_ppl) * 100

    print(f"\nInitial perplexity (1st exposure): {initial_ppl:.4f}")
    print(f"Final perplexity (5th exposure):   {final_ppl:.4f}")
    print(f"Total improvement: {improvement:+.2f}%")

    # Save results
    with open("output.txt", 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("INFERENCE-TIME LEARNING TEST RESULTS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Test text: {test_text[:100]}...\n\n")
        f.write(f"Without updates: {ppl_baseline:.4f}\n")
        f.write(f"With updates (5 passes):\n")
        for i, ppl in enumerate(perplexities):
            f.write(f"  Pass {i+1}: {ppl:.4f}\n")
        f.write(f"\nImprovement: {improvement:+.2f}%\n")

        if improvement > 0:
            f.write("\n✅ INFERENCE-TIME LEARNING WORKS!\n")
        else:
            f.write("\n⚠️  No improvement observed (may need more training)\n")
        f.write("\n")

    # Verdict
    if improvement > 0:
        print("\n✅ INFERENCE-TIME LEARNING WORKS!")
        print("The model improved its predictions after seeing the text multiple times.")
        print("This is TEMPORAL's unique capability that baseline transformers cannot do!")
    elif improvement > -1:
        print("\n⚠️  MARGINAL CHANGE")
        print("No significant improvement observed. This could mean:")
        print("  - Model needs more training epochs")
        print("  - Test text may already be well-represented in training data")
        print("  - Time embeddings need stronger signal")
    else:
        print("\n⚠️  PERFORMANCE DEGRADED")
        print("This is unexpected. Possible reasons:")
        print("  - Model is overfitting to specific examples")
        print("  - Numerical instability in time updates")

    print("\nResults appended to output.txt\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test inference-time learning")
    parser.add_argument('--config', type=str, default='colab',
                        choices=['production', 'colab', 'scaled', 'debug'],
                        help='Configuration to use (must match training config)')
    args = parser.parse_args()

    test_inference_learning(args.config)
