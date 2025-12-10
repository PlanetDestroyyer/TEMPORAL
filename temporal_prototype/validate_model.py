#!/usr/bin/env python3
"""
Simple Model Validation Script

Tests if the trained TEMPORAL model can predict next tokens correctly.
This validates that the architecture works without full comparison testing.

Usage:
  python validate_model.py --config scaled
  python validate_model.py --config colab
"""

import torch
import argparse
from transformers import AutoTokenizer
from model import TemporalTransformer
from config import get_config


def validate_model(config_name='scaled'):
    """Simple validation: test next token prediction"""
    print("\n" + "="*80)
    print("TEMPORAL MODEL VALIDATION")
    print("="*80)
    print("\nValidating that the trained model can predict next tokens...\n")

    # Load config
    config = get_config(config_name)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Configuration: {config_name}")
    print(f"Device: {device}\n")

    # Load checkpoint
    checkpoint_path = 'checkpoints/temporal_production/final.pt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Create model
        model = TemporalTransformer(config).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print(f"✓ Loaded model from {checkpoint_path}")
        print(f"✓ Model has {sum(p.numel() for p in model.parameters()):,} parameters\n")

    except FileNotFoundError:
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        print("Train the model first with: python train_temporal_only.py\n")
        return
    except Exception as e:
        print(f"❌ Error loading model: {e}\n")
        return

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    print("✓ Loaded GPT-2 tokenizer\n")

    # Test cases
    print("="*80)
    print("NEXT TOKEN PREDICTION TESTS")
    print("="*80 + "\n")

    test_prompts = [
        "The quick brown fox",
        "Once upon a time",
        "In the year",
        "The capital of France is",
        "Machine learning is",
    ]

    for i, prompt in enumerate(test_prompts, 1):
        print(f"Test {i}: \"{prompt}\"")
        print("-" * 60)

        # Tokenize
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        # Get predictions
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs['logits']

            # Get top 5 predictions for next token
            next_token_logits = logits[0, -1, :]
            top_k = torch.topk(next_token_logits, k=5)

            print(f"Top 5 next token predictions:")
            for rank, (score, token_id) in enumerate(zip(top_k.values, top_k.indices), 1):
                token = tokenizer.decode([token_id])
                prob = torch.softmax(next_token_logits, dim=-1)[token_id].item()
                print(f"  {rank}. \"{token}\" (prob: {prob:.4f}, score: {score:.2f})")

        print()

    # Perplexity test
    print("="*80)
    print("PERPLEXITY TEST")
    print("="*80 + "\n")

    test_text = "The quick brown fox jumps over the lazy dog. This is a test sentence."
    print(f"Test text: \"{test_text}\"\n")

    input_ids = tokenizer.encode(test_text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs['loss'].item()
        perplexity = torch.exp(torch.tensor(loss)).item()

        print(f"Loss: {loss:.4f}")
        print(f"Perplexity: {perplexity:.4f}\n")

    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80 + "\n")

    print("✅ Model loaded successfully")
    print("✅ Can make next token predictions")
    print(f"✅ Perplexity: {perplexity:.4f}")
    print("\n👉 The TEMPORAL architecture works!")
    print("\nFor more detailed analysis:")
    print(f"  - Inference learning: python test_inference_learning.py --config {config_name}")
    print(f"  - Time embeddings: python test_time_patterns.py --config {config_name}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate trained TEMPORAL model")
    parser.add_argument('--config', type=str, default='scaled',
                        choices=['production', 'colab', 'scaled', 'debug'],
                        help='Configuration to use (must match training config)')
    args = parser.parse_args()

    validate_model(args.config)
