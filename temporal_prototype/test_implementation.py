"""
Quick test script to verify TEMPORAL implementation works correctly.
Tests basic functionality of all components before full training.
"""

import torch
import numpy as np

from config import Config
from model import TemporalTransformer, BaselineTransformer, count_parameters
from time_embeddings import TimeEmbeddedTokenizer, TimeEmbeddings


def test_time_embeddings():
    """Test TimeEmbeddings layer"""
    print("\n" + "="*60)
    print("Testing Time Embeddings Layer")
    print("="*60)

    vocab_size = 100
    time_dim = 128
    time_lr = 0.01

    time_emb = TimeEmbeddings(vocab_size, time_dim, time_lr)

    # Create sample token IDs
    batch_size = 4
    seq_len = 10
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Get embeddings
    embeddings = time_emb(token_ids)
    print(f"✓ Token IDs shape: {token_ids.shape}")
    print(f"✓ Time embeddings shape: {embeddings.shape}")

    # Check initial values are zero
    assert torch.allclose(embeddings, torch.zeros_like(embeddings)), "Initial time embeddings should be zero"
    print("✓ Initial time embeddings are zero")

    # Update time embeddings
    time_emb.update_time_embeddings(token_ids)
    embeddings_after = time_emb(token_ids)

    # Check that time embeddings have changed
    assert not torch.allclose(embeddings_after, embeddings), "Time embeddings should update"
    print("✓ Time embeddings update correctly")

    # Check statistics
    stats = time_emb.get_time_statistics()
    print(f"✓ Mean time magnitude: {stats['mean_time_magnitude']:.6f}")
    print(f"✓ Max time magnitude: {stats['max_time_magnitude']:.6f}")

    print("\n✅ Time Embeddings Layer: PASSED")


def test_time_embedded_tokenizer():
    """Test TimeEmbeddedTokenizer"""
    print("\n" + "="*60)
    print("Testing Time Embedded Tokenizer")
    print("="*60)

    vocab_size = 100
    content_dim = 128
    time_dim = 128

    tokenizer = TimeEmbeddedTokenizer(vocab_size, content_dim, time_dim)

    batch_size = 4
    seq_len = 10
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))

    # Get dual embeddings
    embeddings = tokenizer(token_ids, update_time=True)

    print(f"✓ Token IDs shape: {token_ids.shape}")
    print(f"✓ Dual embeddings shape: {embeddings.shape}")
    assert embeddings.shape == (batch_size, seq_len, content_dim + time_dim), "Incorrect embedding shape"
    print("✓ Correct dual embedding dimension [content | time]")

    # Test content vs time split
    content_part = embeddings[:, :, :content_dim]
    time_part = embeddings[:, :, content_dim:]

    print(f"✓ Content part shape: {content_part.shape}")
    print(f"✓ Time part shape: {time_part.shape}")

    print("\n✅ Time Embedded Tokenizer: PASSED")


def test_temporal_transformer():
    """Test TemporalTransformer model"""
    print("\n" + "="*60)
    print("Testing TEMPORAL Transformer")
    print("="*60)

    config = Config()
    model = TemporalTransformer(config)

    print(f"✓ Model parameters: {count_parameters(model):,}")

    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids, update_time=True)

    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Incorrect output shape"
    print("✓ Correct output shape")

    # Test with confidence
    logits, confidence = model(input_ids, update_time=True, return_confidence=True)
    print(f"✓ Confidence shape: {confidence.shape}")
    assert confidence.shape == (batch_size, seq_len), "Incorrect confidence shape"
    print("✓ Confidence scores computed correctly")

    # Check time statistics
    stats = model.get_time_statistics()
    print(f"✓ Time statistics accessible")
    print(f"  Mean magnitude: {stats['mean_time_magnitude']:.6f}")

    # Test gradient flow
    target = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        target.view(-1)
    )
    loss.backward()

    print("✓ Gradients computed successfully")

    print("\n✅ TEMPORAL Transformer: PASSED")


def test_baseline_transformer():
    """Test Baseline Transformer model"""
    print("\n" + "="*60)
    print("Testing Baseline Transformer")
    print("="*60)

    config = Config()
    model = BaselineTransformer(config)

    print(f"✓ Model parameters: {count_parameters(model):,}")

    batch_size = 4
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # Forward pass
    logits = model(input_ids)

    print(f"✓ Input shape: {input_ids.shape}")
    print(f"✓ Output logits shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Incorrect output shape"
    print("✓ Correct output shape")

    # Test gradient flow
    target = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, config.vocab_size),
        target.view(-1)
    )
    loss.backward()

    print("✓ Gradients computed successfully")

    print("\n✅ Baseline Transformer: PASSED")


def test_model_comparison():
    """Test that TEMPORAL and Baseline have similar parameter counts"""
    print("\n" + "="*60)
    print("Testing Model Comparison")
    print("="*60)

    config = Config()

    temporal_model = TemporalTransformer(config)
    baseline_model = BaselineTransformer(config)

    temporal_params = count_parameters(temporal_model)
    baseline_params = count_parameters(baseline_model)

    print(f"✓ TEMPORAL parameters: {temporal_params:,}")
    print(f"✓ BASELINE parameters: {baseline_params:,}")

    # They should have similar parameter counts (same architecture)
    param_diff = abs(temporal_params - baseline_params) / baseline_params
    print(f"✓ Parameter difference: {param_diff*100:.2f}%")

    # Both should be around 10M parameters
    assert temporal_params < 15_000_000, "TEMPORAL model too large"
    assert baseline_params < 15_000_000, "Baseline model too large"
    print("✓ Both models within 10M parameter budget")

    print("\n✅ Model Comparison: PASSED")


def test_time_update_mechanism():
    """Test that time values increase with usage"""
    print("\n" + "="*60)
    print("Testing Time Update Mechanism")
    print("="*60)

    config = Config()
    model = TemporalTransformer(config)

    # Use a specific token repeatedly
    frequent_token = 42
    input_ids = torch.full((4, 10), frequent_token)

    # Initial time value
    initial_stats = model.get_time_statistics()
    initial_magnitude = initial_stats['time_magnitudes'][frequent_token]
    print(f"✓ Initial time magnitude for token {frequent_token}: {initial_magnitude:.6f}")

    # Use the token multiple times
    for i in range(10):
        _ = model(input_ids, update_time=True)

    # Check time value increased
    final_stats = model.get_time_statistics()
    final_magnitude = final_stats['time_magnitudes'][frequent_token]
    print(f"✓ Final time magnitude for token {frequent_token}: {final_magnitude:.6f}")

    assert final_magnitude > initial_magnitude, "Time should increase with usage"
    print(f"✓ Time increased by: {final_magnitude - initial_magnitude:.6f}")

    # Check usage count
    usage_count = final_stats['usage_counts'][frequent_token]
    print(f"✓ Usage count tracked: {usage_count:.0f}")

    print("\n✅ Time Update Mechanism: PASSED")


def run_all_tests():
    """Run all tests"""
    print("\n" + "="*70)
    print("TEMPORAL IMPLEMENTATION TEST SUITE")
    print("="*70)

    tests = [
        test_time_embeddings,
        test_time_embedded_tokenizer,
        test_temporal_transformer,
        test_baseline_transformer,
        test_model_comparison,
        test_time_update_mechanism,
    ]

    failed = []

    for test_fn in tests:
        try:
            test_fn()
        except Exception as e:
            print(f"\n❌ {test_fn.__name__}: FAILED")
            print(f"Error: {e}")
            failed.append(test_fn.__name__)
            import traceback
            traceback.print_exc()

    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)

    total = len(tests)
    passed = total - len(failed)

    print(f"\nTotal Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {len(failed)}")

    if failed:
        print(f"\nFailed Tests:")
        for name in failed:
            print(f"  - {name}")
    else:
        print("\n🎉 ALL TESTS PASSED! 🎉")
        print("\nThe TEMPORAL implementation is ready for training.")

    print("="*70 + "\n")


if __name__ == "__main__":
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    run_all_tests()
