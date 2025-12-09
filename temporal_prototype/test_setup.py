"""
Quick test to verify all imports and basic functionality
Run this before training to catch errors early
"""

import sys
import os

print("="*70)
print("TEMPORAL Pre-Flight Check")
print("="*70)

errors = []

# Test 1: Import config
print("\n1. Testing config import...")
try:
    from config import get_config, print_config
    config = get_config('colab')
    print("✓ Config imported successfully")
except Exception as e:
    errors.append(f"Config import failed: {e}")
    print(f"✗ Config import failed: {e}")

# Test 2: Import time_embeddings
print("\n2. Testing time_embeddings import...")
try:
    from time_embeddings import TimeEmbeddings, TimeEmbeddedTokenizer
    print("✓ Time embeddings imported successfully")
except Exception as e:
    errors.append(f"Time embeddings import failed: {e}")
    print(f"✗ Time embeddings import failed: {e}")

# Test 3: Import model
print("\n3. Testing model import...")
try:
    from model import TemporalTransformer, BaselineTransformer, create_model, verify_gradient_flow
    print("✓ Model imported successfully")
except Exception as e:
    errors.append(f"Model import failed: {e}")
    print(f"✗ Model import failed: {e}")

# Test 4: Create model instance
print("\n4. Testing model creation...")
try:
    import torch
    config = get_config('colab')
    model = create_model(config, 'temporal')
    print(f"✓ TEMPORAL model created: {model.count_parameters()/1e6:.1f}M params")

    baseline = create_model(config, 'baseline')
    print(f"✓ Baseline model created: {baseline.count_parameters()/1e6:.1f}M params")
except Exception as e:
    errors.append(f"Model creation failed: {e}")
    print(f"✗ Model creation failed: {e}")

# Test 5: Verify gradient flow
print("\n5. Testing gradient flow verification...")
try:
    result = verify_gradient_flow(model)
    if result:
        print("✓ Gradient flow verification passed")
    else:
        errors.append("Gradient flow verification failed")
        print("✗ Gradient flow verification failed")
except Exception as e:
    errors.append(f"Gradient verification failed: {e}")
    print(f"✗ Gradient verification failed: {e}")

# Test 6: Test forward pass
print("\n6. Testing forward pass...")
try:
    batch_size = 2
    seq_len = 32
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))

    # TEMPORAL forward
    output = model(input_ids, labels=input_ids)
    assert 'loss' in output and output['loss'] is not None
    assert 'logits' in output
    print(f"✓ TEMPORAL forward pass successful (loss: {output['loss'].item():.4f})")

    # Baseline forward
    output = baseline(input_ids, labels=input_ids)
    assert 'loss' in output and output['loss'] is not None
    print(f"✓ Baseline forward pass successful (loss: {output['loss'].item():.4f})")
except Exception as e:
    errors.append(f"Forward pass failed: {e}")
    print(f"✗ Forward pass failed: {e}")

# Test 7: Test backward pass
print("\n7. Testing backward pass...")
try:
    output = model(input_ids, labels=input_ids)
    loss = output['loss']
    loss.backward()

    # Check gradients exist
    has_grad = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    if has_grad:
        print("✓ Backward pass successful (gradients computed)")
    else:
        errors.append("No gradients computed")
        print("✗ No gradients computed")
except Exception as e:
    errors.append(f"Backward pass failed: {e}")
    print(f"✗ Backward pass failed: {e}")

# Test 8: Check time embeddings are updated
print("\n8. Testing time embedding updates...")
try:
    initial_time = model.tokenizer.time_embeddings.time_embeddings.data.clone()

    # Do a forward pass with update_time=True
    output = model(input_ids, update_time=True)

    final_time = model.tokenizer.time_embeddings.time_embeddings.data

    # Check if any time values changed
    changed = not torch.allclose(initial_time, final_time)
    if changed:
        print("✓ Time embeddings update during forward pass")
    else:
        print("⚠ Time embeddings didn't change (this is OK during first pass)")
except Exception as e:
    errors.append(f"Time update test failed: {e}")
    print(f"✗ Time update test failed: {e}")

# Summary
print("\n" + "="*70)
if not errors:
    print("✅ ALL CHECKS PASSED! Ready to train.")
else:
    print(f"❌ {len(errors)} ERROR(S) FOUND:")
    for i, err in enumerate(errors, 1):
        print(f"  {i}. {err}")
    sys.exit(1)
print("="*70)
