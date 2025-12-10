# Testing Guide for TEMPORAL

This guide explains how to validate and analyze your TEMPORAL results.

## 📋 Overview

After training TEMPORAL, use these test cases to:
1. **Validate reproducibility** across different random seeds
2. **Test inference-time learning** (TEMPORAL's unique capability)
3. **Analyze time embedding patterns** to understand what was learned
4. **Compare performance** across multiple runs

---

## 🎲 1. Multiple Seed Runs (Reproducibility)

### Purpose
Validate that TEMPORAL consistently beats Baseline across different initializations.

### How to Run

```bash
# Run 1 (seed: 42)
python run_colab.py --seed 42

# Run 2 (seed: 123)
python run_colab.py --seed 123

# Run 3 (seed: 777)
python run_colab.py --seed 777
```

Each run:
- Trains both TEMPORAL and Baseline with the specified seed
- Appends results to `output.txt`
- Takes ~30-60 minutes on GPU

### Expected Output

`output.txt` will contain:
```
================================================================================
NEW RUN: 2025-12-10 14:30:00
SEED: 42
================================================================================

======================================================================
TEMPORAL MODEL TRAINING RESULTS (Seed: 42)
======================================================================
Epoch 1: Train Loss=0.9324, Eval Loss=0.7520, Perplexity=2.12
Epoch 2: Train Loss=0.6851, Eval Loss=0.7250, Perplexity=2.06

Final Results:
  Train Loss: 0.6851
  Eval Loss: 0.7250
  Perplexity: 2.06

======================================================================
BASELINE MODEL TRAINING RESULTS (Seed: 42)
======================================================================
Epoch 1: Train Loss=0.9793, Eval Loss=0.7595, Perplexity=2.14
Epoch 2: Train Loss=0.7021, Eval Loss=0.7338, Perplexity=2.08

Final Results:
  Train Loss: 0.7021
  Eval Loss: 0.7338
  Perplexity: 2.08
```

---

## 📊 2. Compare Across Runs

### Purpose
Statistical analysis of TEMPORAL vs Baseline across multiple seeds.

### How to Run

```bash
# After running multiple seeds
python test_comparison.py
```

### What It Does
- Parses `output.txt` to extract all run results
- Computes mean/std of perplexities
- Calculates win rate (how often TEMPORAL beats Baseline)
- Performs statistical significance test (if ≥3 runs)
- Generates `analysis_summary.txt`

### Expected Output

```
MULTI-RUN COMPARISON ANALYSIS
Total completed runs: 3

INDIVIDUAL RUN RESULTS
Run 1 (Seed: 42):
  TEMPORAL:  Perplexity = 2.06
  BASELINE:  Perplexity = 2.08
  Improvement: +0.96%
  Winner: ✅ TEMPORAL

Run 2 (Seed: 123):
  TEMPORAL:  Perplexity = 2.05
  BASELINE:  Perplexity = 2.09
  Improvement: +1.91%
  Winner: ✅ TEMPORAL

Run 3 (Seed: 777):
  TEMPORAL:  Perplexity = 2.07
  BASELINE:  Perplexity = 2.10
  Improvement: +1.43%
  Winner: ✅ TEMPORAL

AGGREGATE STATISTICS
TEMPORAL Model:
  Mean Perplexity: 2.06 ± 0.01
  Min/Max: 2.05 / 2.07

BASELINE Model:
  Mean Perplexity: 2.09 ± 0.01
  Min/Max: 2.08 / 2.10

Improvement:
  Mean: +1.43% ± 0.48%
  Min/Max: +0.96% / +1.91%

TEMPORAL wins: 3/3 runs (100.0%)

CONCLUSION
✅ STRONG EVIDENCE: TEMPORAL consistently outperforms Baseline
   Average improvement: 1.43%
   Wins 3/3 runs
```

---

## 🔄 3. Test Inference-Time Learning

### Purpose
Validate TEMPORAL's unique capability: improving on repeated text.

### How to Run

```bash
python test_inference_learning.py
```

### What It Does
- Loads trained TEMPORAL model
- Tests on a sample text passage
- Measures perplexity on 1st, 2nd, 3rd, 4th, 5th exposure
- Reports if model improves (lower perplexity) after seeing text multiple times

### Expected Output

```
TEST: INFERENCE-TIME LEARNING
This test validates TEMPORAL's unique capability:
Does the model get better at predicting text after seeing it multiple times?

Test text: The concept of time embeddings in neural networks is fascinating...

Test 1: Without time updates
Perplexity: 2.06

Test 2: With time updates (5 passes)
Pass 1: Perplexity=2.06 (Improvement: +0.00%)
Pass 2: Perplexity=2.04 (Improvement: +0.97%)
Pass 3: Perplexity=2.02 (Improvement: +1.94%)
Pass 4: Perplexity=2.01 (Improvement: +2.43%)
Pass 5: Perplexity=2.00 (Improvement: +2.91%)

RESULTS
Initial perplexity (1st exposure): 2.06
Final perplexity (5th exposure):   2.00
Total improvement: +2.91%

✅ INFERENCE-TIME LEARNING WORKS!
The model improved its predictions after seeing the text multiple times.
This is TEMPORAL's unique capability that baseline transformers cannot do!
```

**Note**: If no improvement is observed:
- Model may need more training epochs
- Test text may already be well-represented in training data
- Try with different/novel text

---

## 🔍 4. Analyze Time Embedding Patterns

### Purpose
Understand what the time embeddings learned.

### How to Run

```bash
python test_time_patterns.py
```

### What It Does
- Extracts time embeddings from trained model
- Analyzes dimension statistics (variance, mean, range)
- Examines sample tokens (common vs rare words)
- Tests correlation with usage frequency
- Creates visualizations (if matplotlib available)

### Expected Output

```
TIME EMBEDDING ANALYSIS
Time embedding shape: (50257, 256)
  Vocabulary size: 50257
  Time dimensions: 256

OVERALL STATISTICS
Time embedding norms:
  Mean: 0.1234 ± 0.0567
  Min:  0.0012
  Max:  0.8934

DIMENSION ANALYSIS
Top 5 most active dimensions (by variance):

  1. Dimension 42:
     Variance: 0.012345
     Mean: 0.000123
     Range: [-0.5234, 0.4567]

  2. Dimension 7:
     Variance: 0.009876
     Mean: -0.001234
     Range: [-0.4321, 0.3456]

SAMPLE TOKEN ANALYSIS
Comparing time embeddings for different frequency tokens:

'the' (ID: 262):
  Norm: 0.2345
  Top 3 dimensions: [0.123, -0.089, 0.045]

'xenophobia' (ID: 35810):
  Norm: 0.0123
  Top 3 dimensions: [0.012, 0.008, -0.003]

✅ Analysis complete!
✓ Visualization saved to: time_embedding_analysis.png
```

---

## 📝 Summary of Outputs

After running all tests, you'll have:

1. **`output.txt`** - Raw results from all training runs
2. **`analysis_summary.txt`** - Statistical comparison across runs
3. **`time_embedding_analysis.png`** - Visualizations of learned patterns
4. **Console output** - Detailed analysis and conclusions

---

## 🎯 Validation Checklist

For a complete validation, ensure:

- [ ] Run training with at least 3 different seeds
- [ ] TEMPORAL beats Baseline in majority of runs
- [ ] Average improvement > 0.5%
- [ ] Inference-time learning shows improvement
- [ ] Time embeddings show meaningful patterns
- [ ] Results are reproducible across seeds

---

## 🚀 Next Steps

After validation:

1. **Scale up**: Try WikiText-103 or larger models
2. **Ablation studies**: Test different time_dim sizes
3. **Novel datasets**: Test on domain-specific data
4. **Continual learning**: Test adaptation to new domains
5. **Publication**: Write up results for arXiv/conference

---

## ❓ Troubleshooting

**Q: TEMPORAL not beating Baseline?**
- Try more training epochs (currently 2, try 5-10)
- Check if dataset is too small (use WikiText-103)
- Verify gradient flow is enabled
- Increase time_dim (try 384 or 512)

**Q: Inference-time learning not working?**
- Model may need more training
- Test with more novel text
- Check update_time=True is being passed
- Verify time embeddings have requires_grad=True

**Q: High variance across seeds?**
- Normal with small datasets (WikiText-2)
- Run more seeds for statistical significance
- Use larger dataset for stability

---

## 📞 Support

If you encounter issues:
1. Check `output.txt` for error messages
2. Verify checkpoints were saved
3. Ensure GPU has enough memory
4. Try with debug config first

Happy testing! 🎉
