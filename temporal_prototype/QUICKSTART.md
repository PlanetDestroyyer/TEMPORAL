# TEMPORAL Quick Start Guide

Get up and running with the TEMPORAL prototype in 5 minutes.

## Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-compatible GPU for faster training

## Installation

### 1. Clone and Navigate

```bash
cd temporal_prototype
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (deep learning framework)
- NumPy (numerical computing)
- Matplotlib & Seaborn (visualization)
- Datasets (data loading)
- tqdm (progress bars)

### 3. Verify Installation

```bash
python check_syntax.py
```

Should output: `✅ All files have valid Python syntax!`

## Running the Prototype

### Option 1: Quick Test (5 minutes)

Test the implementation without full training:

```bash
python test_implementation.py
```

Expected output: All tests passing ✅

### Option 2: Full Training (30-60 minutes)

Train both TEMPORAL and Baseline models:

```bash
# Train both models
python train.py --model both

# Evaluate both models
python evaluate.py --model both

# Generate visualizations
python visualize.py
```

### Option 3: Train Individual Models

```bash
# Train only TEMPORAL
python train.py --model temporal

# Train only Baseline
python train.py --model baseline
```

## What to Expect

### During Training

```
============================================================
Training TEMPORAL
Parameters: 9,876,544
============================================================

Epoch 1/10:
100%|████████████████| 156/156 [00:45<00:00,  3.42it/s, loss=5.234]
  Train Loss: 5.2341
  Val Loss: 4.8765
  Val Perplexity: 131.23
  Mean Time Magnitude: 0.1234
  Max Time Magnitude: 0.5678

...

Final Results for TEMPORAL:
  Val Loss: 4.2341
  Val Perplexity: 68.92
============================================================
```

### After Evaluation

```
============================================================
Evaluating TEMPORAL
============================================================

Test Loss: 4.2341
Test Perplexity: 68.92

--- TEMPORAL-Specific Metrics ---

Time Embedding Statistics:
  Mean Magnitude: 0.3456
  Max Magnitude: 2.1234
  Min Magnitude: 0.0000

  Correlation: 0.8734

  Token Category Analysis:
    VERY_LOW: Accuracy=0.6210, Confidence=0.4234, Count=1234
    LOW: Accuracy=0.7123, Confidence=0.5432, Count=2345
    MEDIUM: Accuracy=0.8234, Confidence=0.6543, Count=3456
    HIGH: Accuracy=0.8732, Confidence=0.6821, Count=4567

============================================================
COMPARISON SUMMARY
============================================================

Perplexity:
  TEMPORAL: 68.92
  BASELINE: 72.47
  → TEMPORAL is BETTER by 4.89%
```

## Output Files

After running, you'll find:

```
temporal_prototype/
├── checkpoints/
│   ├── temporal_epoch_final.pt      # Trained TEMPORAL model
│   └── baseline_epoch_final.pt      # Trained Baseline model
├── logs/
│   ├── temporal_logs.json           # Training metrics
│   └── baseline_logs.json
└── outputs/
    ├── evaluation_results.json      # Test results
    └── plots/
        ├── summary_analysis.png     # Main summary figure
        ├── time_evolution.png       # Time embedding growth
        ├── frequency_vs_time.png    # Correlation plot
        ├── perplexity_comparison.png
        └── token_category_analysis.png
```

## Quick Configuration Changes

Edit `config.py` to customize:

```python
# Faster training (lower quality)
num_epochs = 5
batch_size = 64

# Higher quality (slower)
num_epochs = 20
batch_size = 16
learning_rate = 0.0001

# Larger model
n_layers = 4
n_heads = 8
content_dim = 256
time_dim = 256
```

## Troubleshooting

### Issue: Out of Memory

**Solution**: Reduce batch size in `config.py`:
```python
batch_size = 16  # or even 8
```

### Issue: Training Too Slow

**Solution**: Use smaller dataset or fewer epochs:
```python
num_epochs = 5
```

Or use GPU if available (automatic detection).

### Issue: Dataset Download Fails

The code will automatically fall back to synthetic data if WikiText-2 can't be downloaded. You'll see:
```
Could not load WikiText-2: [error]
Generating synthetic data for testing...
```

This is fine for testing the prototype!

## Understanding Results

### Key Metrics to Check

1. **Perplexity**: Lower is better
   - TEMPORAL should be ≤ Baseline
   - Difference of 3-5% is significant

2. **Time-Frequency Correlation**: Should be > 0.7
   - Confirms frequent tokens have high time values
   - Validates core mechanism

3. **Accuracy by Time Category**: Should increase with time
   - Very Low < Low < Medium < High
   - Shows experiential learning

4. **Confidence-Time Correlation**: Should be > 0.5
   - Indicates "knowing what it knows"
   - Higher correlation = better epistemic awareness

## Next Steps

1. **Review Results**: Check `outputs/evaluation_results.json`
2. **Analyze Plots**: Open files in `outputs/plots/`
3. **Read Analysis**: See `ANALYSIS.md` for detailed interpretation
4. **Experiment**: Modify hyperparameters in `config.py`
5. **Extend**: Add time decay, multi-modal time, or other features

## Getting Help

- Check `README.md` for detailed documentation
- Review `ANALYSIS.md` for interpretation guidance
- Examine test output from `test_implementation.py`
- Review code comments in source files

## Quick Validation Checklist

- [ ] All syntax checks pass (`check_syntax.py`)
- [ ] All tests pass (`test_implementation.py`)
- [ ] Training completes without errors
- [ ] Time embeddings increase during training
- [ ] Frequent tokens have higher time values
- [ ] TEMPORAL perplexity is competitive with baseline
- [ ] Visualizations generate successfully

## Estimated Time Requirements

- **Syntax Check**: < 1 minute
- **Unit Tests**: 2-3 minutes
- **Training (CPU)**: 30-60 minutes
- **Training (GPU)**: 10-20 minutes
- **Evaluation**: 5-10 minutes
- **Visualization**: 1-2 minutes

**Total**: ~1 hour for complete run on CPU, ~30 minutes on GPU

---

Happy experimenting with TEMPORAL! 🚀
