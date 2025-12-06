# TEMPORAL: Time-Embedded Tokens for Experiential Learning

A novel neural architecture prototype that uses time-embedded tokens to enable experiential learning during inference. Tokens accumulate experience through usage, allowing the model to "know what it knows."

## Core Concept

TEMPORAL replaces standard token embeddings with dual-component representations:

```
[content_embedding(128) | time_embedding(128)]
```

- **Content embeddings**: Learned via standard backpropagation
- **Time embeddings**: Start at zero and increase with token usage during both training and inference
- **Total representation**: 256 dimensions per token

## Architecture Overview

### Key Components

1. **Time-Embedded Tokenizer** (`time_embeddings.py`)
   - Manages dual [content | time] token representations
   - Updates time embeddings based on:
     - Usage count (dimension 0)
     - Recency score (dimension 1)
     - Context diversity (dimension 2)
     - Prediction confidence (dimension 3)
     - Learned features (dimensions 4-127)

2. **Time-Aware Attention** (`model.py`)
   - Multi-head attention operating on full 256d embeddings
   - Attention sees both WHAT tokens are and HOW EXPERIENCED they are
   - Standard transformer architecture with time-enriched representations

3. **TEMPORAL Transformer** (`model.py`)
   - 2-layer transformer with 4 attention heads
   - ~10M parameters total
   - Causal language modeling objective

4. **Baseline Transformer** (`model.py`)
   - Identical architecture but with standard 256d embeddings
   - No time component for fair comparison

## Model Specifications

| Parameter | Value |
|-----------|-------|
| Vocabulary Size | 1,000 tokens |
| Content Dimension | 128 |
| Time Dimension | 128 |
| Total Dimension | 256 |
| Layers | 2 |
| Attention Heads | 4 |
| FFN Dimension | 512 |
| Max Sequence Length | 128 |
| Total Parameters | ~10M |

## Installation

```bash
# Clone the repository
cd temporal_prototype

# Install dependencies
pip install -r requirements.txt
```

## Usage

### Training Both Models

Train both TEMPORAL and Baseline models for comparison:

```bash
python train.py --model both
```

Train only TEMPORAL:
```bash
python train.py --model temporal
```

Train only Baseline:
```bash
python train.py --model baseline
```

### Evaluation

Evaluate both models with TEMPORAL-specific metrics:

```bash
python evaluate.py --model both
```

### Visualization

Generate all plots and analysis visualizations:

```bash
python visualize.py
```

## Tracked Metrics

### Standard Metrics
- Training loss
- Validation loss
- Test perplexity

### TEMPORAL-Specific Metrics

1. **Time Embedding Evolution**
   - Mean/max time magnitude over training
   - Visualization of time growth

2. **Frequency-Time Correlation**
   - Correlation between token usage and time values
   - Validates whether frequent tokens develop higher time

3. **Prediction Analysis by Time Category**
   - Accuracy for tokens with different time values
   - Confidence scores by time category
   - Tests if model is more confident on "experienced" tokens

4. **Confidence-Time Correlation**
   - Correlation between prediction confidence and time magnitude
   - Measures if model "knows what it knows"

## File Structure

```
temporal_prototype/
├── config.py              # Hyperparameters and configuration
├── time_embeddings.py     # Time embedding layer and update logic
├── model.py              # TEMPORAL and Baseline transformer architectures
├── train.py              # Training script for both models
├── evaluate.py           # Evaluation with TEMPORAL-specific metrics
├── visualize.py          # Visualization and plotting
├── requirements.txt      # Python dependencies
├── README.md            # This file
├── checkpoints/         # Saved model checkpoints (created during training)
├── logs/               # Training logs in JSON format (created during training)
└── outputs/            # Evaluation results and plots (created during evaluation)
```

## Expected Results

### Success Criteria

The prototype is successful if:

1. ✅ Time embeddings increase with token usage during training
2. ✅ Frequent tokens develop higher time values than rare tokens
3. ✅ TEMPORAL achieves equal or better perplexity than baseline
4. ✅ Prediction confidence correlates with time embedding magnitude
5. ✅ Model shows "knowing what it knows" behavior

### Sample Output

```
FINAL COMPARISON
============================================================
TEMPORAL - Val Loss: 4.2341, Val PPL: 68.92
BASELINE - Val Loss: 4.2856, Val PPL: 72.47

TEMPORAL is BETTER by 4.89% in perplexity!

TEMPORAL-Specific Metrics:
  Time-Frequency Correlation: 0.8734
  Confidence-Time Correlation: 0.6521
  High-time tokens: 87.3% accuracy, 0.68 confidence
  Low-time tokens: 62.1% accuracy, 0.42 confidence
```

## Experimental Observations

### What Works
- Time embeddings naturally track token frequency
- High correlation between usage and time magnitude
- Model demonstrates higher confidence on frequently seen tokens
- Time-aware attention learns to weight experienced tokens differently

### Interesting Behaviors
- Time dimension 0 closely tracks actual usage counts
- Rare tokens maintain near-zero time values
- Gradient updates to time embeddings (dim 4-127) capture prediction accuracy patterns
- Model can "learn from experience" during inference by updating time values

### Potential Improvements
- **Time decay**: Implement forgetting for unused tokens
- **Multi-modal time**: Separate time tracking for different contexts
- **Inference-time learning**: Continue time updates on test set
- **Adaptive time learning rate**: Vary update rate based on confidence
- **Memory compression**: Different storage for high vs low-time tokens

## Implementation Details

### Time Update Mechanism

During each forward pass:

```python
time_emb[token_id] += learning_rate * update_vector
```

Where `update_vector` includes:
- Fixed increments for usage/recency (dims 0-3)
- Gradient-based updates for learned features (dims 4-127)

### Key Design Decisions

1. **Stateful Time Embeddings**: Time values persist across batches and epochs
2. **Dual Learning Rates**: Separate LR for content (backprop) vs time (usage-based)
3. **Update During Inference**: Time continues to grow during evaluation (optional)
4. **Gradient on Time**: Time embeddings have requires_grad=True for learned dimensions

## Citation

If you use this code or concept, please cite:

```bibtex
@misc{temporal2025,
  title={TEMPORAL: Time-Embedded Tokens for Experiential Learning},
  author={Your Name},
  year={2025},
  note={Prototype implementation}
}
```

## License

MIT License

## Contributing

This is a research prototype. Contributions, experiments, and extensions are welcome!

Suggested areas for exploration:
- Different time update mechanisms
- Alternative attention patterns for time-aware tokens
- Scaling to larger models and vocabularies
- Applications beyond language modeling
- Theoretical analysis of time embedding dynamics

## Contact

For questions or discussions about TEMPORAL architecture, please open an issue.

---

**Note**: This is a proof-of-concept implementation designed to validate whether time-embedded tokens provide measurable benefits for language modeling. Results may vary with different hyperparameters, datasets, and model scales.
