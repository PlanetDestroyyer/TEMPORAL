"""
Analyze Time Embedding Patterns

This test examines what the time embeddings have learned:
- Do frequent words have different time patterns than rare words?
- Are there interpretable dimensions?
- How do time embeddings cluster?

Usage:
  python test_time_patterns.py
"""

import torch
import numpy as np
from transformers import AutoTokenizer
from model import TemporalTransformer
from config import get_config
import matplotlib.pyplot as plt
from collections import Counter


def analyze_time_embeddings():
    """Analyze learned time embedding patterns"""
    print("\n" + "="*80)
    print("TIME EMBEDDING ANALYSIS")
    print("="*80)
    print("\nAnalyzing what patterns the time embeddings discovered...\n")

    # Load model
    config = get_config('colab')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"Loading TEMPORAL model...")
    model = TemporalTransformer(config).to(device)

    # Load checkpoint
    checkpoint_path = 'checkpoints/temporal_production/final.pt'
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded checkpoint from {checkpoint_path}\n")
    except Exception as e:
        print(f"❌ Failed to load checkpoint: {e}")
        print("Make sure to train the model first with: python run_colab.py\n")
        return

    # Get time embeddings
    time_emb = model.tokenizer.time_embeddings.time_embeddings.detach().cpu().numpy()
    vocab_size, time_dim = time_emb.shape

    print(f"Time embedding shape: {time_emb.shape}")
    print(f"  Vocabulary size: {vocab_size}")
    print(f"  Time dimensions: {time_dim}\n")

    # Load tokenizer for word analysis
    tokenizer = AutoTokenizer.from_pretrained('gpt2')

    # Analyze statistics
    print("="*80)
    print("OVERALL STATISTICS")
    print("="*80 + "\n")

    mean_norm = np.linalg.norm(time_emb, axis=1).mean()
    std_norm = np.linalg.norm(time_emb, axis=1).std()
    print(f"Time embedding norms:")
    print(f"  Mean: {mean_norm:.4f} ± {std_norm:.4f}")
    print(f"  Min:  {np.linalg.norm(time_emb, axis=1).min():.4f}")
    print(f"  Max:  {np.linalg.norm(time_emb, axis=1).max():.4f}\n")

    # Analyze dimensions
    print("="*80)
    print("DIMENSION ANALYSIS")
    print("="*80 + "\n")

    print("Top 5 most active dimensions (by variance):\n")
    dim_vars = np.var(time_emb, axis=0)
    top_dims = np.argsort(dim_vars)[::-1][:5]

    for rank, dim in enumerate(top_dims, 1):
        var = dim_vars[dim]
        mean = np.mean(time_emb[:, dim])
        print(f"  {rank}. Dimension {dim}:")
        print(f"     Variance: {var:.6f}")
        print(f"     Mean: {mean:.6f}")
        print(f"     Range: [{time_emb[:, dim].min():.4f}, {time_emb[:, dim].max():.4f}]")

    # Sample word analysis
    print("\n" + "="*80)
    print("SAMPLE TOKEN ANALYSIS")
    print("="*80 + "\n")

    sample_tokens = [
        "the", "and", "of", "to", "in",     # Very common
        "python", "neural", "network",       # Domain specific
        "xenophobia", "quintessential"       # Rare words
    ]

    print("Comparing time embeddings for different frequency tokens:\n")

    for word in sample_tokens:
        try:
            token_id = tokenizer.encode(word, add_special_tokens=False)[0]
            time_vec = time_emb[token_id]
            norm = np.linalg.norm(time_vec)

            print(f"'{word}' (ID: {token_id}):")
            print(f"  Norm: {norm:.4f}")
            print(f"  Top 3 dimensions: {time_vec[top_dims[:3]]}")
        except:
            print(f"'{word}': Not found")

        print()

    # Correlation with usage counts (if available)
    if hasattr(model.tokenizer.time_embeddings, 'usage_counts'):
        usage_counts = model.tokenizer.time_embeddings.usage_counts.cpu().numpy()
        print("="*80)
        print("USAGE COUNT CORRELATION")
        print("="*80 + "\n")

        # Compute correlation between time embedding norm and usage
        norms = np.linalg.norm(time_emb, axis=1)
        valid_mask = usage_counts > 0
        if valid_mask.sum() > 10:
            corr = np.corrcoef(norms[valid_mask], usage_counts[valid_mask])[0, 1]
            print(f"Correlation between time embedding norm and usage: {corr:.4f}")
            if abs(corr) > 0.5:
                print("  → Strong correlation! Time tracks usage frequency.")
            elif abs(corr) > 0.3:
                print("  → Moderate correlation.")
            else:
                print("  → Weak correlation. Time learned something else.")
        print()

    # Save analysis
    print("="*80)
    print("SAVING RESULTS")
    print("="*80 + "\n")

    with open("output.txt", 'a') as f:
        f.write("\n" + "="*80 + "\n")
        f.write("TIME EMBEDDING PATTERN ANALYSIS\n")
        f.write("="*80 + "\n\n")

        f.write(f"Time embedding shape: {time_emb.shape}\n")
        f.write(f"Mean norm: {mean_norm:.4f} ± {std_norm:.4f}\n\n")

        f.write("Top 5 most active dimensions:\n")
        for rank, dim in enumerate(top_dims, 1):
            f.write(f"  {rank}. Dim {dim}: variance={dim_vars[dim]:.6f}\n")

        f.write("\nSample tokens:\n")
        for word in sample_tokens[:5]:
            try:
                token_id = tokenizer.encode(word, add_special_tokens=False)[0]
                norm = np.linalg.norm(time_emb[token_id])
                f.write(f"  '{word}': norm={norm:.4f}\n")
            except:
                pass

        f.write("\n")

    # Try to create visualization (if matplotlib works)
    try:
        print("Creating visualization...")

        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Plot 1: Distribution of time embedding norms
        norms = np.linalg.norm(time_emb, axis=1)
        axes[0, 0].hist(norms, bins=50, alpha=0.7, color='blue')
        axes[0, 0].set_xlabel('Time Embedding Norm')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Distribution of Time Embedding Norms')
        axes[0, 0].grid(True, alpha=0.3)

        # Plot 2: Dimension variance
        axes[0, 1].bar(range(min(20, time_dim)), dim_vars[:20])
        axes[0, 1].set_xlabel('Dimension')
        axes[0, 1].set_ylabel('Variance')
        axes[0, 1].set_title('Time Embedding Dimension Variance (Top 20)')
        axes[0, 1].grid(True, alpha=0.3)

        # Plot 3: Scatter of first two principal dimensions
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        time_2d = pca.fit_transform(time_emb[:1000])  # Sample for speed
        axes[1, 0].scatter(time_2d[:, 0], time_2d[:, 1], alpha=0.3, s=1)
        axes[1, 0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
        axes[1, 0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
        axes[1, 0].set_title('Time Embeddings (PCA)')
        axes[1, 0].grid(True, alpha=0.3)

        # Plot 4: Heatmap of sample tokens
        sample_indices = [tokenizer.encode(w, add_special_tokens=False)[0]
                         for w in sample_tokens[:10] if tokenizer.encode(w, add_special_tokens=False)]
        if sample_indices:
            sample_embeddings = time_emb[sample_indices, :20]  # First 20 dims
            im = axes[1, 1].imshow(sample_embeddings, aspect='auto', cmap='coolwarm')
            axes[1, 1].set_xlabel('Time Dimension')
            axes[1, 1].set_ylabel('Token')
            axes[1, 1].set_yticks(range(len(sample_indices)))
            axes[1, 1].set_yticklabels([sample_tokens[i] for i in range(len(sample_indices))])
            axes[1, 1].set_title('Time Embeddings Heatmap (Sample Tokens)')
            plt.colorbar(im, ax=axes[1, 1])

        plt.tight_layout()
        plt.savefig('time_embedding_analysis.png', dpi=150, bbox_inches='tight')
        print("✓ Visualization saved to: time_embedding_analysis.png\n")

    except Exception as e:
        print(f"⚠️  Could not create visualization: {e}\n")

    print("✅ Analysis complete!")
    print("Results appended to output.txt\n")


if __name__ == "__main__":
    try:
        analyze_time_embeddings()
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
