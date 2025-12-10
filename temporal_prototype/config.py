"""
Production Configuration for TEMPORAL
SOTA settings for research-grade experiments
"""

class ProductionConfig:
    """Production/Research-grade configuration"""

    # ============================================================================
    # MODEL ARCHITECTURE (SOTA)
    # ============================================================================

    # Vocabulary & Embeddings
    vocab_size = 50257  # GPT-2 tokenizer vocab size
    content_dim = 384   # Content embedding dimension
    time_dim = 384      # Time embedding dimension
    total_dim = 768     # Total: 384 content + 384 time

    # Transformer Architecture
    n_layers = 12       # Standard for small models (GPT-2 small: 12)
    n_heads = 12        # Should divide total_dim evenly
    ff_dim = 3072       # 4x total_dim (standard practice)
    dropout = 0.1       # Standard dropout
    max_seq_length = 1024  # Context window

    # Model size: ~125M parameters (comparable to GPT-2 small)

    # ============================================================================
    # TIME EMBEDDING SETTINGS
    # ============================================================================

    time_learning_mode = 'gradient'  # 'gradient' or 'hybrid'
    # 'gradient': Pure self-learning through backprop (recommended)
    # 'hybrid': Gradient + optional inference-time updates

    # ============================================================================
    # DATASET (SOTA)
    # ============================================================================

    # Primary dataset
    dataset_name = "wikitext"
    dataset_config = "wikitext-103-raw-v1"  # Larger than wikitext-2
    # Alternative options:
    # - "EleutherAI/pile" (subset)
    # - "allenai/c4" (en)
    # - "togethercomputer/RedPajama-Data-1T-Sample"

    # Tokenizer
    tokenizer_name = "gpt2"  # or "EleutherAI/gpt-neo-125M"

    # Preprocessing
    block_size = 1024  # Sequence length for training
    preprocessing_num_workers = 4

    # ============================================================================
    # TRAINING (Production-Grade)
    # ============================================================================

    # Batch settings
    batch_size = 8      # Per-device batch size
    gradient_accumulation_steps = 16  # Effective batch = 8 * 16 = 128
    effective_batch_size = batch_size * gradient_accumulation_steps

    # Training duration
    num_epochs = 3      # Standard for large datasets
    max_steps = None    # Or set max steps instead of epochs

    # Optimization (SOTA)
    learning_rate = 3e-4        # Standard for transformers
    weight_decay = 0.1          # AdamW weight decay
    adam_beta1 = 0.9            # Adam beta1
    adam_beta2 = 0.95           # Adam beta2 (LLaMA value)
    adam_epsilon = 1e-8         # Adam epsilon
    max_grad_norm = 1.0         # Gradient clipping

    # Learning rate schedule
    lr_scheduler_type = "cosine"  # cosine, linear, constant
    warmup_ratio = 0.1            # 10% warmup
    warmup_steps = None           # Or set explicit warmup steps

    # Mixed precision
    fp16 = False        # FP16 training (if supported)
    bf16 = True         # BF16 training (better for modern GPUs)

    # ============================================================================
    # EVALUATION
    # ============================================================================

    eval_strategy = "steps"     # "steps" or "epoch"
    eval_steps = 500            # Evaluate every N steps
    eval_batch_size = 16        # Larger batch for evaluation

    # Metrics
    metric_for_best_model = "eval_loss"
    greater_is_better = False

    # ============================================================================
    # LOGGING & CHECKPOINTING
    # ============================================================================

    # Logging
    logging_steps = 100
    logging_first_step = True
    log_level = "info"

    # Checkpointing
    save_strategy = "steps"
    save_steps = 1000
    save_total_limit = 3        # Keep only 3 best checkpoints

    # Output directories
    output_dir = "outputs/temporal_production"
    logging_dir = "logs/temporal_production"
    checkpoint_dir = "checkpoints/temporal_production"

    # ============================================================================
    # EXPERIMENT TRACKING
    # ============================================================================

    # Weights & Biases (disabled by default to avoid interactive prompts)
    use_wandb = False
    wandb_project = "temporal-research"
    wandb_entity = None  # Your wandb username/team
    wandb_api_key = None  # Set to your API key to enable: "0437ee052ff28e5c4dfd888f85c623eafb3857c0"
    run_name = None      # Auto-generated if None

    # ============================================================================
    # HARDWARE & PERFORMANCE
    # ============================================================================

    # Device
    device = "cuda"  # Will auto-detect
    use_cpu = False

    # Multi-GPU
    local_rank = -1              # For distributed training
    ddp_find_unused_parameters = False

    # Performance optimizations
    dataloader_num_workers = 4
    dataloader_pin_memory = True
    torch_compile = False        # PyTorch 2.0 compile (experimental)

    # ============================================================================
    # REPRODUCIBILITY
    # ============================================================================

    seed = 42
    deterministic = True

    # ============================================================================
    # DEBUGGING
    # ============================================================================

    debug = False
    max_train_samples = None  # Limit training samples for debugging
    max_eval_samples = None   # Limit eval samples for debugging


class ColabConfig(ProductionConfig):
    """Optimized for Google Colab/Kaggle (T4/P100 GPU)"""

    # Moderate model size for free-tier GPUs
    content_dim = 256
    time_dim = 256
    total_dim = 512
    n_layers = 6
    n_heads = 8
    ff_dim = 2048

    # Smaller batches
    batch_size = 4
    gradient_accumulation_steps = 8
    max_seq_length = 512
    block_size = 512  # Match max_seq_length for consistency

    # Quick validation runs
    num_epochs = 2

    # Dataset
    dataset_config = "wikitext-2-raw-v1"  # Smaller dataset

    # Performance (reduce workers for Colab/Kaggle)
    dataloader_num_workers = 2
    preprocessing_num_workers = 2

    # Model size: ~76M parameters


class ScaledConfig(ProductionConfig):
    """Scaled-up configuration for comprehensive validation

    This config is designed to show stronger TEMPORAL advantages:
    - Larger model (more capacity to learn time patterns)
    - More training (time embeddings need experience to learn)
    - Limited dataset size for reasonable training time

    Suitable for: Kaggle P100, Colab Pro, or any GPU with 16GB+ VRAM
    Training time: ~4-6 hours on P100, ~6-10 hours on T4
    """

    # Larger model architecture
    content_dim = 384       # +50% vs Colab
    time_dim = 384          # +50% vs Colab
    total_dim = 768         # GPT-2 small size
    n_layers = 12           # 2x vs Colab (standard for GPT-2 small)
    n_heads = 12            # Matches total_dim
    ff_dim = 3072           # 4x total_dim (standard)

    # Training batches
    batch_size = 4          # Same as Colab (memory constraint)
    gradient_accumulation_steps = 16  # 2x vs Colab = effective batch 64
    max_seq_length = 1024   # Longer context
    block_size = 1024

    # MORE TRAINING - Critical for time embeddings to learn!
    num_epochs = 3          # 1.5x vs Colab (balanced for time)

    # Filtered WikiText-103 dataset (sentences with 10-30 tokens)
    # LIMITED to 100k samples for reasonable training time
    dataset_name = "carlosejimenez/wikitext-103-raw-v1_sents_min_len10_max_len30"
    dataset_config = None  # Custom dataset doesn't need config

    # LIMIT dataset size for reasonable training time
    # 100k samples * 3 epochs = 300k sample-epochs (4x colab's 72k)
    max_train_samples = 100_000  # 3x WikiText-2, reasonable training time
    max_eval_samples = 10_000    # 10% for validation

    # Evaluation
    eval_steps = 500        # Adjust based on dataset size
    save_steps = 1000

    # Performance
    dataloader_num_workers = 2
    preprocessing_num_workers = 4

    # Model size: ~355M parameters (GPT-2 small scale)
    # Training time estimate: ~10-12 hours on P100


class FastDebugConfig(ProductionConfig):
    """Quick debugging configuration"""

    # Tiny model
    content_dim = 128
    time_dim = 128
    total_dim = 256
    n_layers = 2
    n_heads = 4
    ff_dim = 512

    # Tiny dataset
    dataset_config = "wikitext-2-raw-v1"
    max_train_samples = 1000
    max_eval_samples = 100

    # Fast training
    num_epochs = 1
    batch_size = 2
    gradient_accumulation_steps = 2
    eval_steps = 50
    save_steps = 100

    # Debug mode
    debug = True


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_config(config_name="production"):
    """Get configuration by name"""
    configs = {
        "production": ProductionConfig(),
        "colab": ColabConfig(),
        "scaled": ScaledConfig(),
        "debug": FastDebugConfig(),
    }
    return configs.get(config_name, ProductionConfig())


def print_config(config):
    """Print configuration summary"""
    print("\n" + "="*70)
    print("CONFIGURATION SUMMARY")
    print("="*70)

    print(f"\nModel Architecture:")
    print(f"  Vocab Size: {config.vocab_size:,}")
    print(f"  Content Dim: {config.content_dim}")
    print(f"  Time Dim: {config.time_dim}")
    print(f"  Total Dim: {config.total_dim}")
    print(f"  Layers: {config.n_layers}")
    print(f"  Heads: {config.n_heads}")
    print(f"  FFN Dim: {config.ff_dim}")

    # Estimate parameters
    params = estimate_parameters(config)
    print(f"  Estimated Parameters: {params/1e6:.1f}M")

    print(f"\nDataset:")
    print(f"  Name: {config.dataset_name}")
    print(f"  Config: {config.dataset_config}")
    print(f"  Block Size: {config.block_size}")

    print(f"\nTraining:")
    print(f"  Batch Size (per device): {config.batch_size}")
    print(f"  Gradient Accumulation: {config.gradient_accumulation_steps}")
    print(f"  Effective Batch Size: {config.effective_batch_size}")
    print(f"  Learning Rate: {config.learning_rate}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Mixed Precision: FP16={config.fp16}, BF16={config.bf16}")

    print(f"\nTime Learning:")
    print(f"  Mode: {config.time_learning_mode}")

    print("="*70 + "\n")


def estimate_parameters(config):
    """Estimate total model parameters"""
    # Embeddings
    content_emb = config.vocab_size * config.content_dim
    time_emb = config.vocab_size * config.time_dim

    # Each transformer layer
    # Attention: 4 * d^2 (QKV + output projection)
    # FFN: 2 * d * ff_dim
    layer_params = (4 * config.total_dim ** 2) + (2 * config.total_dim * config.ff_dim)
    total_layers = config.n_layers * layer_params

    # Output projection
    output_proj = config.total_dim * config.vocab_size

    total = content_emb + time_emb + total_layers + output_proj
    return total
