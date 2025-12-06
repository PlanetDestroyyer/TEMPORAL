"""
Configuration file for TEMPORAL prototype.
Contains hyperparameters for model architecture, training, and evaluation.
"""

class Config:
    # Model Architecture
    vocab_size = 1000  # Small vocabulary for prototype
    content_dim = 128  # Content embedding dimension
    time_dim = 128     # Time embedding dimension (parallel to content)
    total_dim = 256    # Total token representation [128 content | 128 time]

    n_layers = 2       # Number of transformer layers
    n_heads = 4        # Number of attention heads
    ff_dim = 512       # Feedforward dimension
    dropout = 0.1      # Dropout rate
    max_seq_length = 128  # Maximum sequence length

    # Time Embedding Update Parameters
    time_lr = 0.01     # Learning rate for time embedding updates
    time_decay = 0.0   # Optional: decay rate for unused tokens (0 = no decay)

    # Time embedding dimensions allocation:
    # dim 0: Usage count increment
    # dim 1: Recency score
    # dim 2: Context diversity score
    # dim 3: Prediction confidence
    # dim 4-127: Learned through gradient on prediction accuracy

    # Training Parameters
    batch_size = 32
    num_epochs = 10
    learning_rate = 0.0003  # Learning rate for content embeddings and model params
    weight_decay = 0.01
    grad_clip = 1.0
    warmup_steps = 100

    # Dataset
    dataset_name = "wikitext"
    dataset_config = "wikitext-2-raw-v1"

    # Logging and Checkpointing
    log_interval = 100  # Log every N steps
    eval_interval = 500  # Evaluate every N steps
    save_interval = 1000  # Save checkpoint every N steps

    # Visualization
    track_tokens = [10, 50, 100, 200, 500]  # Specific tokens to track time evolution

    # Paths
    checkpoint_dir = "checkpoints"
    log_dir = "logs"
    output_dir = "outputs"

    # Device
    device = "cuda" # Will be set to "cpu" if CUDA not available

    # Random seed
    seed = 42

    @classmethod
    def get_baseline_config(cls):
        """Returns config for baseline model (no time embeddings)"""
        config = cls()
        config.use_time_embeddings = False
        config.total_dim = 256  # All 256 dims for content only
        return config

    @classmethod
    def get_temporal_config(cls):
        """Returns config for TEMPORAL model (with time embeddings)"""
        config = cls()
        config.use_time_embeddings = True
        return config
