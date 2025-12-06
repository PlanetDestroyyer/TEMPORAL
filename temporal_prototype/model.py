"""
TEMPORAL Transformer Architecture
Implements time-aware attention and experiential learning through time embeddings.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from time_embeddings import TimeEmbeddedTokenizer


class TimeAwareAttention(nn.Module):
    """
    Multi-head attention mechanism that operates on time-embedded tokens.
    Attention sees both WHAT the token is (content) and how EXPERIENCED it is (time).
    """

    def __init__(self, embed_dim, n_heads, dropout=0.1):
        super().__init__()
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"

        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        self.scale = math.sqrt(self.head_dim)

        # Query, Key, Value projections operate on full embed_dim (content + time)
        self.qkv_proj = nn.Linear(embed_dim, 3 * embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
               embed_dim includes both content and time dimensions
            mask: Optional attention mask

        Returns:
            output: Tensor of shape (batch_size, seq_len, embed_dim)
            attention_weights: Tensor of shape (batch_size, n_heads, seq_len, seq_len)
        """
        batch_size, seq_len, embed_dim = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (batch_size, seq_len, 3 * embed_dim)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch_size, n_heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.scale
        # (batch_size, n_heads, seq_len, seq_len)

        # Apply mask if provided (for causal attention)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        out = torch.matmul(attn_weights, v)  # (batch_size, n_heads, seq_len, head_dim)
        out = out.permute(0, 2, 1, 3).contiguous()  # (batch_size, seq_len, n_heads, head_dim)
        out = out.reshape(batch_size, seq_len, embed_dim)

        # Output projection
        out = self.out_proj(out)

        return out, attn_weights


class TransformerBlock(nn.Module):
    """
    Transformer block with time-aware attention.
    """

    def __init__(self, embed_dim, n_heads, ff_dim, dropout=0.1):
        super().__init__()

        self.attention = TimeAwareAttention(embed_dim, n_heads, dropout)
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        self.feed_forward = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, embed_dim)
            mask: Optional attention mask

        Returns:
            output: Tensor of shape (batch_size, seq_len, embed_dim)
        """
        # Self-attention with residual connection
        attn_out, attn_weights = self.attention(self.norm1(x), mask)
        x = x + attn_out

        # Feed-forward with residual connection
        ff_out = self.feed_forward(self.norm2(x))
        x = x + ff_out

        return x


class TemporalTransformer(nn.Module):
    """
    TEMPORAL Transformer: Language model with time-embedded tokens.
    Tokens accumulate experience through usage, enabling experiential learning.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Time-embedded tokenizer
        self.tokenizer = TimeEmbeddedTokenizer(
            vocab_size=config.vocab_size,
            content_dim=config.content_dim,
            time_dim=config.time_dim,
            time_lr=config.time_lr
        )

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, config.max_seq_length, config.total_dim)
        )

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.total_dim,
                n_heads=config.n_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout
            ) for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.total_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Output projection to vocabulary
        self.output_proj = nn.Linear(config.total_dim, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        # Initialize positional encodings
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)

        # Initialize output projection
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, input_ids, update_time=True, return_confidence=False):
        """
        Forward pass through TEMPORAL transformer.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            update_time: Whether to update time embeddings
            return_confidence: Whether to return prediction confidence

        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            confidence: Optional tensor of confidence scores
        """
        batch_size, seq_len = input_ids.shape

        # Get time-embedded tokens [content | time]
        x = self.tokenizer(input_ids, update_time=update_time)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)
        # (1, 1, seq_len, seq_len)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)

        if return_confidence:
            # Compute confidence as max probability
            probs = F.softmax(logits, dim=-1)
            confidence, _ = probs.max(dim=-1)
            return logits, confidence

        return logits

    def get_time_statistics(self):
        """Get time embedding statistics for analysis"""
        return self.tokenizer.get_time_statistics()

    def reset_time_embeddings(self):
        """Reset time embeddings to zero"""
        self.tokenizer.reset_time_embeddings()


class BaselineTransformer(nn.Module):
    """
    Baseline transformer without time embeddings.
    Uses standard 256-dimensional content embeddings for comparison.
    """

    def __init__(self, config):
        super().__init__()
        self.config = config

        # Standard token embeddings (no time component)
        self.token_embeddings = nn.Embedding(config.vocab_size, config.total_dim)

        # Positional encoding
        self.pos_encoding = nn.Parameter(
            torch.zeros(1, config.max_seq_length, config.total_dim)
        )

        # Transformer blocks (identical architecture to TEMPORAL)
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=config.total_dim,
                n_heads=config.n_heads,
                ff_dim=config.ff_dim,
                dropout=config.dropout
            ) for _ in range(config.n_layers)
        ])

        self.norm = nn.LayerNorm(config.total_dim)
        self.dropout = nn.Dropout(config.dropout)

        # Output projection
        self.output_proj = nn.Linear(config.total_dim, config.vocab_size)

        self._init_weights()

    def _init_weights(self):
        """Initialize weights"""
        nn.init.normal_(self.token_embeddings.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.pos_encoding, mean=0.0, std=0.02)
        nn.init.normal_(self.output_proj.weight, mean=0.0, std=0.02)
        if self.output_proj.bias is not None:
            nn.init.zeros_(self.output_proj.bias)

    def forward(self, input_ids, return_confidence=False):
        """
        Forward pass through baseline transformer.

        Args:
            input_ids: Tensor of shape (batch_size, seq_len)
            return_confidence: Whether to return prediction confidence

        Returns:
            logits: Tensor of shape (batch_size, seq_len, vocab_size)
            confidence: Optional tensor of confidence scores
        """
        batch_size, seq_len = input_ids.shape

        # Get token embeddings
        x = self.token_embeddings(input_ids)

        # Add positional encoding
        x = x + self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x)

        # Create causal mask
        mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device)).unsqueeze(0).unsqueeze(0)

        # Pass through transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        # Project to vocabulary
        logits = self.output_proj(x)

        if return_confidence:
            probs = F.softmax(logits, dim=-1)
            confidence, _ = probs.max(dim=-1)
            return logits, confidence

        return logits


def count_parameters(model):
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
