"""
Time Embedding Layer for TEMPORAL architecture.
Manages mutable time embeddings that update based on token usage.
"""

import torch
import torch.nn as nn
import numpy as np
from collections import defaultdict


class TimeEmbeddings(nn.Module):
    """
    Time embedding layer that maintains and updates time values for each token.
    Time embeddings start at zero and increase with usage during both training and inference.
    """

    def __init__(self, vocab_size, time_dim, time_lr=0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.time_dim = time_dim
        self.time_lr = time_lr

        # Initialize time embeddings to zero (learnable but updated via custom mechanism)
        self.time_embeddings = nn.Parameter(
            torch.zeros(vocab_size, time_dim),
            requires_grad=True
        )

        # Statistics tracking (not part of model parameters)
        self.register_buffer('usage_counts', torch.zeros(vocab_size))
        self.register_buffer('last_used_step', torch.zeros(vocab_size))
        self.register_buffer('context_diversity', torch.zeros(vocab_size))

        self.current_step = 0

    def forward(self, token_ids):
        """
        Get time embeddings for given token IDs.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len)

        Returns:
            time_emb: Tensor of shape (batch_size, seq_len, time_dim)
        """
        return self.time_embeddings[token_ids]

    def compute_update_vector(self, token_ids, contexts=None, confidence=None):
        """
        Compute the update vector for time embeddings based on usage.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len)
            contexts: Optional context representations for diversity calculation
            confidence: Optional prediction confidence scores

        Returns:
            update_vectors: Tensor of shape (batch_size, seq_len, time_dim)
        """
        batch_size, seq_len = token_ids.shape
        device = token_ids.device

        # Initialize update vector
        update_vectors = torch.zeros(batch_size, seq_len, self.time_dim, device=device)

        # Dimension 0: Usage count increment (constant increment)
        update_vectors[:, :, 0] = 1.0

        # Dimension 1: Recency score (higher for recent steps)
        # Compute recency as 1.0 / (steps_since_last_use + 1)
        for b in range(batch_size):
            for s in range(seq_len):
                token_id = token_ids[b, s].item()
                steps_since = self.current_step - self.last_used_step[token_id].item()
                recency = 1.0 / (steps_since + 1.0)
                update_vectors[b, s, 1] = recency

        # Dimension 2: Context diversity score
        # Simple version: increment if token appears in new context
        if contexts is not None:
            # contexts: (batch_size, seq_len, hidden_dim)
            for b in range(batch_size):
                for s in range(seq_len):
                    token_id = token_ids[b, s].item()
                    # Simplified: use a small increment to represent diversity
                    update_vectors[b, s, 2] = 0.1

        # Dimension 3: Prediction confidence
        if confidence is not None:
            update_vectors[:, :, 3] = confidence

        # Dimensions 4-127: Will be updated through gradient-based learning
        # These are learned automatically through backprop on the time embeddings
        # We don't manually set them here

        return update_vectors

    def update_time_embeddings(self, token_ids, contexts=None, confidence=None):
        """
        Update time embeddings based on token usage.
        This is called during forward pass to accumulate experience.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len)
            contexts: Optional context representations
            confidence: Optional prediction confidence scores
        """
        with torch.no_grad():
            # Compute update vectors
            update_vectors = self.compute_update_vector(token_ids, contexts, confidence)

            # Update time embeddings for each token
            batch_size, seq_len = token_ids.shape
            for b in range(batch_size):
                for s in range(seq_len):
                    token_id = token_ids[b, s].item()

                    # Update time embedding
                    self.time_embeddings.data[token_id] += self.time_lr * update_vectors[b, s]

                    # Update statistics
                    self.usage_counts[token_id] += 1
                    self.last_used_step[token_id] = self.current_step

            self.current_step += 1

    def get_time_statistics(self):
        """
        Get statistics about time embeddings for analysis.

        Returns:
            dict with various statistics
        """
        with torch.no_grad():
            time_magnitudes = torch.norm(self.time_embeddings, dim=1)

            return {
                'mean_time_magnitude': time_magnitudes.mean().item(),
                'max_time_magnitude': time_magnitudes.max().item(),
                'min_time_magnitude': time_magnitudes.min().item(),
                'usage_counts': self.usage_counts.cpu().numpy(),
                'time_values_dim0': self.time_embeddings[:, 0].cpu().numpy(),
                'time_values_dim1': self.time_embeddings[:, 1].cpu().numpy(),
                'time_magnitudes': time_magnitudes.cpu().numpy(),
            }

    def reset_time_embeddings(self):
        """Reset all time embeddings to zero (useful for experiments)"""
        with torch.no_grad():
            self.time_embeddings.zero_()
            self.usage_counts.zero_()
            self.last_used_step.zero_()
            self.context_diversity.zero_()
            self.current_step = 0


class TimeEmbeddedTokenizer(nn.Module):
    """
    Manages dual-component token representations: [content_embedding | time_embedding]
    """

    def __init__(self, vocab_size, content_dim, time_dim, time_lr=0.01):
        super().__init__()
        self.vocab_size = vocab_size
        self.content_dim = content_dim
        self.time_dim = time_dim
        self.total_dim = content_dim + time_dim

        # Content embeddings (learned normally via backprop)
        self.content_embeddings = nn.Embedding(vocab_size, content_dim)

        # Time embeddings (updated via usage-based mechanism)
        self.time_embeddings = TimeEmbeddings(vocab_size, time_dim, time_lr)

    def forward(self, token_ids, update_time=True, contexts=None, confidence=None):
        """
        Get dual-component embeddings for tokens.

        Args:
            token_ids: Tensor of shape (batch_size, seq_len)
            update_time: Whether to update time embeddings based on usage
            contexts: Optional context representations
            confidence: Optional confidence scores

        Returns:
            embeddings: Tensor of shape (batch_size, seq_len, content_dim + time_dim)
        """
        # Get content embeddings
        content_emb = self.content_embeddings(token_ids)

        # Get time embeddings
        time_emb = self.time_embeddings(token_ids)

        # Concatenate [content | time]
        embeddings = torch.cat([content_emb, time_emb], dim=-1)

        # Update time embeddings if requested (during training and inference)
        if update_time:
            self.time_embeddings.update_time_embeddings(token_ids, contexts, confidence)

        return embeddings

    def get_time_statistics(self):
        """Get time embedding statistics"""
        return self.time_embeddings.get_time_statistics()

    def reset_time_embeddings(self):
        """Reset time embeddings to zero"""
        self.time_embeddings.reset_time_embeddings()
