"""
Training script for TEMPORAL and Baseline transformers.
Trains both models and logs metrics for comparison.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import os
import json
import numpy as np
from tqdm import tqdm
import argparse

from config import Config
from model import TemporalTransformer, BaselineTransformer, count_parameters


class TextDataset(Dataset):
    """Simple text dataset for language modeling"""

    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return max(0, len(self.data) - self.seq_length)

    def __getitem__(self, idx):
        chunk = self.data[idx:idx + self.seq_length + 1]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_wikitext_data(config):
    """
    Load and prepare WikiText-2 dataset.
    For the prototype, we'll create synthetic data if datasets library not available.
    """
    try:
        from datasets import load_dataset

        # Load WikiText-2
        dataset = load_dataset(config.dataset_name, config.dataset_config)

        # Build vocabulary from training data
        train_text = " ".join(dataset['train']['text'])
        vocab = build_vocab(train_text, config.vocab_size)

        # Tokenize datasets
        train_tokens = tokenize_text(dataset['train']['text'], vocab)
        val_tokens = tokenize_text(dataset['validation']['text'], vocab)
        test_tokens = tokenize_text(dataset['test']['text'], vocab)

        print(f"Loaded WikiText-2: Train={len(train_tokens)}, Val={len(val_tokens)}, Test={len(test_tokens)}")
        print(f"Vocabulary size: {len(vocab)}")

        return train_tokens, val_tokens, test_tokens, vocab

    except Exception as e:
        print(f"Could not load WikiText-2: {e}")
        print("Generating synthetic data for testing...")
        return generate_synthetic_data(config)


def build_vocab(text, max_vocab_size):
    """Build vocabulary from text"""
    # Simple character-level or word-level tokenization
    words = text.split()
    word_counts = {}
    for word in words:
        word_counts[word] = word_counts.get(word, 0) + 1

    # Sort by frequency and take top max_vocab_size
    sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
    vocab = {'<PAD>': 0, '<UNK>': 1, '<BOS>': 2, '<EOS>': 3}

    for word, _ in sorted_words[:max_vocab_size - 4]:
        vocab[word] = len(vocab)

    return vocab


def tokenize_text(text_list, vocab):
    """Convert text to token IDs"""
    tokens = []
    for text in text_list:
        words = text.split()
        for word in words:
            tokens.append(vocab.get(word, vocab['<UNK>']))
    return tokens


def generate_synthetic_data(config):
    """Generate synthetic data for testing"""
    print("Generating synthetic data...")

    # Create synthetic sequences
    np.random.seed(config.seed)
    train_tokens = np.random.randint(0, config.vocab_size, size=50000).tolist()
    val_tokens = np.random.randint(0, config.vocab_size, size=5000).tolist()
    test_tokens = np.random.randint(0, config.vocab_size, size=5000).tolist()

    vocab = {f'token_{i}': i for i in range(config.vocab_size)}

    return train_tokens, val_tokens, test_tokens, vocab


class Trainer:
    """Trainer class for both TEMPORAL and Baseline models"""

    def __init__(self, model, config, train_dataset, val_dataset, model_name="model"):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.model_name = model_name

        # Setup device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )

        # Setup learning rate scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config.num_epochs * len(train_dataset) // config.batch_size
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Logging
        self.train_losses = []
        self.val_losses = []
        self.val_perplexities = []
        self.time_stats_history = []

        # Create checkpoint directory
        os.makedirs(config.checkpoint_dir, exist_ok=True)
        os.makedirs(config.log_dir, exist_ok=True)

        self.global_step = 0

    def train_epoch(self, epoch):
        """Train for one epoch"""
        self.model.train()
        epoch_loss = 0
        num_batches = 0

        # Create dataloader
        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            drop_last=True
        )

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{self.config.num_epochs}")

        for batch_idx, (x, y) in enumerate(pbar):
            x, y = x.to(self.device), y.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()

            if isinstance(self.model, TemporalTransformer):
                logits = self.model(x, update_time=True)
            else:
                logits = self.model(x)

            # Compute loss
            loss = self.criterion(logits.view(-1, self.config.vocab_size), y.view(-1))

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)

            self.optimizer.step()
            self.scheduler.step()

            # Logging
            epoch_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            pbar.set_postfix({'loss': loss.item()})

            # Periodic logging
            if self.global_step % self.config.log_interval == 0:
                self.train_losses.append({
                    'step': self.global_step,
                    'loss': loss.item()
                })

            # Periodic evaluation
            if self.global_step % self.config.eval_interval == 0:
                val_loss, val_ppl = self.evaluate()
                print(f"\nStep {self.global_step}: Val Loss={val_loss:.4f}, Val PPL={val_ppl:.4f}")
                self.model.train()

            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self.save_checkpoint(epoch)

        avg_loss = epoch_loss / num_batches
        return avg_loss

    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0

        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False
        )

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(self.device), y.to(self.device)

                if isinstance(self.model, TemporalTransformer):
                    logits = self.model(x, update_time=False)
                else:
                    logits = self.model(x)

                loss = self.criterion(logits.view(-1, self.config.vocab_size), y.view(-1))
                total_loss += loss.item()
                num_batches += 1

        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)

        self.val_losses.append({'step': self.global_step, 'loss': avg_loss})
        self.val_perplexities.append({'step': self.global_step, 'perplexity': perplexity})

        return avg_loss, perplexity

    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print(f"Training {self.model_name}")
        print(f"Parameters: {count_parameters(self.model):,}")
        print(f"{'='*60}\n")

        for epoch in range(self.config.num_epochs):
            # Train epoch
            train_loss = self.train_epoch(epoch)

            # Evaluate
            val_loss, val_ppl = self.evaluate()

            print(f"\nEpoch {epoch+1}/{self.config.num_epochs}:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss: {val_loss:.4f}")
            print(f"  Val Perplexity: {val_ppl:.4f}")

            # Log time statistics for TEMPORAL model
            if isinstance(self.model, TemporalTransformer):
                time_stats = self.model.get_time_statistics()
                print(f"  Mean Time Magnitude: {time_stats['mean_time_magnitude']:.4f}")
                print(f"  Max Time Magnitude: {time_stats['max_time_magnitude']:.4f}")
                self.time_stats_history.append({
                    'epoch': epoch,
                    'stats': time_stats
                })

        # Final evaluation
        final_val_loss, final_val_ppl = self.evaluate()
        print(f"\n{'='*60}")
        print(f"Final Results for {self.model_name}:")
        print(f"  Val Loss: {final_val_loss:.4f}")
        print(f"  Val Perplexity: {final_val_ppl:.4f}")
        print(f"{'='*60}\n")

        # Save final checkpoint
        self.save_checkpoint('final')

        # Save training logs
        self.save_logs()

        return final_val_loss, final_val_ppl

    def save_checkpoint(self, epoch):
        """Save model checkpoint"""
        checkpoint_path = os.path.join(
            self.config.checkpoint_dir,
            f"{self.model_name}_epoch_{epoch}.pt"
        )

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'config': self.config.__dict__
        }

        torch.save(checkpoint, checkpoint_path)
        print(f"Saved checkpoint: {checkpoint_path}")

    def save_logs(self):
        """Save training logs"""
        log_path = os.path.join(self.config.log_dir, f"{self.model_name}_logs.json")

        logs = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'val_perplexities': self.val_perplexities,
            'time_stats_history': self.time_stats_history if isinstance(self.model, TemporalTransformer) else None
        }

        with open(log_path, 'w') as f:
            json.dump(logs, f, indent=2, default=lambda x: x.tolist() if isinstance(x, np.ndarray) else x)

        print(f"Saved logs: {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train TEMPORAL and Baseline models")
    parser.add_argument('--model', type=str, choices=['temporal', 'baseline', 'both'], default='both',
                        help='Which model to train')
    args = parser.parse_args()

    # Load configuration
    config = Config()

    # Set random seeds
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Load data
    train_tokens, val_tokens, test_tokens, vocab = load_wikitext_data(config)

    # Create datasets
    train_dataset = TextDataset(train_tokens, config.max_seq_length)
    val_dataset = TextDataset(val_tokens, config.max_seq_length)

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Val dataset size: {len(val_dataset)}")

    results = {}

    # Train TEMPORAL model
    if args.model in ['temporal', 'both']:
        print("\n" + "="*60)
        print("TRAINING TEMPORAL MODEL")
        print("="*60)

        temporal_model = TemporalTransformer(config)
        temporal_trainer = Trainer(temporal_model, config, train_dataset, val_dataset, "temporal")
        temporal_val_loss, temporal_val_ppl = temporal_trainer.train()

        results['temporal'] = {
            'val_loss': temporal_val_loss,
            'val_perplexity': temporal_val_ppl
        }

    # Train Baseline model
    if args.model in ['baseline', 'both']:
        print("\n" + "="*60)
        print("TRAINING BASELINE MODEL")
        print("="*60)

        baseline_model = BaselineTransformer(config)
        baseline_trainer = Trainer(baseline_model, config, train_dataset, val_dataset, "baseline")
        baseline_val_loss, baseline_val_ppl = baseline_trainer.train()

        results['baseline'] = {
            'val_loss': baseline_val_loss,
            'val_perplexity': baseline_val_ppl
        }

    # Print comparison if both models trained
    if args.model == 'both':
        print("\n" + "="*60)
        print("FINAL COMPARISON")
        print("="*60)
        print(f"TEMPORAL - Val Loss: {results['temporal']['val_loss']:.4f}, Val PPL: {results['temporal']['val_perplexity']:.4f}")
        print(f"BASELINE - Val Loss: {results['baseline']['val_loss']:.4f}, Val PPL: {results['baseline']['val_perplexity']:.4f}")

        if results['temporal']['val_perplexity'] < results['baseline']['val_perplexity']:
            improvement = ((results['baseline']['val_perplexity'] - results['temporal']['val_perplexity']) /
                          results['baseline']['val_perplexity'] * 100)
            print(f"\nTEMPORAL is BETTER by {improvement:.2f}% in perplexity!")
        else:
            difference = ((results['temporal']['val_perplexity'] - results['baseline']['val_perplexity']) /
                         results['baseline']['val_perplexity'] * 100)
            print(f"\nBASELINE is better by {difference:.2f}% in perplexity")

    print("\nTraining complete!")


if __name__ == "__main__":
    main()
