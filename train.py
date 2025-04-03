import json
import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import GPT2LMHeadModel, GPT2Tokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm


class ReverseGPT2Dataset(Dataset):
    """Dataset for training GPT-2 to predict prompts from outputs by reversing token sequences."""

    def __init__(self, file_path, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []

        # Load data from JSONL file
        with open(file_path, "r") as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example["text"])

        print(f"Loaded {len(self.examples)} examples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Tokenize the text
        tokens = self.tokenizer.encode(
            text, truncation=True, max_length=self.max_length
        )

        # Reverse the tokens
        reversed_tokens = tokens[::-1]

        # Convert to tensor
        input_ids = torch.tensor(reversed_tokens)

        return input_ids


def collate_batch(batch, pad_token_id):
    """Collate function for creating batches with padding."""
    # Find max length in batch
    max_length = max(len(seq) for seq in batch)

    # Create padded and masked tensors
    input_ids = []
    attention_mask = []
    labels = []

    for seq in batch:
        # Calculate padding length
        padding_length = max_length - len(seq)

        # Create padded sequence and attention mask
        padded_seq = torch.cat(
            [seq, torch.full((padding_length,), pad_token_id, dtype=torch.long)]
        )
        mask = torch.cat(
            [
                torch.ones(len(seq), dtype=torch.long),
                torch.zeros(padding_length, dtype=torch.long),
            ]
        )

        input_ids.append(padded_seq)
        attention_mask.append(mask)

        # For language modeling, labels are the same as inputs
        # but we use -100 for padding tokens to ignore them in the loss
        label = padded_seq.clone()
        label[mask == 0] = -100  # Set padding positions to -100 (ignored in loss)
        labels.append(label)

    # Stack all sequences into a batch
    return {
        "input_ids": torch.stack(input_ids),
        "attention_mask": torch.stack(attention_mask),
        "labels": torch.stack(labels),
    }


def train_model(
    model, train_dataloader, optimizer, scheduler, device, epochs, checkpoint_dir
):
    """Train the model with checkpointing."""
    # Create checkpoint directory if it doesn't exist
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)

    model.train()
    best_loss = float("inf")

    for epoch in range(epochs):
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            # Update metrics
            total_loss += loss.item()
            avg_loss = total_loss / (progress_bar.n + 1)
            progress_bar.set_postfix({"loss": avg_loss})

        # Calculate average loss for the epoch
        avg_epoch_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} completed. Average loss: {avg_epoch_loss}")

        # Save checkpoint after each epoch
        checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint-epoch-{epoch + 1}")
        model.save_pretrained(checkpoint_path)
        print(f"Checkpoint saved to {checkpoint_path}")

        # Save the best model so far (optional)
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_model_path = os.path.join(checkpoint_dir, "best_model")
            model.save_pretrained(best_model_path)
            print(f"New best model saved with loss: {best_loss}")


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Train GPT-2 to predict prompts from outputs by reversing tokens."
    )

    # Data and model arguments
    parser.add_argument(
        "--data_path",
        type=str,
        default="data/small-117M-k40.train.jsonl",
        help="Path to the JSONL file containing training data",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="gpt2",
        help="GPT-2 model size to use (gpt2, gpt2-medium, gpt2-large, gpt2-xl)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save model checkpoints",
    )
    parser.add_argument(
        "--final_model_dir",
        type=str,
        default="2tpg",
        help="Directory to save the final model",
    )

    # Training hyperparameters
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=3, help="Number of training epochs"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=5e-5, help="Learning rate"
    )
    parser.add_argument(
        "--warmup_steps",
        type=int,
        default=100,
        help="Number of warmup steps for learning rate scheduler",
    )

    # Other settings
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for reproducibility"
    )
    parser.add_argument(
        "--resume_from",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from",
    )

    args = parser.parse_args()

    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load tokenizer and model
    print(f"Loading {args.model_name} model...")

    if args.resume_from:
        print(f"Resuming from checkpoint: {args.resume_from}")
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        model = GPT2LMHeadModel.from_pretrained(args.resume_from)
    else:
        tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)
        model = GPT2LMHeadModel.from_pretrained(args.model_name)

    # Set padding token (GPT-2 doesn't have one by default)
    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    # Move model to device
    model = model.to(device)

    # Create dataset and dataloader
    print(f"Preparing dataset from {args.data_path}...")
    dataset = ReverseGPT2Dataset(args.data_path, tokenizer, max_length=args.max_length)

    # Create dataloader with custom collate function
    train_dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
    )

    # Setup optimizer and scheduler
    print("Setting up training...")
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    total_steps = len(train_dataloader) * args.epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=total_steps
    )

    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    train_model(
        model=model,
        train_dataloader=train_dataloader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        checkpoint_dir=args.output_dir,
    )

    # Save the final model
    Path(args.final_model_dir).mkdir(parents=True, exist_ok=True)
    print(f"Saving final model to {args.final_model_dir}...")
    model.save_pretrained(args.final_model_dir)
    tokenizer.save_pretrained(args.final_model_dir)
    print("Training complete!")


if __name__ == "__main__":
    main()
