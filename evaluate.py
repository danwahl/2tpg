import json
import argparse
import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer


class ReverseEvalDataset(Dataset):
    """Dataset for evaluating reverse token prediction."""

    def __init__(self, file_path, tokenizer, max_length=512, reverse=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.examples = []
        self.reverse = reverse

        # Load data from JSONL file
        with open(file_path, "r") as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example["text"])

        print(f"Loaded {len(self.examples)} evaluation examples from {file_path}")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        text = self.examples[idx]

        # Tokenize the text
        tokens = self.tokenizer.encode(
            text, truncation=True, max_length=self.max_length
        )

        # Reverse the tokens for reverse prediction (can be disabled for normal evaluation)
        if self.reverse:
            tokens = tokens[::-1]

        # Convert to tensor
        input_ids = torch.tensor(tokens)

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


def evaluate_model(model, eval_dataloader, device):
    """Evaluate the model on a dataset."""
    model.eval()
    total_loss = 0
    total_tokens = 0

    with torch.no_grad():
        progress_bar = tqdm(eval_dataloader, desc="Evaluating")
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            # Forward pass
            outputs = model(
                input_ids=input_ids, attention_mask=attention_mask, labels=labels
            )
            loss = outputs.loss

            # Count non-padding tokens
            non_padding = (labels != -100).sum().item()

            # Add to totals
            total_loss += loss.item() * non_padding
            total_tokens += non_padding

            # Update progress bar
            avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
            perplexity = np.exp(avg_loss)
            progress_bar.set_postfix({"loss": avg_loss, "ppl": perplexity})

    # Calculate final metrics
    avg_loss = total_loss / total_tokens if total_tokens > 0 else 0
    perplexity = np.exp(avg_loss)

    return {"loss": avg_loss, "perplexity": perplexity, "total_tokens": total_tokens}


def evaluate_examples(model, tokenizer, device, examples, max_length=50, reverse=True):
    """Generate predictions for a few examples to qualitatively evaluate the model."""
    model.eval()
    results = []

    for example in examples:
        # Tokenize the input
        tokens = tokenizer.encode(example, return_tensors="pt")

        # Get original token count (for slicing later)
        original_length = tokens.shape[1]

        if reverse:
            # Reverse the tokens for reverse prediction
            reversed_tokens = torch.flip(tokens, dims=[1])
            model_input = reversed_tokens.to(device)
        else:
            # Normal prediction (forward)
            model_input = tokens.to(device)

        # Create attention mask
        attention_mask = torch.ones_like(model_input).to(device)

        # Generate text with the model
        output = model.generate(
            model_input,
            attention_mask=attention_mask,
            max_length=original_length + max_length,
            do_sample=True,
            temperature=1.2,
            top_k=40,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

        if reverse:
            # Reverse the output tokens back to normal order
            generated_tokens = torch.flip(output[0], dims=[0])[:max_length].cpu()
        else:
            # Get the newly generated tokens (forward prediction)
            generated_tokens = output[0][
                original_length : original_length + max_length
            ].cpu()

        # Decode the generated tokens
        generated_text = tokenizer.decode(generated_tokens, skip_special_tokens=True)

        results.append(
            {
                "input": example,
                "generated": generated_text,
                "mode": "reverse" if reverse else "forward",
            }
        )

    return results


def load_model(model_name, use_local=False):
    """Load any causal language model and tokenizer."""
    try:
        if use_local:
            # Load from local files
            print(f"Loading model from local path: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
            model = AutoModelForCausalLM.from_pretrained(
                model_name, local_files_only=True
            )
        else:
            # Load from Hugging Face
            print(f"Loading model from Hugging Face: {model_name}")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)

        # Set padding token if not defined
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        return model, tokenizer

    except Exception as e:
        print(f"Error loading model {model_name}: {str(e)}")
        raise


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Benchmark causal language models")

    # Model arguments
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Model name or path (e.g., gpt2, meta-llama/Llama-2-7b-hf, Qwen/Qwen-7B)",
    )
    parser.add_argument(
        "--use_local", action="store_true", help="Use local model checkpoint"
    )

    # Data arguments
    parser.add_argument(
        "--eval_file",
        type=str,
        required=True,
        help="Path to the evaluation data JSONL file",
    )
    parser.add_argument(
        "--max_length", type=int, default=1024, help="Maximum sequence length"
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="Evaluation batch size"
    )

    # Mode arguments
    parser.add_argument(
        "--reverse",
        action="store_true",
        help="Evaluate reversed prediction (like 2-TPG)",
    )
    parser.add_argument(
        "--normal", action="store_true", help="Evaluate normal forward prediction"
    )

    # Example evaluation
    parser.add_argument(
        "--examples",
        nargs="+",
        default=[],
        help="Optional examples to generate continuations for",
    )

    args = parser.parse_args()

    # Check if at least one mode is selected
    if not args.reverse and not args.normal:
        print("Warning: No evaluation mode selected. Defaulting to reverse prediction.")
        args.reverse = True

    # Set device and precision
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Set half precision for large models to save memory
    is_large_model = any(
        x in args.model_name.lower()
        for x in ["llama", "qwen", "gemma", "mistral", "7b", "13b", "70b"]
    )
    dtype = (
        torch.float16 if is_large_model and torch.cuda.is_available() else torch.float32
    )
    print(f"Using precision: {dtype}")

    # Results dictionary
    results = {}

    # Load model
    print(f"Loading model: {args.model_name}")
    model, tokenizer = load_model(args.model_name, args.use_local)

    # Move model to device and set precision
    model = model.to(device)
    if dtype == torch.float16:
        model = model.half()

    # Create evaluation dataset
    print(f"Loading evaluation data from {args.eval_file}")

    # Evaluate in reverse mode
    if args.reverse:
        print("\n" + "=" * 30 + " REVERSE PREDICTION " + "=" * 30)

        eval_dataset = ReverseEvalDataset(
            args.eval_file, tokenizer, max_length=args.max_length, reverse=True
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
        )

        reverse_results = evaluate_model(model, eval_dataloader, device)
        results["reverse"] = reverse_results

        # Print reverse results
        print("\nReverse Prediction Results:")
        print(f"Average Loss: {reverse_results['loss']:.4f}")
        print(f"Perplexity: {reverse_results['perplexity']:.4f}")
        print(f"Total Tokens: {reverse_results['total_tokens']}")

        # Evaluate examples
        if args.examples:
            print("\nGenerating reversed completions:")
            example_results = evaluate_examples(
                model, tokenizer, device, args.examples, reverse=True
            )
            for i, result in enumerate(example_results):
                print(f"\nExample {i + 1}:")
                print(f"Input: {result['input']}")
                print(f"Generated (what came before): {result['generated']}")

    # Evaluate in normal (forward) mode
    if args.normal:
        print("\n" + "=" * 30 + " FORWARD PREDICTION " + "=" * 30)

        eval_dataset = ReverseEvalDataset(
            args.eval_file, tokenizer, max_length=args.max_length, reverse=False
        )
        eval_dataloader = DataLoader(
            eval_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda batch: collate_batch(batch, tokenizer.pad_token_id),
        )

        forward_results = evaluate_model(model, eval_dataloader, device)
        results["forward"] = forward_results

        # Print forward results
        print("\nForward Prediction Results:")
        print(f"Average Loss: {forward_results['loss']:.4f}")
        print(f"Perplexity: {forward_results['perplexity']:.4f}")
        print(f"Total Tokens: {forward_results['total_tokens']}")

        # Evaluate examples
        if args.examples:
            print("\nGenerating forward completions:")
            example_results = evaluate_examples(
                model, tokenizer, device, args.examples, reverse=False
            )
            for i, result in enumerate(example_results):
                print(f"\nExample {i + 1}:")
                print(f"Input: {result['input']}")
                print(f"Generated (what comes next): {result['generated']}")

    # Compare if both modes were used
    if args.reverse and args.normal:
        print("\n" + "=" * 30 + " COMPARISON " + "=" * 30)
        print(f"Reverse Perplexity: {results['reverse']['perplexity']:.4f}")
        print(f"Forward Perplexity: {results['forward']['perplexity']:.4f}")

        if results["reverse"]["perplexity"] < results["forward"]["perplexity"]:
            print(
                "The model performs BETTER at reverse prediction than forward prediction!"
            )
        else:
            print(
                "The model performs BETTER at forward prediction than reverse prediction."
            )

    print("\nBenchmarking complete!")


if __name__ == "__main__":
    main()
