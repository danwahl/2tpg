import torch
import argparse
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(model_path, use_local_checkpoint=False):
    """
    Load a fine-tuned GPT-2 model and tokenizer either from a local checkpoint or Hugging Face.

    Args:
        model_path (str): Path to local checkpoint or Hugging Face model ID
        use_local_checkpoint (bool): Whether to load from a local checkpoint

    Returns:
        tuple: (model, tokenizer, device)
    """
    print(
        f"Loading {'local checkpoint' if use_local_checkpoint else 'model from Hugging Face'}: {model_path}"
    )

    # Load the tokenizer
    if use_local_checkpoint:
        # Use base GPT-2 tokenizer for local checkpoint (since it may not be saved with the model)
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        model = GPT2LMHeadModel.from_pretrained(model_path, local_files_only=True)
    else:
        # Load from Hugging Face Hub
        tokenizer = GPT2Tokenizer.from_pretrained(model_path)
        model = GPT2LMHeadModel.from_pretrained(model_path)

    # Set padding token to be the EOS token
    tokenizer.pad_token = tokenizer.eos_token

    # Check if GPU is available and move model to GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    print(f"Using device: {device}")

    return model, tokenizer, device


def reverse_generate_text(
    prompt, model, tokenizer, device, max_length=100, num_return_sequences=1
):
    """
    Generate text using a model trained on reversed tokens.

    Args:
        prompt (str): The input prompt
        model: The pre-trained GPT-2 model
        tokenizer: The tokenizer for the model
        device: The device (CPU or GPU) to run on
        max_length (int): Maximum length of the generated text
        num_return_sequences (int): Number of sequences to generate

    Returns:
        list: List of generated text sequences
    """
    # Tokenize the prompt
    tokens = tokenizer.encode(prompt, return_tensors="pt")

    # Reverse the tokens (since our model was trained on reversed sequences)
    reversed_tokens = torch.flip(tokens, dims=[1])

    # Create attention mask and move to device
    attention_mask = torch.ones_like(reversed_tokens)
    reversed_tokens = reversed_tokens.to(device)
    attention_mask = attention_mask.to(device)

    # Generate text with the model that was trained on reversed sequences
    output = model.generate(
        reversed_tokens,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        do_sample=True,
        temperature=1.2,
        top_k=40,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    # Reverse the output tokens back to normal order
    reversed_outputs = [torch.flip(ids, dims=[0]) for ids in output]

    # Decode the reversed output tokens
    generated_texts = [
        tokenizer.decode(ids, skip_special_tokens=True) for ids in reversed_outputs
    ]

    return generated_texts


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Generate text using a reverse-trained GPT-2 model"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="And they lived happily ever after.",
        help="Input prompt for text generation",
    )
    parser.add_argument(
        "--max_length", type=int, default=150, help="Maximum length of generated text"
    )
    parser.add_argument(
        "--num_sequences", type=int, default=1, help="Number of sequences to generate"
    )
    parser.add_argument(
        "--use_local",
        action="store_true",
        help="Use local checkpoint instead of Hugging Face model",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="drwahl/2tpg",
        help="Path to local checkpoint or Hugging Face model ID",
    )

    args = parser.parse_args()

    # Load the model, tokenizer, and determine device
    model, tokenizer, device = load_model(args.model_path, args.use_local)

    # Print input prompt
    print(f"Input prompt: {args.prompt}")

    # Generate text with reverse model
    generated_texts = reverse_generate_text(
        args.prompt,
        model,
        tokenizer,
        device,
        max_length=args.max_length,
        num_return_sequences=args.num_sequences,
    )

    # Print the generated text
    for i, text in enumerate(generated_texts):
        print(f"\nGenerated Text {i + 1}:\n{text}")


if __name__ == "__main__":
    main()
