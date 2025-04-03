import torch
from transformers import GPT2LMHeadModel, GPT2Tokenizer


def load_model(checkpoint_path):
    """
    Load a fine-tuned GPT-2 model from a local checkpoint and the base tokenizer.

    Args:
        checkpoint_path (str): Path to the saved model checkpoint

    Returns:
        tuple: (model, tokenizer, device)
    """
    import torch

    # Load the base GPT-2 tokenizer (since it may not be saved with the model)
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

    # Load model from checkpoint
    print(f"Loading model from {checkpoint_path}")
    model = GPT2LMHeadModel.from_pretrained(checkpoint_path)

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
    # Path to your checkpoint
    checkpoint_path = "checkpoints/checkpoint-epoch-1"  # Adjust to your checkpoint path

    # Load the model, tokenizer, and determine device
    model, tokenizer, device = load_model(checkpoint_path)

    # Example prompt (or get from user input)
    prompt = """So we beat on, boats against the current, borne back ceaselessly into the past."""
    print(f"Input prompt: {prompt}")

    # Generate text with reverse model
    generated_texts = reverse_generate_text(
        prompt, model, tokenizer, device, max_length=256
    )

    # Print the generated text
    for i, text in enumerate(generated_texts):
        print(f"\nGenerated Text {i + 1}:\n{text}")


if __name__ == "__main__":
    main()
