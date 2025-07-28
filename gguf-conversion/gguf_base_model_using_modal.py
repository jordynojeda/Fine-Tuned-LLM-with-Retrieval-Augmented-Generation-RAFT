# gguf_base_model_using_modal.py

import os
import modal
import torch

# Define the Modal app
app = modal.App("convert_gguf")

# Load the Docker image used for the conversion environment
unsloth_image = modal.Image.from_dockerfile("gguf-conversion/Dockerfile")

@app.function(
    gpu="T4",
    image=unsloth_image,
    timeout=60 * 60 * 24,  # 24 hours
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def convert_base_model_gguf(
    hf_model_repo: str,
    save_model_repo: str,
    quantization_method: str = "not_quantized",
    load_in_4bit: bool = False,
) -> str:
    """
    Converts a Hugging Face base model to GGUF format and uploads it to the Hub.

    Args:
        hf_model_repo (str): Source HF model repository (e.g., "meta-llama/Llama-3.2-3B").
        save_model_repo (str): Destination repo to push the GGUF-converted model.
        quantization_method (str): GGUF quantization type (e.g., "q4_K_M", "not_quantized").
        load_in_4bit (bool): Whether to load the model in 4-bit mode.

    Returns:
        str: Empty string upon success (can be updated to return a success message).
    """
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=hf_model_repo,
        max_seq_length=2048,
        dtype=torch.float16,
        load_in_4bit=load_in_4bit,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    # tokenizer.clean_up_tokenization_spaces = False

    model.push_to_hub_gguf(
        save_model_repo,
        tokenizer,
        quantization_method=quantization_method,
        token=os.environ["HF_ACCESS_TOKEN"],
    )

    return ""


@app.local_entrypoint()
def main():
    """Local entrypoint for running GGUF conversion from Hugging Face repo."""
    hf_model_repo = "meta-llama/Llama-3.2-3B"
    save_model_repo = "jordynojeda/Llama-3.2-3B-GGU-f16-4b"
    quantization_method = "not_quantized"
    load_in_4bit = True

    convert_base_model_gguf.remote(
        hf_model_repo=hf_model_repo,
        save_model_repo=save_model_repo,
        quantization_method=quantization_method,
        load_in_4bit=load_in_4bit,
    )


# -- Run this script with:
# modal run gguf-conversion/gguf_base_model_using_modal.py