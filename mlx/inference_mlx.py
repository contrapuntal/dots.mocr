"""
End-to-end dots.mocr inference on Apple Silicon using pure MLX.

Pipeline:
  1. Preprocess image via qwen_vl_utils (PyTorch, CPU-only)
  2. Run MLX vision tower on pixel_values -> vision embeddings
  3. Tokenize prompt, get text embeddings, inject vision embeddings
  4. Generate text autoregressively via mlx-lm
"""
import os
import sys
import time

os.environ["LOCAL_RANK"] = "0"

import numpy as np
import mlx.core as mx
import mlx.nn as mlx_nn
from mlx_lm import load as mlx_load

from vision_tower import DotsVisionTower, VisionConfig, load_vision_weights

# Use PyTorch processor for image preprocessing only (CPU, no GPU needed)
import torch
from transformers import AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info

WEIGHTS_DIR = "../weights/DotsMOCR"
MLX_TEXT_MODEL = "./mlx_text_model"
IMAGE_TOKEN_ID = 151665
MLX_MAX_PIXELS = 8_000_000


def preprocess_image(processor, image_path, prompt_text):
    """Use PyTorch processor to get pixel_values, grid_thw, and input_ids."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": prompt_text},
    ]}]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(
        text=[text], images=image_inputs,
        padding=True, return_tensors="pt",
    )

    return {
        "pixel_values": inputs["pixel_values"].float().numpy(),
        "image_grid_thw": inputs["image_grid_thw"].numpy(),
        "input_ids": inputs["input_ids"].numpy(),
    }


def prepare_inputs_embeds(text_model, vision_embeddings, input_ids):
    """Replace image token embeddings with vision embeddings.

    This is the MLX equivalent of DotsOCRForCausalLM.prepare_inputs_embeds.
    """
    input_ids_mx = mx.array(input_ids)

    # Get text embeddings for all tokens
    text_embeds = text_model.model.embed_tokens(input_ids_mx)  # (1, seq_len, hidden)

    # Find image token positions
    img_mask = (input_ids_mx == IMAGE_TOKEN_ID)  # (1, seq_len)
    num_image_tokens = int(img_mask.sum().item())

    if num_image_tokens == 0:
        return text_embeds

    num_vision = vision_embeddings.shape[0]

    # Match PyTorch reference: if more image tokens than vision embeddings,
    # truncate the mask (processor/template drift tolerance)
    if num_image_tokens > num_vision:
        print(f"[warn] img tokens ({num_image_tokens}) > vision embeddings ({num_vision}), truncating mask")
        mask_flat = img_mask.reshape(-1)
        indices = mx.argwhere(mask_flat).reshape(-1)
        indices = indices[:num_vision]
        new_mask = mx.zeros_like(mask_flat)
        # Rebuild mask with only the first num_vision positions
        mask_np_full = np.zeros(mask_flat.shape, dtype=bool)
        mask_np_full[np.array(indices)] = True
        img_mask = mx.array(mask_np_full).reshape(img_mask.shape)
        num_image_tokens = num_vision

    assert num_image_tokens == num_vision, (
        f"Image token count {num_image_tokens} != vision embeddings {num_vision}"
    )

    # Flatten to 2D for indexing
    seq_len = text_embeds.shape[1]
    hidden = text_embeds.shape[2]
    flat_text = text_embeds.reshape(seq_len, hidden)
    mask_flat = img_mask.reshape(seq_len)

    # Cast to float32 for numpy (MLX bfloat16 isn't numpy-compatible)
    combined = np.array(flat_text.astype(mx.float32))
    mask_np = np.array(mask_flat)
    vision_np = np.array(vision_embeddings.astype(mx.float32))
    combined[mask_np] = vision_np

    return mx.array(combined).astype(flat_text.dtype).reshape(1, seq_len, hidden)


def generate_with_embeddings(text_model, tokenizer, input_embeddings, input_ids, max_tokens=2048):
    """Autoregressive generation starting from pre-computed embeddings."""
    from mlx_lm.models.cache import make_prompt_cache

    # Create KV cache
    cache = make_prompt_cache(text_model)

    # Prefill: forward pass with embeddings, populates KV cache
    logits = text_model(mx.array(input_ids), cache=cache, input_embeddings=input_embeddings)
    mx.eval(logits)

    # Greedy decode with KV cache
    generated = []
    next_token = mx.argmax(logits[:, -1, :], axis=-1)
    mx.eval(next_token)

    eos_token_id = tokenizer.eos_token_id

    for i in range(max_tokens):
        token_id = next_token.item()
        if token_id == eos_token_id:
            break
        generated.append(token_id)

        # Decode one token at a time, reusing KV cache
        logits = text_model(next_token.reshape(1, 1), cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1, :], axis=-1)
        mx.eval(next_token)

    return tokenizer.decode(generated)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", default="../dots.mocr/demo/demo_image1.jpg")
    parser.add_argument("--prompt-mode", default="prompt_layout_all_en")
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--max-pixels", type=int, default=MLX_MAX_PIXELS)
    args = parser.parse_args()

    # Load prompts from source file directly (avoids importing dots_mocr package
    # which pulls in fitz/cairosvg dependencies we don't have in the mlx venv)
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "prompts", os.path.join(WEIGHTS_DIR, "../../dots_mocr/utils/prompts.py")  # ../dots_mocr/utils/prompts.py
    )
    prompts_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(prompts_mod)
    prompt_text = prompts_mod.dict_promptmode_to_prompt[args.prompt_mode]

    # 1. Load PyTorch processor (CPU only, for tokenization)
    print("Loading processor...")
    processor = AutoProcessor.from_pretrained(
        WEIGHTS_DIR, trust_remote_code=True,
        max_pixels=args.max_pixels,
    )

    # 2. Preprocess image
    print(f"Preprocessing image: {args.image}")
    inputs = preprocess_image(processor, args.image, prompt_text)
    print(f"  pixel_values: {inputs['pixel_values'].shape}")
    print(f"  grid_thw: {inputs['image_grid_thw']}")
    print(f"  input_ids: {inputs['input_ids'].shape}")

    # 3. Load MLX vision tower
    print("Loading MLX vision tower...")
    vc = VisionConfig()
    vision_model = DotsVisionTower(vc)
    load_vision_weights(vision_model, WEIGHTS_DIR)

    # 4. Run vision tower
    print("Running vision encoder...")
    t0 = time.time()
    pixel_values = mx.array(inputs["pixel_values"])
    grid_thw = mx.array(inputs["image_grid_thw"])
    vision_embeddings = vision_model(pixel_values, grid_thw)
    mx.eval(vision_embeddings)
    vision_time = time.time() - t0
    print(f"  Vision embeddings: {vision_embeddings.shape} ({vision_time:.1f}s)")

    # 5. Load MLX text model
    print("Loading MLX text model...")
    text_model, tokenizer = mlx_load(MLX_TEXT_MODEL)

    # 6. Prepare combined embeddings
    print("Preparing embeddings...")
    input_embeddings = prepare_inputs_embeds(
        text_model, vision_embeddings, inputs["input_ids"]
    )
    mx.eval(input_embeddings)
    print(f"  Combined embeddings: {input_embeddings.shape}")

    # 7. Generate
    print(f"Generating (max {args.max_tokens} tokens)...")
    t0 = time.time()
    output = generate_with_embeddings(
        text_model, tokenizer, input_embeddings,
        inputs["input_ids"], max_tokens=args.max_tokens,
    )
    gen_time = time.time() - t0
    num_tokens = len(tokenizer.encode(output))
    print(f"  Generated {num_tokens} tokens in {gen_time:.1f}s ({num_tokens/gen_time:.1f} tok/s)")

    print(f"\n{'='*60}")
    print(f"OUTPUT ({args.prompt_mode}):")
    print(f"{'='*60}")
    print(output[:3000])
    if len(output) > 3000:
        print("... [truncated]")


if __name__ == "__main__":
    main()
