"""
Validate MLX vision tower output matches PyTorch.
Loads the same weights, runs the same image, compares outputs.
"""
import sys
import os
import types
import importlib.machinery

# flash_attn shim for PyTorch model loading
if "flash_attn" not in sys.modules:
    _m = types.ModuleType("flash_attn")
    _m.__spec__ = importlib.machinery.ModuleSpec(name="flash_attn", loader=None)
    _m.__path__ = []
    def _stub(*a, **kw): raise RuntimeError("no flash_attn")
    _m.flash_attn_varlen_func = _stub
    sys.modules["flash_attn"] = _m

os.environ["LOCAL_RANK"] = "0"

import numpy as np
import torch
import mlx.core as mx
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from vision_tower import DotsVisionTower, VisionConfig, load_vision_weights

from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = str(_SCRIPT_DIR / ".." / "weights" / "DotsMOCR")
TEST_IMAGE = str(_SCRIPT_DIR / ".." / "demo" / "demo_image1.jpg")


def get_pytorch_output():
    """Run PyTorch vision tower and return embeddings + test inputs."""
    print("Loading PyTorch model...")
    config = AutoConfig.from_pretrained(WEIGHTS_DIR, trust_remote_code=True)
    config.vision_config.attn_implementation = "sdpa"
    config.use_sliding_window = False

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS_DIR, config=config, attn_implementation="sdpa",
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(WEIGHTS_DIR, trust_remote_code=True)

    messages = [{"role": "user", "content": [
        {"type": "image", "image": TEST_IMAGE},
        {"type": "text", "text": "test"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")

    pixel_values = inputs["pixel_values"].float()
    grid_thw = inputs["image_grid_thw"]

    print(f"  pixel_values: {pixel_values.shape}")
    print(f"  grid_thw: {grid_thw}")

    with torch.no_grad():
        pt_output = model.vision_tower(pixel_values, grid_thw, bf16=False)

    print(f"  PyTorch output: {pt_output.shape}")
    return pt_output.numpy(), pixel_values.numpy(), grid_thw.numpy()


def get_mlx_output(pixel_values_np, grid_thw_np):
    """Run MLX vision tower with the same inputs."""
    print("Loading MLX vision tower...")
    vc = VisionConfig()
    mlx_model = DotsVisionTower(vc)
    load_vision_weights(mlx_model, WEIGHTS_DIR)

    pixel_values = mx.array(pixel_values_np)
    grid_thw = mx.array(grid_thw_np)

    print(f"  Running MLX forward...")
    mlx_output = mlx_model(pixel_values, grid_thw)
    mx.eval(mlx_output)

    print(f"  MLX output: {mlx_output.shape}")
    return np.array(mlx_output)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-cache", action="store_true",
                        help="Force recompute PyTorch reference (ignore stale cache)")
    args = parser.parse_args()

    cache_path = "/tmp/dots_mocr_pt_vision_ref.npz"
    if args.no_cache and os.path.exists(cache_path):
        os.remove(cache_path)
        print("Cleared cached PyTorch reference")

    if os.path.exists(cache_path):
        print(f"Loading cached PyTorch reference from {cache_path}")
        print(f"  (use --no-cache to force recompute)")
        data = np.load(cache_path)
        pt_output, pixel_values, grid_thw = data["output"], data["pixel_values"], data["grid_thw"]
    else:
        pt_output, pixel_values, grid_thw = get_pytorch_output()
        np.savez(cache_path, output=pt_output, pixel_values=pixel_values, grid_thw=grid_thw)
        print(f"Cached PyTorch reference to {cache_path}")

    mlx_output = get_mlx_output(pixel_values, grid_thw)

    print(f"\n=== Comparison ===")
    print(f"PyTorch shape: {pt_output.shape}")
    print(f"MLX shape:     {mlx_output.shape}")
    print(f"Shape match:   {pt_output.shape == mlx_output.shape}")

    if pt_output.shape == mlx_output.shape:
        max_diff = np.abs(pt_output - mlx_output).max()
        mean_diff = np.abs(pt_output - mlx_output).mean()
        pt_flat = pt_output.flatten()
        mlx_flat = mlx_output.flatten()
        cos_sim = np.dot(pt_flat, mlx_flat) / (
            np.linalg.norm(pt_flat) * np.linalg.norm(mlx_flat)
        )
        print(f"Max diff:      {max_diff:.6f}")
        print(f"Mean diff:     {mean_diff:.6f}")
        print(f"Cosine sim:    {cos_sim:.6f}")

        if max_diff < 0.1 and cos_sim > 0.99:
            print("\nPASS: MLX output matches PyTorch within tolerance")
        else:
            print(f"\nFAIL: Outputs diverge (max_diff={max_diff:.4f}, cos_sim={cos_sim:.4f})")
            sys.exit(1)
    else:
        print("\nFAIL: Shape mismatch")
        sys.exit(1)


if __name__ == "__main__":
    main()
