"""
Extract text-only weights from dots.mocr and convert to MLX format.

Two-step process:
  1. Strip vision_tower weights, remap config to qwen2 -> ./text_backbone/
  2. Convert to MLX via mlx_lm.convert -> ./mlx_text_model/
"""
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from safetensors.torch import load_file, save_file

# Anchor paths relative to this script, not the caller's cwd
_SCRIPT_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = str(_SCRIPT_DIR / ".." / "weights" / "DotsMOCR")
INTERMEDIATE_DIR = str(_SCRIPT_DIR / "text_backbone")
OUTPUT_DIR = str(_SCRIPT_DIR / "mlx_text_model")


def main():
    # Clean stale artifacts from previous runs
    for d in [INTERMEDIATE_DIR, OUTPUT_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
            print(f"Removed stale {d}/")
    # 1. Load and adapt config
    with open(os.path.join(WEIGHTS_DIR, "config.json")) as f:
        config = json.load(f)

    # Remove vision-specific fields
    config.pop("vision_config", None)
    config.pop("image_token_id", None)
    config.pop("video_token_id", None)
    config.pop("auto_map", None)

    # Remap to qwen2 so mlx-lm recognizes it
    config["model_type"] = "qwen2"
    config["architectures"] = ["Qwen2ForCausalLM"]

    os.makedirs(INTERMEDIATE_DIR, exist_ok=True)
    with open(os.path.join(INTERMEDIATE_DIR, "config.json"), "w") as f:
        json.dump(config, f, indent=2)
    print(f"Config written to {INTERMEDIATE_DIR}/config.json")

    # 2. Filter and merge safetensors (strip vision_tower.* weights)
    index_path = os.path.join(WEIGHTS_DIR, "model.safetensors.index.json")
    with open(index_path) as f:
        index = json.load(f)

    shard_files = sorted(set(index["weight_map"].values()))
    print(f"Loading {len(shard_files)} shards...")

    text_weights = {}
    vision_count = 0
    for shard in shard_files:
        shard_path = os.path.join(WEIGHTS_DIR, shard)
        weights = load_file(shard_path)
        for key, tensor in weights.items():
            if key.startswith("vision_tower."):
                vision_count += 1
                continue
            text_weights[key] = tensor

    print(f"Kept {len(text_weights)} text tensors, removed {vision_count} vision tensors")

    # 3. Save as single shard
    output_shard = os.path.join(INTERMEDIATE_DIR, "model.safetensors")
    save_file(text_weights, output_shard)

    new_index = {
        "metadata": {"total_size": sum(t.nelement() * t.element_size() for t in text_weights.values())},
        "weight_map": {k: "model.safetensors" for k in text_weights},
    }
    with open(os.path.join(INTERMEDIATE_DIR, "model.safetensors.index.json"), "w") as f:
        json.dump(new_index, f, indent=2)

    # 4. Copy tokenizer files
    for fname in ["tokenizer.json", "tokenizer_config.json", "special_tokens_map.json",
                   "merges.txt", "vocab.json", "generation_config.json"]:
        src = os.path.join(WEIGHTS_DIR, fname)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(INTERMEDIATE_DIR, fname))

    total_mb = os.path.getsize(output_shard) / 1e6
    print(f"Text backbone extracted to {INTERMEDIATE_DIR}/ ({total_mb:.0f} MB)")

    # 5. Convert to MLX format
    print(f"\nConverting to MLX format -> {OUTPUT_DIR}/")
    subprocess.run(
        [sys.executable, "-m", "mlx_lm", "convert",
         "--hf-path", INTERMEDIATE_DIR, "--mlx-path", OUTPUT_DIR],
        check=True,
    )
    print(f"MLX text model ready at {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
