# dots.mocr

3B parameter vision-language OCR model from rednote-hilab. Architecture: Qwen2-based text backbone (`Qwen2ForCausalLM`) with custom vision tower (`DotsVisionTransformer`), packaged as `DotsOCRForCausalLM`. Processor uses `Qwen2_5_VLProcessor`.

## Setup

- Python env: `uv venv` with Python 3.12, located at `.venv/`
- Install: `uv pip install -e .` (editable install)
- Model weights: `./weights/DotsMOCR/` (~5.7GB safetensors, downloaded via `python3 tools/download_model.py`)
- Transformers pinned to 4.51.0 (model's native version; 5.x breaks `prepare_inputs_for_generation`)

## Running on Apple Silicon (MPS)

```bash
source .venv/bin/activate
DYLD_FALLBACK_LIBRARY_PATH=/opt/homebrew/lib python3 demo/demo_mps.py
```

`DYLD_FALLBACK_LIBRARY_PATH` is needed because `cairosvg` can't find Homebrew's `libcairo` otherwise (external drive path issue).

### Local modifications for MPS compatibility

These files in `weights/DotsMOCR/` were modified from upstream:

- **`modeling_dots_vision.py`**:
  - `flash_attn` import wrapped in try/except (CUDA-only package)
  - `VisionSdpaAttention.forward` rewritten to split by `cu_seqlens` and call maskless `F.scaled_dot_product_attention` per sub-sequence. This avoids allocating the O(N^2) attention mask that caused OOM on large images. Inspired by [dots.ocr#74](https://github.com/rednote-hilab/dots.ocr/issues/74).
- **`config.json`**: Vision `attn_implementation` changed from `flash_attention_2` to `sdpa`
- **`configuration_dots.py`**: Fixed `DotsVLProcessor.__init__` positional arg mismatch with transformers 4.51's `Qwen2_5_VLProcessor` (no `video_processor` param)

### MPS demo script approach (demo_mps.py, test_all_modes_mps.py)

- **SDPA attention**: Both vision tower and main model use `attn_implementation="sdpa"`. The `flash_attn` import in `modeling_dots_vision.py` is guarded by try/except so no shim is needed.
- **float16**: Uses `torch.float16` instead of `bfloat16`. The vision tower's `forward()` auto-skips the `bfloat16()` cast when `device.type == "mps"`.
- **Sliding window disabled**: `config.use_sliding_window=False` suppresses "not implemented" warning
- **Processor max_pixels**: Set to `MPS_MAX_PIXELS` (4M) to control actual tensor size fed to the vision tower

### MPS limitations

- **Max image size**: Capped at ~4M pixels (`MPS_MAX_PIXELS` in demo scripts). The vision tower has 42 attention layers; combined with model weights and LLM KV cache, larger images exhaust unified memory. The processor's `max_pixels` parameter enforces this at the tokenization level.
- **No Metal flash attention**: `mps-flash-attention` / `metal-flash-attention` do not work on macOS 26 + M4 Max. Apple banned SIMD-group async copies, and the Metal shader compiler (XPC service) crashes with `AGXMetalG16X Code=2` when trying to JIT-compile attention kernels for M4's GPU family. See [metal-flash-attention#32](https://github.com/philipturner/metal-flash-attention/issues/32).
- **Eager attention only for main LLM**: The main model (Qwen2-based) uses SDPA but MPS SDPA falls back to eager math for the language model layers. This is slower than flash attention on CUDA.
- **Web parsing quality**: At 4M pixels, a 25M pixel webpage image is resized to ~1500x2600, losing fine detail. Web parsing technically passes but produces minimal output. Higher quality requires more memory headroom or CUDA.

### Test results (M4 Max, 128GB, MPS)

All 8 prompt modes smoke-tested via `demo/test_all_modes_mps.py` (SDPA + fp16, max_pixels=4M). These are smoke/perf tests (non-exception = PASS), not correctness validations. Outputs hitting `max_new_tokens` are flagged as truncated.

| Mode | Status | Time | Tokens | Speed | Notes |
|------|--------|------|--------|-------|-------|
| prompt_layout_all_en | PASS | 130s | 2048 | 15.8 tok/s | Full layout + text (truncated) |
| prompt_layout_only_en | PASS | 47s | 375 | 8.1 tok/s | Bboxes only |
| prompt_ocr | PASS | 106s | 2048 | 19.3 tok/s | Text extraction (truncated) |
| prompt_grounding_ocr | PASS | 105s | 2048 | 19.5 tok/s | Region-specific OCR (truncated) |
| prompt_web_parsing | PASS | 35s | 32 | 0.9 tok/s | Resized to 4M px; sparse output |
| prompt_scene_spotting | PASS | 12s | 298 | 25.7 tok/s | Scene text detection |
| prompt_image_to_svg | PASS | 41s | 1238 | 30.5 tok/s | SVG code generation |
| prompt_general | PASS | 105s | 2048 | 19.5 tok/s | Free-form VQA (truncated) |

8/8 modes pass (smoke test). Performance is ~15-30 tok/s depending on mode. Modes hitting 2048 tokens were truncated at `max_new_tokens` limit.

## Running on Apple Silicon (MLX) — Recommended

Use [mlx-vlm](https://github.com/Blaizzy/mlx-vlm) for MLX inference. It loads dots.mocr weights directly — no conversion needed. See [mlx-vlm PR #749](https://github.com/Blaizzy/mlx-vlm/pull/749) for the dots_ocr model implementation.

```bash
pip install mlx-vlm
```

```python
from mlx_vlm import load, generate
from mlx_vlm.prompt_utils import apply_chat_template
from transformers import Qwen2VLImageProcessor

model, processor = load("rednote-hilab/dots.mocr")
processor.image_processor = Qwen2VLImageProcessor.from_pretrained("rednote-hilab/dots.mocr")

prompt = apply_chat_template(processor, model.config, "Extract the text content from this image.", num_images=1)
result = generate(model, processor, prompt, image=["demo/demo_image1.jpg"], max_tokens=2048)
print(result.text)
```

**Note**: The `Qwen2VLImageProcessor` swap is needed because mlx-vlm loads the fast image processor by default, which rejects `return_tensors="mlx"` on transformers 5.x ([#525](https://github.com/Blaizzy/mlx-vlm/issues/525)). The slow processor handles it correctly.

### MLX performance (M4 Max, 128GB)

| Metric | PyTorch MPS | MLX (mlx-vlm) |
|--------|------------|---------------|
| Text generation | 15-20 tok/s | **74 tok/s** (5x) |
| Max pixels before OOM | 4M | **8M+** |
| Peak memory | ~15 GB | **7.5 GB** |

### Why not CoreML

CoreML conversion was attempted but failed: `torch.jit.trace` bakes dynamic tensor shapes as `aten::Int` ops that `coremltools` cannot convert. The vision tower's `grid_thw`-driven control flow makes the export boundary inherently dynamic.

## Project structure

- `dots_mocr/` — main package (parser, model inference, utilities)
- `demo/` — demo scripts (`demo_hf.py` for CUDA, `demo_mps.py` for Apple Silicon, `test_all_modes_mps.py` for comprehensive MPS test, `demo_vllm.py` for vLLM server)
- `tools/` — model download script
- `assets/` — showcase images

## Key files

- `dots_mocr/parser.py` — `DotsMOCRParser` class, supports both vLLM server and HF local inference
- `dots_mocr/utils/prompts.py` — prompt templates for different tasks (`dict_promptmode_to_prompt`)
- `dots_mocr/model/inference.py` — vLLM client (OpenAI-compatible API)
