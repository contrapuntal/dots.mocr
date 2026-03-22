# dots.mocr MLX/CoreML Conversion Design

**Date**: 2026-03-20
**Goal**: Full end-to-end dots.mocr inference on Apple Silicon via MLX/CoreML — faster speed and higher resolution than PyTorch MPS.
**Target**: dots.mocr (3B, Qwen3-VL architecture) on M4 Max, 128GB

## Strategy

Try FL33TW00D's CoreML+MLX approach first, adapted for dots.mocr. If CoreML conversion fails, fall back to pure MLX.

**Why CoreML+MLX first:**
- FL33TW00D demonstrated vision encoder conversion for the nearly-identical dots.ocr architecture
- MLX already supports Qwen2 text generation via `mlx-lm`
- CoreML uses its own optimized compute graph, avoiding the MPS O(N^2) attention mask problem
- CoreML can target the Neural Engine (4-12x more power efficient than GPU)

**Fallback (pure MLX):**
- Port the vision tower to MLX manually (rewrite `DotsVisionTransformer` using `mlx.nn`)
- MLX has native memory-efficient attention
- More work upfront but no framework boundary to debug

## Project Setup

**Project directory**: `mlx/` subdirectory within the dots.mocr repo, with its own `uv venv`. Keeps the PyTorch setup in the root `.venv/` untouched as a fallback.

**Dependencies** (separate venv):
- `torch==2.7.0` (match FL33TW00D's tested version for CoreML tracing)
- `transformers==4.51.0` (dots.mocr's native version)
- `coremltools==9.0b1` (CoreML conversion)
- `mlx`, `mlx-lm` (text backbone inference)
- `qwen-vl-utils` (image preprocessing)
- `coremlprofiler`, `cyclopts` (conversion utilities)

**Model weights**: Symlink to existing `../dots.mocr/weights/DotsMOCR/` — no need to duplicate 5.7GB.

## Phase 1: Convert Vision Encoder to CoreML

Adapt FL33TW00D's `convert.py` for dots.mocr:

1. **Load dots.mocr model** with existing MPS patches (flash_attn shim, sdpa config, processor fix).
2. **Create a wrapper class** that isolates the vision tower: takes `pixel_values` + `image_grid_thw` as input, outputs vision embeddings.
3. **Simplify for tracing**: Single image only (no batch/video), remove dynamic control flow in `cu_seqlens` loop, fix shapes for `torch.jit.trace`.
4. **Convert via coremltools**: `ct.convert(traced_model, ...)` with FP16 precision, targeting `CPU_AND_GPU` compute units.
5. **Validate**: Compare CoreML output vs PyTorch output on the same test image. Must match within tolerance (~0.006 max diff based on FL33TW00D's results).

**Key adaptation**: dots.mocr's vision tower is 42 layers, 12 heads, 1536 embed dim. The `VisionSdpaAttention` must use standard SDPA without a mask for tracing — already implemented in our modified `modeling_dots_vision.py`.

## Phase 2: Convert Text Backbone to MLX

The text backbone is Qwen2-based, which `mlx-lm` already supports:

1. **Extract text-only weights**: Filter out `vision_tower.*` tensors from safetensors shards, relabel config as `qwen2` model type.
2. **Convert to MLX format**: `mlx_lm.convert --hf-path ./text_only_weights -q` (4-bit quantization optional, reduces ~6GB to ~1.5GB).
3. **Validate**: Run text-only generation to confirm the backbone works in MLX.

**Risk**: dots.mocr uses Qwen3-VL architecture, not vanilla Qwen2. If `mlx-lm` doesn't handle the specific layer config, we may need to register a custom model class. Check `mlx-lm`'s model registry for Qwen3/Qwen2.5-VL support before starting.

## Phase 3: Integration (Vision + Text End-to-End)

This is the unpublished part — FL33TW00D hasn't released this yet.

1. **Image preprocessing**: Use `qwen_vl_utils.process_vision_info` + `Qwen2VLImageProcessor` (same as PyTorch path) to produce `pixel_values` and `image_grid_thw`.
2. **Vision encoding**: Run CoreML model on preprocessed pixels to get vision embeddings tensor.
3. **Embedding injection**:
   - Tokenize the text prompt via MLX tokenizer.
   - Locate image token positions (token ID 151665).
   - Replace image token embeddings with CoreML vision embeddings.
   - This is the `prepare_inputs_embeds` logic from `modeling_dots_ocr.py`, reimplemented for MLX.
4. **Text generation**: Feed combined embeddings into MLX text model, generate autoregressively.
5. **Decode**: Standard `mlx-lm` token decoding.

**The critical bridge**: Converting a CoreML output tensor (numpy/MLMultiArray) to an MLX array for injection into the text model. This should be zero-copy via `mx.array(coreml_output.numpy())`.

## Fallback: Pure MLX

If CoreML conversion fails (unsupported ops, tracing failures from dynamic shapes):

1. **Rewrite vision tower in MLX**: Port `DotsVisionTransformer`, `VisionSdpaAttention`, `DotsPatchEmbed`, `PatchMerger`, `VisionRotaryEmbedding` to `mlx.nn` equivalents.
2. **Load weights manually**: Map PyTorch state dict keys to MLX model structure, load via `mx.load`.
3. **Advantage**: MLX's `mx.fast.scaled_dot_product_attention` is memory-efficient natively — no mask needed, no O(N^2) issue.
4. **Disadvantage**: More code to write and maintain (~300-400 lines of MLX model code).

## Success Criteria

The conversion is complete when:

1. **Accuracy**: OCR output on `demo/demo_image1.jpg` with `prompt_layout_all_en` matches PyTorch output structurally (same bboxes, same text content, minor float differences acceptable).
2. **All modes**: At least `prompt_ocr`, `prompt_layout_all_en`, and `prompt_general` produce correct output.
3. **Speed**: Faster than PyTorch MPS baseline (currently 15-30 tok/s).
4. **Resolution**: Can handle images >4M pixels without OOM (target: 8M+ pixels).

## Risk Summary

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| CoreML tracing fails on vision tower dynamic shapes | Medium | High | Fall back to pure MLX |
| `mlx-lm` doesn't support Qwen3-VL variant | Medium | Medium | Register custom MLX model class |
| Embedding injection bridge has numerical drift | Low | High | Compare per-layer outputs, not just final |
| CoreML FP16 accuracy insufficient for OCR | Low | Medium | Use FP32 or mixed precision |
| `coremltools` beta incompatible with torch 2.7 on macOS 26 | Low | High | Try stable coremltools release |

## References

- FL33TW00D's dots.ocr.ne: https://github.com/FL33TW00D/dots.ocr.ne
- FL33TW00D's HF blog: https://huggingface.co/blog/dots-ocr-ne
- Tomorrow's Innovations MLX conversion: https://www.tomorrowsinnovations.co/blog/converting-dots-ocr-to-mlx-part-1
- dots.ocr issue #74 (NPU/CPU/MPS support): https://github.com/rednote-hilab/dots.ocr/issues/74
- dots.ocr PR #119 (macOS MPS support): https://github.com/rednote-hilab/dots.ocr/pull/119
- Metal flash attention status: https://github.com/philipturner/metal-flash-attention/issues/32
