"""
Convert dots.mocr vision tower to CoreML.

Wraps the vision tower (DotsVisionTransformer) to take pixel_values + image_grid_thw
and output vision embeddings. Traces with torch.jit.trace and converts via coremltools.
"""
import os
import sys
import types
import importlib.machinery

# flash_attn shim needed during model loading
if "flash_attn" not in sys.modules:
    _m = types.ModuleType("flash_attn")
    _m.__spec__ = importlib.machinery.ModuleSpec(name="flash_attn", loader=None)
    _m.__path__ = []
    def _stub(*a, **kw): raise RuntimeError("no flash_attn")
    _m.flash_attn_varlen_func = _stub
    sys.modules["flash_attn"] = _m

import torch
import torch.nn as nn
import numpy as np
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from PIL import Image

WEIGHTS_DIR = "../weights/DotsMOCR"
OUTPUT_PATH = "./vision_encoder.mlpackage"
TEST_IMAGE = "../demo/demo_image1.jpg"


class VisionTowerWrapper(nn.Module):
    """Wraps the vision tower for tracing. Single-image only.

    Patches VisionSdpaAttention to use plain SDPA without cu_seqlens split,
    since single-image means the entire sequence is one block — no split needed.
    This avoids the .tolist() and torch.split() ops that CoreML can't convert.
    """

    def __init__(self, vision_tower):
        super().__init__()
        self.vision_tower = vision_tower
        self._patch_attention_for_tracing()

    def _patch_attention_for_tracing(self):
        """Replace per-sub-sequence SDPA with plain SDPA for single-image tracing."""
        import torch.nn.functional as F
        from types import MethodType

        # Grab the rotary embedding function from the model's module
        attn_module = sys.modules[type(self.vision_tower.blocks[0].attn).__module__]
        _apply_rope = attn_module.apply_rotary_pos_emb_vision

        for block in self.vision_tower.blocks:
            attn = block.attn
            if type(attn).__name__ == "VisionSdpaAttention":
                def _make_forward(apply_rope_fn):
                    def _simple_forward(self, hidden_states, cu_seqlens, rotary_pos_emb=None):
                        seq_length = hidden_states.shape[0]
                        q, k, v = self.qkv(hidden_states).reshape(
                            seq_length, 3, self.num_heads, -1
                        ).permute(1, 0, 2, 3).unbind(0)
                        q = apply_rope_fn(q.unsqueeze(0), rotary_pos_emb).squeeze(0)
                        k = apply_rope_fn(k.unsqueeze(0), rotary_pos_emb).squeeze(0)
                        q = q.transpose(0, 1)
                        k = k.transpose(0, 1)
                        v = v.transpose(0, 1)
                        attn_output = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
                        attn_output = attn_output.transpose(0, 1)
                        attn_output = attn_output.reshape(seq_length, -1)
                        attn_output = self.proj(attn_output)
                        return attn_output
                    return _simple_forward
                attn.forward = MethodType(_make_forward(_apply_rope), attn)

    def forward(self, pixel_values, image_grid_thw):
        return self.vision_tower(pixel_values, image_grid_thw, bf16=False)


def get_test_inputs(processor, image_path):
    """Preprocess a test image and return pixel_values + grid_thw."""
    messages = [{"role": "user", "content": [
        {"type": "image", "image": image_path},
        {"type": "text", "text": "test"},
    ]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, _ = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt")
    return inputs["pixel_values"].float(), inputs["image_grid_thw"]


def main():
    print("Loading dots.mocr model...")
    config = AutoConfig.from_pretrained(WEIGHTS_DIR, trust_remote_code=True)
    config.vision_config.attn_implementation = "sdpa"
    config.use_sliding_window = False
    config.sliding_window = None

    model = AutoModelForCausalLM.from_pretrained(
        WEIGHTS_DIR, config=config, attn_implementation="sdpa",
        torch_dtype=torch.float32, low_cpu_mem_usage=True,
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(WEIGHTS_DIR, trust_remote_code=True)
    print("Model loaded.")

    # Get test inputs
    pixel_values, image_grid_thw = get_test_inputs(processor, TEST_IMAGE)
    print(f"Test inputs: pixel_values={pixel_values.shape}, grid_thw={image_grid_thw}")

    # Get PyTorch reference output
    wrapper = VisionTowerWrapper(model.vision_tower)
    wrapper.eval()
    with torch.no_grad():
        ref_output = wrapper(pixel_values, image_grid_thw)
    print(f"PyTorch vision output: {ref_output.shape}, dtype={ref_output.dtype}")

    # Trace
    print("Tracing vision tower...")
    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (pixel_values, image_grid_thw))
    print("Trace successful.")

    # Verify trace
    with torch.no_grad():
        trace_output = traced(pixel_values, image_grid_thw)
    max_diff = (ref_output - trace_output).abs().max().item()
    print(f"Trace verification: max_diff={max_diff:.6f}")

    # Convert to CoreML
    print("Converting to CoreML...")
    import coremltools as ct

    coreml_model = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="pixel_values", shape=pixel_values.shape),
            ct.TensorType(name="image_grid_thw", shape=image_grid_thw.shape),
        ],
        outputs=[ct.TensorType(name="vision_embeddings")],
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    coreml_model.save(OUTPUT_PATH)
    print(f"CoreML model saved to {OUTPUT_PATH}")

    # Validate CoreML output
    print("Validating CoreML output...")
    coreml_out = coreml_model.predict({
        "pixel_values": pixel_values.numpy(),
        "image_grid_thw": image_grid_thw.numpy(),
    })
    coreml_embeddings = coreml_out["vision_embeddings"]
    ref_np = ref_output.numpy()

    max_diff = np.abs(ref_np - coreml_embeddings).max()
    mean_diff = np.abs(ref_np - coreml_embeddings).mean()
    print(f"CoreML validation: max_diff={max_diff:.6f}, mean_diff={mean_diff:.6f}")
    print(f"Shape match: {ref_np.shape} == {coreml_embeddings.shape}: {ref_np.shape == coreml_embeddings.shape}")

    if max_diff < 0.05:
        print("PASS: CoreML output matches PyTorch within tolerance")
    else:
        print(f"WARNING: max_diff={max_diff:.6f} exceeds threshold 0.05")


if __name__ == "__main__":
    main()
