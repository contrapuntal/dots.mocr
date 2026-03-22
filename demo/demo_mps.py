"""
dots.mocr inference on Apple Silicon (MPS backend).

Changes from demo_hf.py:
  - Uses SDPA attention instead of flash_attention_2
  - Disables sliding window attention (not implemented for SDPA on MPS)
  - Uses float16 (vision tower auto-skips bf16 cast on MPS)
  - Sends inputs to "mps" instead of "cuda"
  - Caps processor max_pixels to avoid OOM on large images
"""
import os

if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from dots_mocr.utils import dict_promptmode_to_prompt

# Cap at 4M pixels. The vision tower's 42 attention layers + model weights
# + LLM KV cache exceed unified memory at higher resolutions.
# The processor's max_pixels parameter enforces this at tokenization time.
MPS_MAX_PIXELS = 4_000_000


def inference(image_path, prompt, model, processor, device="mps", dtype=torch.float16):
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image_path},
                {"type": "text", "text": prompt},
            ],
        }
    ]

    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(
        text=[text],
        images=image_inputs,
        videos=video_inputs,
        padding=True,
        return_tensors="pt",
    )

    for k, v in list(inputs.items()):
        if isinstance(v, torch.Tensor):
            v = v.to(device)
            if torch.is_floating_point(v):
                v = v.to(dtype=dtype)
            inputs[k] = v
    inputs.pop("mm_token_type_ids", None)

    generated_ids = model.generate(**inputs, max_new_tokens=4096)
    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return output_text[0]


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"Using device: {device}, dtype: {dtype}")

    model_path = "./weights/DotsMOCR"
    print(f"Loading model from {model_path}...")

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    if getattr(config, "vision_config", None) is not None:
        config.vision_config.attn_implementation = "sdpa"
    config.use_sliding_window = False
    config.sliding_window = None

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        config=config,
        attn_implementation="sdpa",
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    processor = AutoProcessor.from_pretrained(
        model_path, trust_remote_code=True,
        max_pixels=MPS_MAX_PIXELS,
    )
    print("Model loaded.")

    image_path = "demo/demo_image1.jpg"
    prompt_mode = "prompt_layout_all_en"
    prompt = dict_promptmode_to_prompt[prompt_mode]
    print(f"\n--- {prompt_mode} ---")
    print(f"Image: {image_path}")
    result = inference(image_path, prompt, model, processor, device=device, dtype=dtype)
    print(f"\nResult:\n{result[:2000]}...")
