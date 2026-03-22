"""
Smoke test of all dots.mocr prompt modes on Apple Silicon (MPS).
Loads model once, runs each mode with an appropriate test image.
Reports PASS/FAIL (non-exception) and flags outputs that hit the token cap.

Uses SDPA attention + fp16 for MPS compatibility and memory efficiency.
This is a smoke/perf test, not a correctness validation.
"""
import os
import sys
import time

if "LOCAL_RANK" not in os.environ:
    os.environ["LOCAL_RANK"] = "0"

import torch
from transformers import AutoModelForCausalLM, AutoProcessor, AutoConfig
from qwen_vl_utils import process_vision_info
from PIL import Image
from dots_mocr.utils import dict_promptmode_to_prompt

# Cap at 4M pixels. The vision tower's 42 attention layers + model weights
# + LLM KV cache exceed unified memory at higher resolutions.
MPS_MAX_PIXELS = 4_000_000

MAX_NEW_TOKENS = 2048


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

    t0 = time.time()
    generated_ids = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    elapsed = time.time() - t0

    generated_ids_trimmed = [
        out_ids[len(in_ids):]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    num_tokens = generated_ids_trimmed[0].shape[0]
    return output_text[0], elapsed, num_tokens


TEST_CASES = [
    {
        "mode": "prompt_layout_all_en",
        "image": "demo/demo_image1.jpg",
        "description": "Full layout parsing (bbox + category + text)",
    },
    {
        "mode": "prompt_layout_only_en",
        "image": "demo/demo_image1.jpg",
        "description": "Layout detection only (bbox + category, no text)",
    },
    {
        "mode": "prompt_ocr",
        "image": "demo/demo_image1.jpg",
        "description": "Pure text extraction",
    },
    {
        "mode": "prompt_grounding_ocr",
        "image": "demo/demo_image1.jpg",
        "description": "Extract text from specific bounding box",
        "bbox": [145, 235, 1565, 705],
    },
    {
        "mode": "prompt_web_parsing",
        "image": "assets/showcase/origin/webpage_1.png",
        "description": "Webpage layout parsing",
    },
    {
        "mode": "prompt_scene_spotting",
        "image": "assets/showcase/origin/scene_1.jpg",
        "description": "Scene text detection",
    },
    {
        "mode": "prompt_image_to_svg",
        "image": "assets/showcase/origin/svg_1.png",
        "description": "Image to SVG code generation",
    },
    {
        "mode": "prompt_general",
        "image": "demo/demo_image1.jpg",
        "description": "General visual QA",
        "custom_prompt": "What type of document is this? Describe its content briefly.",
    },
]


if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    dtype = torch.float16 if device == "mps" else torch.float32
    print(f"Device: {device}, dtype: {dtype}")
    print(f"PyTorch: {torch.__version__}")
    print()

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
    print("Model loaded.\n")

    results = []
    for tc in TEST_CASES:
        mode = tc["mode"]
        image_path = tc["image"]
        desc = tc["description"]

        prompt = dict_promptmode_to_prompt[mode]

        if mode == "prompt_grounding_ocr":
            prompt = prompt + str(tc["bbox"])
        elif mode == "prompt_image_to_svg":
            with Image.open(image_path) as img:
                prompt = prompt.replace("{width}", str(img.width))
                prompt = prompt.replace("{height}", str(img.height))
        elif mode == "prompt_general":
            prompt = tc.get("custom_prompt", prompt)

        print(f"{'='*60}")
        print(f"MODE: {mode}")
        print(f"DESC: {desc}")
        print(f"IMAGE: {image_path}")
        print(f"{'='*60}")

        try:
            output, elapsed, num_tokens = inference(
                image_path, prompt, model, processor, device=device, dtype=dtype
            )
            status = "PASS"
            hit_cap = num_tokens >= MAX_NEW_TOKENS
            preview = output[:500].replace("\n", "\\n")
            print(f"STATUS: {status}{'  [HIT_TOKEN_CAP]' if hit_cap else ''}")
            print(f"TIME: {elapsed:.1f}s | TOKENS: {num_tokens} | SPEED: {num_tokens/elapsed:.1f} tok/s")
            print(f"OUTPUT: {preview}...")
        except Exception as e:
            status = "FAIL"
            elapsed = 0
            num_tokens = 0
            hit_cap = False
            print(f"STATUS: {status}")
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()

        results.append({
            "mode": mode, "status": status,
            "time": elapsed, "tokens": num_tokens,
            "hit_cap": hit_cap,
        })
        print()

        if device == "mps":
            torch.mps.empty_cache()

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY (smoke test)")
    print(f"{'='*60}")
    passed = sum(1 for r in results if r["status"] == "PASS")
    total = len(results)
    for r in results:
        tok_speed = f"{r['tokens']/r['time']:.1f} tok/s" if r["time"] > 0 else "N/A"
        cap = " [HIT_TOKEN_CAP]" if r.get("hit_cap") else ""
        print(f"  [{r['status']}] {r['mode']:30s} {r['time']:6.1f}s  {r['tokens']:5d} tokens  {tok_speed}{cap}")
    print(f"\nResult: {passed}/{total} passed")
    sys.exit(0 if passed == total else 1)
