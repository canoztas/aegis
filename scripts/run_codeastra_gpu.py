#!/usr/bin/env python3
"""
Run CodeAstra-7B (LoRA adapter) on GPU using PEFT + Transformers.
"""

import argparse
import json
import os
import sys
from typing import Optional


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run CodeAstra-7B with GPU")
    parser.add_argument(
        "--adapter",
        default="rootxhacker/CodeAstra-7B",
        help="HF adapter (LoRA) repo id",
    )
    parser.add_argument(
        "--base-model",
        default=None,
        help="Base model repo id (defaults to adapter's base)",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help="Prompt text (if omitted, uses --file or a default prompt)",
    )
    parser.add_argument(
        "--file",
        default=None,
        help="Path to a file containing code to analyze",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=512,
        help="Max new tokens to generate",
    )
    parser.add_argument(
        "--min-new-tokens",
        type=int,
        default=16,
        help="Min new tokens to generate (set 0 to disable)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling p",
    )
    parser.add_argument(
        "--do-sample",
        action="store_true",
        help="Enable sampling (default: greedy)",
    )
    parser.add_argument(
        "--device-map",
        default="auto",
        help="Transformers device_map (auto|sequential|balanced|balanced_low_0|none)",
    )
    parser.add_argument(
        "--load-4bit",
        action="store_true",
        help="Load base model in 4-bit (requires bitsandbytes + GPU)",
    )
    parser.add_argument(
        "--load-8bit",
        action="store_true",
        help="Load base model in 8-bit (requires bitsandbytes + GPU)",
    )
    parser.add_argument(
        "--dtype",
        default="bf16",
        choices=["bf16", "fp16", "fp32"],
        help="Compute dtype for model",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Write output to file instead of stdout",
    )
    parser.add_argument(
        "--show-full-text",
        action="store_true",
        help="Print full prompt + completion (default: completion only)",
    )
    return parser.parse_args()


def _read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as handle:
        return handle.read()


def _build_prompt(code: str) -> str:
    return (
        "Analyze the following code for security vulnerabilities.\n"
        "Return ONLY valid JSON matching exactly this structure:\n"
        "{\n"
        "  \"findings\": [\n"
        "    {\n"
        "      \"file_path\": \"sample.py\",\n"
        "      \"line_start\": <number>,\n"
        "      \"line_end\": <number>,\n"
        "      \"snippet\": \"<code snippet>\",\n"
        "      \"cwe\": \"<CWE-id or null>\",\n"
        "      \"severity\": \"critical|high|medium|low|info\",\n"
        "      \"confidence\": <0.0-1.0>,\n"
        "      \"title\": \"<short title>\",\n"
        "      \"category\": \"<vulnerability type/category>\",\n"
        "      \"description\": \"<detailed explanation>\",\n"
        "      \"recommendation\": \"<how to fix>\"\n"
        "    }\n"
        "  ]\n"
        "}\n\n"
        "Code to analyze:\n"
        "```\n"
        f"{code}\n"
        "```\n\n"
        "Return only the JSON payload. No prose."
    )


def _dtype_from_arg(dtype: str):
    import torch

    if dtype == "bf16":
        return torch.bfloat16
    if dtype == "fp16":
        return torch.float16
    return torch.float32


def _has_bitsandbytes() -> bool:
    try:
        import bitsandbytes  # noqa: F401
        return True
    except Exception:
        return False


def _resolve_base_model(adapter_id: str, base_model: Optional[str]) -> str:
    if base_model:
        return base_model
    from peft import PeftConfig

    cfg = PeftConfig.from_pretrained(adapter_id)
    return cfg.base_model_name_or_path


def main() -> int:
    args = _parse_args()

    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import PeftModel
    except Exception as exc:
        print(f"[error] Missing dependencies: {exc}")
        print("Install: pip install torch transformers peft accelerate bitsandbytes")
        return 1

    if not torch.cuda.is_available():
        print("[warn] CUDA not available. This script is intended for GPU use.")
        print("       If you want CPU, rerun without GPU expectations.")

    if args.load_4bit or args.load_8bit:
        if not _has_bitsandbytes():
            print("[error] bitsandbytes is required for 4-bit/8-bit loading.")
            print("Install: pip install bitsandbytes")
            return 1

    adapter_id = args.adapter
    base_model_id = _resolve_base_model(adapter_id, args.base_model)

    prompt = args.prompt
    if not prompt and args.file:
        code = _read_file(args.file)
        prompt = _build_prompt(code)
    if not prompt:
        sample = "def add(a, b):\n    return a + b"
        prompt = _build_prompt(sample)

    device_map = None if args.device_map == "none" else args.device_map
    dtype = _dtype_from_arg(args.dtype)

    model_kwargs = {
        "device_map": device_map,
        "torch_dtype": dtype,
        "trust_remote_code": True,
        "return_dict": True,
    }
    if args.load_4bit:
        model_kwargs["load_in_4bit"] = True
    if args.load_8bit:
        model_kwargs["load_in_8bit"] = True

    print("[info] Loading base model:", base_model_id)
    base_model = AutoModelForCausalLM.from_pretrained(base_model_id, **model_kwargs)

    print("[info] Loading adapter:", adapter_id)
    model = PeftModel.from_pretrained(base_model, adapter_id)

    tokenizer = AutoTokenizer.from_pretrained(base_model_id, trust_remote_code=True)
    model.eval()

    inputs = tokenizer(prompt, return_tensors="pt")
    if device_map is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}

    generate_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "do_sample": bool(args.do_sample),
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.eos_token_id,
    }
    if args.min_new_tokens and args.min_new_tokens > 0:
        generate_kwargs["min_new_tokens"] = args.min_new_tokens

    print("[info] Generating...")
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **generate_kwargs)

    input_len = inputs["input_ids"].shape[-1]
    if args.show_full_text:
        text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    else:
        generated = output_ids[0][input_len:]
        text = tokenizer.decode(generated, skip_special_tokens=True)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as handle:
            handle.write(text)
        print(f"[info] Wrote output to {args.output}")
    else:
        print(text)

    return 0


if __name__ == "__main__":
    sys.exit(main())
