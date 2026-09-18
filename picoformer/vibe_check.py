"""Stream a training checkpoint's next 512 tokens using only the CPU."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import os
from pathlib import Path
import sys
from unittest.mock import patch

from picoformer.consolidate_checkpoint import consolidate_checkpoint, is_consolidated_model


def load_cpu_model(model_dir: Path):
    """Load float32 weights, including Qwen3.5's CPU-compatible operators."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained(model_dir, local_files_only=True)
    cpu_operators = nullcontext()
    if config.model_type == "qwen3_5_text":
        from transformers.models.qwen3_5 import modeling_qwen3_5

        # Installed FLA/causal-conv1d kernels are CUDA-only; even constructing
        # FusedRMSNormGated can initialize CUDA. Patch before model construction.
        # Each layer retains the fallback callables after this context exits.
        cpu_operators = patch.multiple(
            modeling_qwen3_5,
            FusedRMSNormGated=None,
            causal_conv1d_fn=None,
            causal_conv1d_update=None,
            chunk_gated_delta_rule=None,
            fused_recurrent_gated_delta_rule=None,
            is_fast_path_available=False,
        )
    with cpu_operators:
        model = AutoModelForCausalLM.from_pretrained(
            model_dir,
            config=config,
            local_files_only=True,
            dtype=torch.float32,
            device_map="cpu",
            attn_implementation="eager",
        )
    return model.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "checkpoint", type=Path,
        help="saved epoch_<N>_step_<N> directory or consolidated Hugging Face model",
    )
    parser.add_argument("--prompt", default="Hi, my name is Alex. Your name is")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--tokenizer", default=None,
        help="tokenizer directory or HF ID (defaults to checkpoint tokenizer, then SmolLM2-135M)",
    )
    parser.add_argument("--num-threads", type=int, default=4, help="CPU threads (default: 4)")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.num_threads < 1:
        parser.error("--max-new-tokens and --num-threads must be at least 1")
    if not args.prompt:
        parser.error("--prompt must not be empty")
    checkpoint = args.checkpoint.expanduser().resolve()
    if not checkpoint.is_dir():
        parser.error(f"checkpoint directory not found: {checkpoint}")

    # Hide GPUs before importing torch, transformers, or the NeMo converter.
    # This only affects this process, so an ongoing training job is untouched.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    import torch
    from transformers import AutoTokenizer, GenerationConfig, TextStreamer

    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(1)
    if is_consolidated_model(checkpoint):
        model_dir = checkpoint
    else:
        print(f"Preparing checkpoint: {checkpoint}", file=sys.stderr, flush=True)
        model_dir = consolidate_checkpoint(checkpoint, num_threads=args.num_threads)
    tokenizer_source = args.tokenizer or (
        str(model_dir) if (model_dir / "tokenizer_config.json").is_file()
        else "HuggingFaceTB/SmolLM2-135M"
    )
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)
    print(f"Loading on CPU (float32): {model_dir}", file=sys.stderr, flush=True)
    model = load_cpu_model(model_dir)
    inputs = tokenizer(args.prompt, return_tensors="pt", add_special_tokens=False).to("cpu")
    if inputs.input_ids.numel() == 0:
        parser.error("--prompt must encode to at least one token")

    # Keep predicting through EOS to produce the requested number of tokens.
    # A fresh config avoids inheriting sampling/stopping rules from training.
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.max_new_tokens,
        do_sample=False,
        use_cache=True,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    print(f"Generating {args.max_new_tokens} tokens…", file=sys.stderr, flush=True)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
            streamer=TextStreamer(tokenizer, skip_special_tokens=True),
        )
    print(f"Generated {output.shape[1] - inputs.input_ids.shape[1]} tokens on CPU.", file=sys.stderr)


if __name__ == "__main__":
    main()
