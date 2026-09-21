"""Stream a training checkpoint's next 512 tokens using only the CPU."""

from __future__ import annotations

import argparse
from contextlib import nullcontext
import math
import os
from pathlib import Path
import sys
from unittest.mock import patch

from picoformer.consolidate_checkpoint import consolidate_checkpoint, is_consolidated_model


class PresencePenaltyLogitsProcessor:
    """Penalize each previously generated token once, excluding the prompt."""

    def __init__(self, penalty: float, prompt_length: int):
        self.penalty = penalty
        self.prompt_length = prompt_length

    def __call__(self, input_ids, scores):
        generated_ids = input_ids[:, self.prompt_length:]
        return scores.scatter(
            1, generated_ids, scores.gather(1, generated_ids) - self.penalty
        )


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
    parser.add_argument("--prompt", default="Today the weather forecast is")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    # Qwen3.5's recommended general thinking-mode sampling parameters.
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="sampling temperature; 0 selects greedy decoding (default: 1.0)")
    parser.add_argument("--top-p", type=float, default=0.95,
                        help="nucleus sampling probability (default: 0.95)")
    parser.add_argument("--top-k", type=int, default=20,
                        help="top-k sampling cutoff; 0 disables (default: 20)")
    parser.add_argument("--min-p", type=float, default=0.0,
                        help="minimum probability relative to the most likely token (default: 0.0)")
    parser.add_argument("--presence-penalty", type=float, default=1.5,
                        help="additive penalty for previously generated tokens (default: 1.5)")
    parser.add_argument("--repetition-penalty", type=float, default=1.0,
                        help="multiplicative repetition penalty; 1 disables (default: 1.0)")
    parser.add_argument(
        "--tokenizer", default=None,
        help="tokenizer directory or HF ID (defaults to checkpoint tokenizer, then SmolLM2-135M)",
    )
    parser.add_argument("--num-threads", type=int, default=4, help="CPU threads (default: 4)")
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.num_threads < 1:
        parser.error("--max-new-tokens and --num-threads must be at least 1")
    if not math.isfinite(args.temperature) or args.temperature < 0:
        parser.error("--temperature must be finite and nonnegative")
    if not 0 < args.top_p <= 1:
        parser.error("--top-p must be in (0, 1]")
    if args.top_k < 0:
        parser.error("--top-k must be nonnegative")
    if not 0 <= args.min_p <= 1:
        parser.error("--min-p must be in [0, 1]")
    if not math.isfinite(args.presence_penalty) or not -2 <= args.presence_penalty <= 2:
        parser.error("--presence-penalty must be in [-2, 2]")
    if not math.isfinite(args.repetition_penalty) or args.repetition_penalty <= 0:
        parser.error("--repetition-penalty must be finite and positive")
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
    sampling = (
        dict(temperature=args.temperature, top_p=args.top_p,
             top_k=args.top_k, min_p=args.min_p)
        if args.temperature > 0 else {}
    )
    generation_config = GenerationConfig(
        max_new_tokens=args.max_new_tokens,
        min_new_tokens=args.max_new_tokens,
        do_sample=args.temperature > 0,
        repetition_penalty=args.repetition_penalty,
        **sampling,
        use_cache=True,
        eos_token_id=None,
        pad_token_id=tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0,
    )
    # Transformers does not provide a native presence-penalty processor.
    logits_processor = []
    if args.presence_penalty:
        logits_processor.append(PresencePenaltyLogitsProcessor(
            args.presence_penalty, inputs.input_ids.shape[1]
        ))
    print(f"Generating {args.max_new_tokens} tokens…", file=sys.stderr, flush=True)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            generation_config=generation_config,
            logits_processor=logits_processor,
            streamer=TextStreamer(tokenizer, skip_special_tokens=True),
        )
    print(f"Generated {output.shape[1] - inputs.input_ids.shape[1]} tokens on CPU.", file=sys.stderr)


if __name__ == "__main__":
    main()
