import contextlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast

from picoformer.vibe_check import PresencePenaltyLogitsProcessor, main


class VibeCheckTest(unittest.TestCase):
    def test_presence_penalty_excludes_prompt_and_counts_each_token_once(self):
        processor = PresencePenaltyLogitsProcessor(1.5, prompt_length=2)
        scores = torch.tensor([[1., 2., 3., 4.], [4., 3., 2., 1.]])
        ids = torch.tensor([[0, 1, 2, 2], [2, 3, 1, 1]])
        torch.testing.assert_close(
            processor(ids, scores),
            torch.tensor([[1., 2., 1.5, 4.], [4., 1.5, 2., 1.]]),
        )
        torch.testing.assert_close(processor(ids[:, :2], scores), scores)

    def test_cli_generates_with_defaults_and_overrides(self):
        tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0, "hello": 1, "world": 2})),
            unk_token="[UNK]", pad_token="[UNK]",
        )
        model = GPT2LMHeadModel(GPT2Config(
            vocab_size=3, n_positions=16, n_embd=8, n_layer=1, n_head=1,
            bos_token_id=None, eos_token_id=None, pad_token_id=0,
        )).eval()
        cases = [
            ([], True, 1.0, 0.95, 20, 0.0, 1.5, 1.0),
            (["--temperature", "0.7", "--top-p", "0.8", "--top-k", "2",
              "--min-p", "0.1", "--presence-penalty", "0.5",
              "--repetition-penalty", "1.2"], True, 0.7, 0.8, 2, 0.1, 0.5, 1.2),
            (["--temperature", "0", "--presence-penalty", "0"],
             False, None, None, None, None, 0.0, 1.0),
        ]
        with tempfile.TemporaryDirectory() as directory:
            tokenizer.save_pretrained(directory)
            model.save_pretrained(directory)
            for flags, sample, temperature, top_p, top_k, min_p, presence, repetition in cases:
                with self.subTest(flags=flags), contextlib.ExitStack() as stack:
                    stack.enter_context(patch("sys.argv", [
                        "vibe_check", directory, "--prompt", "hello", "--max-new-tokens", "3",
                        *flags,
                    ]))
                    stack.enter_context(patch("torch.set_num_interop_threads"))
                    stack.enter_context(patch("picoformer.vibe_check.load_cpu_model", return_value=model))
                    generate = stack.enter_context(patch.object(model, "generate", wraps=model.generate))
                    stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
                    stderr = stack.enter_context(contextlib.redirect_stderr(io.StringIO()))
                    main()
                    self.assertIn("Generated 3 tokens on CPU.", stderr.getvalue())
                    config = generate.call_args.kwargs["generation_config"]
                    self.assertEqual(config.do_sample, sample)
                    self.assertEqual(config.repetition_penalty, repetition)
                    if sample:
                        self.assertEqual(
                            (config.temperature, config.top_p, config.top_k, config.min_p),
                            (temperature, top_p, top_k, min_p),
                        )
                    processors = generate.call_args.kwargs["logits_processor"]
                    if presence:
                        self.assertEqual(processors[0].penalty, presence)
                        self.assertEqual(processors[0].prompt_length, 1)
                    else:
                        self.assertEqual(processors, [])

    def test_invalid_decoding_parameters_fail_before_loading(self):
        cases = [("--temperature", "-1"), ("--temperature", "nan"),
                 ("--top-p", "0"), ("--top-p", "1.1"), ("--top-k", "-1"),
                 ("--min-p", "nan"), ("--min-p", "1.1"),
                 ("--presence-penalty", "inf"), ("--presence-penalty", "3"),
                 ("--repetition-penalty", "0"), ("--repetition-penalty", "nan")]
        for flag, value in cases:
            with self.subTest(flag=flag, value=value), \
                    patch("sys.argv", ["vibe_check", str(Path.cwd()), flag, value]), \
                    contextlib.redirect_stderr(io.StringIO()), \
                    self.assertRaises(SystemExit) as error:
                main()
            self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
