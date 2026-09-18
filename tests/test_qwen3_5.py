import unittest

import torch
from nemo_automodel import NeMoAutoModelForCausalLM
from nemo_automodel._transformers.registry import ModelRegistry
from nemo_automodel.components.distributed.parallelizer import (
    Qwen3_5ParallelizationStrategy,
    get_parallelization_strategy,
)
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig

from picoformer.models.qwen3_5 import (
    PicoformerAutoModelForCausalLM,
    PicoformerQwen3_5ForCausalLM,
)


class Qwen35CompatibilityTest(unittest.TestCase):
    def test_project_factory_is_recognized_as_nemo_auto_model(self):
        self.assertIs(PicoformerAutoModelForCausalLM, NeMoAutoModelForCausalLM)

    def test_project_subclass_is_registered_with_nemo(self):
        config = Qwen3_5TextConfig(architectures=["Qwen3_5ForCausalLM"])
        self.assertIs(
            ModelRegistry.resolve_custom_model_cls("Qwen3_5ForCausalLM", config),
            PicoformerQwen3_5ForCausalLM,
        )

    def test_project_subclass_uses_qwen_fsdp_strategy(self):
        config = Qwen3_5TextConfig(
            architectures=["Qwen3_5ForCausalLM"],
            vocab_size=128,
            hidden_size=36,
            intermediate_size=72,
            num_hidden_layers=1,
            num_attention_heads=9,
            num_key_value_heads=3,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=9,
            linear_num_value_heads=9,
        )
        model = PicoformerQwen3_5ForCausalLM(config)

        self.assertIsInstance(
            get_parallelization_strategy(model), Qwen3_5ParallelizationStrategy
        )

    def test_initializer_uses_configured_embedding_std(self):
        config = Qwen3_5TextConfig(
            architectures=["Qwen3_5ForCausalLM"],
            vocab_size=20_000,
            hidden_size=36,
            intermediate_size=72,
            num_hidden_layers=1,
            num_attention_heads=9,
            num_key_value_heads=3,
            head_dim=4,
            linear_key_head_dim=4,
            linear_value_head_dim=4,
            linear_num_key_heads=9,
            linear_num_value_heads=9,
            initializer_range=0.02,
            tie_word_embeddings=True,
        )
        model = PicoformerQwen3_5ForCausalLM(config)
        model.initialize_weights(buffer_device=torch.device("cpu"), dtype=torch.float32)

        embedding = model.get_input_embeddings().weight
        self.assertIs(embedding, model.get_output_embeddings().weight)
        self.assertTrue(torch.isclose(embedding.std(), torch.tensor(0.02), atol=1e-4))


if __name__ == "__main__":
    unittest.main()
