# Copyright 2025 Dustin Loring
# This file is part of GPT-OSS-Vision-20B.
# 
# Based on the original GPT-OSS test suite from Hugging Face & OpenAI's GPT-OSS.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
#
# Changes:
# - Adapted to support GPT-OSS-Vision-20B
# - Added multimodal (image+text) test cases
# - Added NoPE (No Positional Embedding in periodic layers) config testing
# - Updated model references to AIGym/gpt_oss_vision_20B
# - Updated contact: Dustin Loring <Dustinwloring1988@gmail.com>
"""Testing suite for the PyTorch GPT-OSS-Vision-20B model (multimodal + NoPE)."""

import inspect
import json
import os
import subprocess
import tempfile
import unittest
from pathlib import Path

import pytest
from parameterized import parameterized

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    is_torch_available,
)
from transformers.testing_utils import (
    cleanup,
    require_read_token,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)

from ...causal_lm_tester import CausalLMModelTest, CausalLMModelTester
from ...test_configuration_common import ConfigTester

if is_torch_available():
    import torch
    from transformers import (
        GPTOSSVisionConfig,
        GPTOSSVisionModel,
        GPTOSSVisionForCausalLM,
    )
    NUM_GPUS = torch.cuda.device_count()


class GPTOSSVisionModelTester(CausalLMModelTester):
    if is_torch_available():
        config_class = GPTOSSVisionConfig
        base_model_class = GPTOSSVisionModel
        causal_lm_class = GPTOSSVisionForCausalLM

    pipeline_model_mapping = (
        {
            "feature-extraction": GPTOSSVisionModel,
            "text-generation": GPTOSSVisionForCausalLM,
        }
        if is_torch_available()
        else {}
    )


@require_torch
class GPTOSSVisionModelTest(CausalLMModelTest, unittest.TestCase):
    """Unit tests for GPT-OSS-Vision-20B"""

    all_model_classes = (GPTOSSVisionModel, GPTOSSVisionForCausalLM) if is_torch_available() else ()
    pipeline_model_mapping = (
        {
            "feature-extraction": GPTOSSVisionModel,
            "text-generation": GPTOSSVisionForCausalLM,
        }
        if is_torch_available()
        else {}
    )

    test_headmasking = False
    test_pruning = False
    _is_stateful = True
    model_split_percents = [0.5, 0.6]
    model_tester_class = GPTOSSVisionModelTester

    def setUp(self):
        self.model_tester = GPTOSSVisionModelTester(self)
        self.config_tester = ConfigTester(
            self,
            config_class=GPTOSSVisionConfig,
            hidden_size=37,
            vision_embed_dim=64,
            use_nope=True,
        )

    @unittest.skip("HybridCache limitations remain from GPT-OSS base")
    def test_model_outputs_equivalence(self, **kwargs):
        pass

    @pytest.mark.parametrize("use_image", [False, True])
    def test_forward_with_optional_image(self, use_image):
        """Ensure model runs with text-only or multimodal inputs."""
        from PIL import Image
        import io

        model = GPTOSSVisionForCausalLM(self.config_tester.config)
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

        inputs = tokenizer("Roses are red,", return_tensors="pt")
        if use_image:
            # Create dummy white image
            image = Image.new("RGB", (224, 224), color="white")
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            buf.seek(0)
            inputs["pixel_values"] = torch.randn(1, 3, 224, 224)  # Simulate vision encoder output

        outputs = model(**inputs)
        self.assertIsNotNone(outputs.logits)

    def test_config_nope_enabled(self):
        """Check NoPE parameter toggles."""
        config = GPTOSSVisionConfig(use_nope=True, nope_stride=4)
        self.assertTrue(config.use_nope)
        self.assertEqual(config.nope_stride, 4)


# Integration tests would remain similar to original GPT-OSS tests,
# but model_id updated to point to Dustin Loring's AIGym repo.

RESULTS_PATH = Path(__file__).parent.parent.parent / "fixtures/gpt_oss_vision/integration_tests.json"


def distributed_worker(quantized, model_size, kernels, attn_impl, mode):
    """Torchrun worker for distributed GPT-OSS-Vision-20B inference."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from transformers.testing_utils import torch_device

    input_text = [
        "Roses are red, violets",
        "Describe the image in one sentence:",
    ]
    quantized = quantized.lower() == "true"
    kernels = kernels.lower() == "true"

    model_id = f"AIGym/gpt_oss_vision_{model_size}"
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        tp_plan="auto",
        use_kernels=kernels,
    ).to(torch_device)
    model.set_attn_implementation(attn_impl)
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

    inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(torch_device)
    output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
    output_texts = tokenizer.batch_decode(output, skip_special_tokens=False)

    if int(os.environ.get("RANK", "0")) == 0:
        result_entry = {
            "quantized": quantized,
            "model": model_size,
            "kernels": kernels,
            "attn_impl": attn_impl,
            "mode": mode,
            "outputs": output_texts,
        }
        if os.path.exists(RESULTS_PATH):
            with open(RESULTS_PATH, "r") as f:
                results = json.load(f)
        else:
            results = []
        results.append(result_entry)
        with open(RESULTS_PATH, "w") as f:
            json.dump(results, f, indent=2)


@slow
@require_torch_accelerator
class GPTOSSVisionIntegrationTest(unittest.TestCase):
    input_text = [
        "Roses are red, violets",
        "Describe the image in one sentence:",
    ]

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    PARAMETERS = [
        (False, "20B", False, "eager", "eval"),
        (True, "20B", True, "eager", "eval"),
    ]

    @parameterized.expand(PARAMETERS)
    @require_read_token
    def test_model_outputs(self, quantized, model, kernels, attn_impl, mode):
        model_id = f"AIGym/gpt_oss_vision_{model}"
        output_texts = self.load_and_forward(model_id, attn_impl, self.input_text, use_kernels=kernels)
        self.assertTrue(all(isinstance(x, str) for x in output_texts))

    @staticmethod
    def load_and_forward(model_id, attn_implementation, input_text, **kwargs):
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation=attn_implementation,
            **kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")
        inputs = tokenizer(input_text, return_tensors="pt", padding=True).to(model.device)
        output = model.generate(**inputs, max_new_tokens=20, do_sample=False)
        return tokenizer.batch_decode(output, skip_special_tokens=False)
