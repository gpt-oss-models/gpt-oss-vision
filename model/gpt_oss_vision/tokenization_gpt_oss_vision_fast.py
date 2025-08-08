# coding=utf-8
# Copyright 2025 Dustin Loring
# 
# Based on the original GPT-OSS fast tokenization from Hugging Face & OpenAI's GPT-OSS.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes:
# - Adapted for GPT-OSS-Vision multimodal support
# - Added vision token handling capabilities
# - Contact: Dustin Loring <Dustinwloring1988@gmail.com>
"""Fast Tokenization classes for GPT-OSS-Vision."""

import json
from typing import List, Optional, Tuple

from tokenizers import pre_tokenizers, processors

from ...tokenization_utils_base import AddedToken
from ...tokenization_utils_fast import PreTrainedTokenizerFast
from ...utils import logging

logger = logging.get_logger(__name__)

VOCAB_FILES_NAMES = {"tokenizer_file": "tokenizer.json"}

PRETRAINED_VOCAB_FILES_MAP = {
    "AIGym/gpt_oss_vision_20B": {
        "tokenizer_file": "https://huggingface.co/AIGym/gpt_oss_vision_20B/resolve/main/tokenizer.json",
    },
}

PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES = {
    "AIGym/gpt_oss_vision_20B": 131072,
}


class GPTOSSVisionTokenizerFast(PreTrainedTokenizerFast):
    """
    Construct a "fast" GPT-OSS-Vision tokenizer (backed by HuggingFace's *tokenizers* library). Based on byte-level
    Byte-Pair-Encoding.

    This tokenizer inherits from [`PreTrainedTokenizerFast`] which contains most of the main methods. Users should
    refer to this superclass for more information regarding those methods.

    Args:
        tokenizer_object (`tokenizers.Tokenizer`):
            An instantiated instance of a built-in tokenizer class from the `tokenizers` library.
        unk_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The unknown token. A token that is not in the vocabulary cannot be converted to an ID and is set to be this
            token instead.
        bos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The beginning of sequence token that was used during pretraining. Can be used a sequence classifier token.
        eos_token (`str`, *optional*, defaults to `<|endoftext|>`):
            The end of sequence token.
        pad_token (`str`, *optional*, defaults to `None`):
            The token used for padding, for example when batching sequences of different lengths.
        add_prefix_space (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial space to the input. This allows to treat the leading word just as any
            other word. (GPT-OSS tokenizer detect beginning of words by the preceding space).
        add_bos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add an initial <|endoftext|> token to the input. This allows to treat the leading word just as any other word.
        add_eos_token (`bool`, *optional*, defaults to `False`):
            Whether or not to add a final <|endoftext|> token to the input. This allows to treat the leading word just as any other word.
        trim_offsets (`bool`, *optional*, defaults to `True`):
            Whether or not the post-processing step should trim offsets to avoid including whitespace.
    """

    vocab_files_names = VOCAB_FILES_NAMES
    pretrained_vocab_files_map = PRETRAINED_VOCAB_FILES_MAP
    max_model_input_sizes = PRETRAINED_POSITIONAL_EMBEDDINGS_SIZES
    model_input_names = ["input_ids", "attention_mask"]

    def __init__(
        self,
        tokenizer_object=None,
        unk_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        pad_token=None,
        add_prefix_space=False,
        add_bos_token=False,
        add_eos_token=False,
        trim_offsets=True,
        **kwargs,
    ):
        bos_token = AddedToken(bos_token, lstrip=False, rstrip=False) if isinstance(bos_token, str) else bos_token
        eos_token = AddedToken(eos_token, lstrip=False, rstrip=False) if isinstance(eos_token, str) else eos_token
        unk_token = AddedToken(unk_token, lstrip=False, rstrip=False) if isinstance(unk_token, str) else unk_token
        pad_token = AddedToken(pad_token, lstrip=False, rstrip=False) if isinstance(pad_token, str) else pad_token

        super().__init__(
            tokenizer_object=tokenizer_object,
            unk_token=unk_token,
            bos_token=bos_token,
            eos_token=eos_token,
            pad_token=pad_token,
            add_prefix_space=add_prefix_space,
            add_bos_token=add_bos_token,
            add_eos_token=add_eos_token,
            trim_offsets=trim_offsets,
            **kwargs,
        )

    def save_pretrained(self, save_directory, **kwargs):
        """Save the tokenizer to a directory."""
        super().save_pretrained(save_directory, **kwargs)

    def decode(
        self,
        token_ids,
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: bool = None,
        **kwargs,
    ) -> str:
        return super().decode(
            token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def prepare_for_tokenization(self, text, is_split_into_words=False, **kwargs):
        add_prefix_space = kwargs.pop("add_prefix_space", self.add_prefix_space)
        if is_split_into_words or add_prefix_space:
            text = " " + text
        return (text, kwargs)
