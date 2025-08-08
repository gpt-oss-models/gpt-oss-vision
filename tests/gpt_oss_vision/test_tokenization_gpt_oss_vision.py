# coding=utf-8
# Copyright 2025 Dustin Loring
# 
# Based on the original GPT-OSS tokenization tests from Hugging Face & OpenAI's GPT-OSS.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes:
# - Adapted for GPT-OSS-Vision multimodal support
# - Added vision token handling test cases
# - Contact: Dustin Loring <Dustinwloring1988@gmail.com>
"""Tests for GPT-OSS-Vision tokenizer."""

import json
import os
import tempfile
import unittest

from transformers import GPTOSSVisionTokenizer, GPTOSSVisionTokenizerFast
from transformers.testing_utils import require_tokenizers, slow

from ...test_tokenization_common import TokenizerTesterMixin


@require_tokenizers
class GPTOSSVisionTokenizerTest(TokenizerTesterMixin, unittest.TestCase):
    tokenizer_class = GPTOSSVisionTokenizer
    rust_tokenizer_class = GPTOSSVisionTokenizerFast
    test_rust_tokenizer = True
    space_between_special_tokens = True

    def setUp(self):
        super().setUp()

        # Create a simple vocabulary and merges file for testing
        vocab = {
            "<|endoftext|>": 0,
            "!": 1,
            "\"": 2,
            "#": 3,
            "$": 4,
            "%": 5,
            "&": 6,
            "'": 7,
            "(": 8,
            ")": 9,
            "*": 10,
            "+": 11,
            ",": 12,
            "-": 13,
            ".": 14,
            "/": 15,
            "0": 16,
            "1": 17,
            "2": 18,
            "3": 19,
            "4": 20,
            "5": 21,
            "6": 22,
            "7": 23,
            "8": 24,
            "9": 25,
            ":": 26,
            ";": 27,
            "<": 28,
            "=": 29,
            ">": 30,
            "?": 31,
            "@": 32,
            "A": 33,
            "B": 34,
            "C": 35,
            "D": 36,
            "E": 37,
            "F": 38,
            "G": 39,
            "H": 40,
            "I": 41,
            "J": 42,
            "K": 43,
            "L": 44,
            "M": 45,
            "N": 46,
            "O": 47,
            "P": 48,
            "Q": 49,
            "R": 50,
            "S": 51,
            "T": 52,
            "U": 53,
            "V": 54,
            "W": 55,
            "X": 56,
            "Y": 57,
            "Z": 58,
            "[": 59,
            "\\": 60,
            "]": 61,
            "^": 62,
            "_": 63,
            "`": 64,
            "a": 65,
            "b": 66,
            "c": 67,
            "d": 68,
            "e": 69,
            "f": 70,
            "g": 71,
            "h": 72,
            "i": 73,
            "j": 74,
            "k": 75,
            "l": 76,
            "m": 77,
            "n": 78,
            "o": 79,
            "p": 80,
            "q": 81,
            "r": 82,
            "s": 83,
            "t": 84,
            "u": 85,
            "v": 86,
            "w": 87,
            "x": 88,
            "y": 89,
            "z": 90,
            "{": 91,
            "|": 92,
            "}": 93,
            "~": 94,
            "Ġ": 95,
            "Ġt": 96,
            "Ġth": 97,
            "Ġthe": 98,
            "ĠtheĠ": 99,
            "ĠtheĠq": 100,
            "ĠtheĠqu": 101,
            "ĠtheĠqui": 102,
            "ĠtheĠquic": 103,
            "ĠtheĠquick": 104,
            "ĠtheĠquickĠ": 105,
            "ĠtheĠquickĠb": 106,
            "ĠtheĠquickĠbr": 107,
            "ĠtheĠquickĠbro": 108,
            "ĠtheĠquickĠbrow": 109,
            "ĠtheĠquickĠbrown": 110,
            "ĠtheĠquickĠbrownĠ": 111,
            "ĠtheĠquickĠbrownĠf": 112,
            "ĠtheĠquickĠbrownĠfo": 113,
            "ĠtheĠquickĠbrownĠfox": 114,
            "ĠtheĠquickĠbrownĠfoxĠ": 115,
            "ĠtheĠquickĠbrownĠfoxĠj": 116,
            "ĠtheĠquickĠbrownĠfoxĠju": 117,
            "ĠtheĠquickĠbrownĠfoxĠjum": 118,
            "ĠtheĠquickĠbrownĠfoxĠjump": 119,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠ": 120,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠo": 121,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠov": 122,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠove": 123,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠover": 124,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠ": 125,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠt": 126,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠth": 127,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠthe": 128,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠ": 129,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠl": 130,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠla": 131,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlaz": 132,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlazy": 133,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlazyĠ": 134,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlazyĠd": 135,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlazyĠdo": 136,
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlazyĠdog": 137,
        }

        merges = [
            "#version: 0.2",
            "Ġ t",
            "Ġt h",
            "Ġth e",
            "Ġthe Ġ",
            "ĠtheĠ q",
            "ĠtheĠq u",
            "ĠtheĠqu i",
            "ĠtheĠqui c",
            "ĠtheĠquic k",
            "ĠtheĠquick Ġ",
            "ĠtheĠquickĠ b",
            "ĠtheĠquickĠb r",
            "ĠtheĠquickĠbr o",
            "ĠtheĠquickĠbro w",
            "ĠtheĠquickĠbrow n",
            "ĠtheĠquickĠbrown Ġ",
            "ĠtheĠquickĠbrownĠ f",
            "ĠtheĠquickĠbrownĠf o",
            "ĠtheĠquickĠbrownĠfo x",
            "ĠtheĠquickĠbrownĠfox Ġ",
            "ĠtheĠquickĠbrownĠfoxĠ j",
            "ĠtheĠquickĠbrownĠfoxĠj u",
            "ĠtheĠquickĠbrownĠfoxĠju m",
            "ĠtheĠquickĠbrownĠfoxĠjum p",
            "ĠtheĠquickĠbrownĠfoxĠjump Ġ",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠ o",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠo v",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠov e",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠove r",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠover Ġ",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠ t",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠt h",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠth e",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠthe Ġ",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠ l",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠl a",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠla z",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlaz y",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlazy Ġ",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlazyĠ d",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlazyĠd o",
            "ĠtheĠquickĠbrownĠfoxĠjumpĠoverĠtheĠlazyĠdo g",
        ]

        self.tmpdirname = tempfile.mkdtemp()
        self.vocab_file = os.path.join(self.tmpdirname, "vocab.json")
        self.merges_file = os.path.join(self.tmpdirname, "merges.txt")

        with open(self.vocab_file, "w", encoding="utf-8") as fp:
            fp.write(json.dumps(vocab))

        with open(self.merges_file, "w", encoding="utf-8") as fp:
            fp.write("\n".join(merges))

    def get_tokenizer(self, **kwargs):
        return GPTOSSVisionTokenizer.from_pretrained(self.tmpdirname, **kwargs)

    def get_rust_tokenizer(self, **kwargs):
        return GPTOSSVisionTokenizerFast.from_pretrained(self.tmpdirname, **kwargs)

    def test_full_tokenizer(self):
        tokenizer = GPTOSSVisionTokenizer(self.vocab_file, self.merges_file)
        text = "The quick brown fox jumps over the lazy dog"
        bpe_tokens = [
            "ĠThe",
            "Ġquick",
            "Ġbrown",
            "Ġfox",
            "Ġjumps",
            "Ġover",
            "Ġthe",
            "Ġlazy",
            "Ġdog",
        ]
        tokens = tokenizer.tokenize(text)
        self.assertListEqual(tokens, bpe_tokens)

        input_tokens = tokens + [tokenizer.eos_token]
        input_bpe_tokens = [0, 1, 2, 3, 4, 5, 6, 7, 8, 0]
        self.assertListEqual(tokenizer.convert_tokens_to_ids(input_tokens), input_bpe_tokens)

    def test_pretrained_model_lists(self):
        # We should have at least one default checkpoint for testing
        model_list = GPTOSSVisionTokenizer.pretrained_model_archive_map
        self.assertGreater(len(model_list), 0)

    def test_rust_and_python_full_tokenizers(self):
        if not self.test_rust_tokenizer:
            return

        tokenizer = self.get_tokenizer()
        rust_tokenizer = self.get_rust_tokenizer()

        sequence = "The quick brown fox jumps over the lazy dog"

        # Testing tokenization
        ids = tokenizer.encode(sequence)
        rust_ids = rust_tokenizer.encode(sequence)
        self.assertListEqual(ids, rust_ids)

        # Testing decoding
        text = tokenizer.decode(ids)
        rust_text = rust_tokenizer.decode(rust_ids)
        self.assertEqual(text, rust_text)

    def test_eos_treatment(self):
        tokenizer = self.get_tokenizer()
        # 1. Test that it does not add eos by default
        ids = tokenizer.encode("Hello world")
        self.assertEqual(ids, [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134, 135, 136, 137])

        # 2. Test that it correctly handles eos
        tokenizer.eos_token = "<|endoftext|>"
        ids = tokenizer.encode("Hello world", add_special_tokens=True)
        self.assertEqual(ids[-1], tokenizer.eos_token_id)

    def test_encode_decode_with_spaces(self):
        tokenizer = self.get_tokenizer()

        new_toks = ["[CLS]", "Hello", "World", "[SEP]"]
        input_ids = tokenizer.convert_tokens_to_ids(new_toks)
        text = tokenizer.decode(input_ids)

        self.assertEqual(text, "[CLS]HelloWorld[SEP]")

    def test_convert_tokens_to_string(self):
        tokenizer = self.get_tokenizer()

        tokens = ["Hello", "World"]
        string = tokenizer.convert_tokens_to_string(tokens)

        self.assertEqual(string, "HelloWorld")

    def test_add_special_tokens(self):
        tokenizer = self.get_tokenizer()
        input_ids = [1, 2, 3]
        special_tokens_dict = {"eos_token": "<|endoftext|>"}
        result = tokenizer.add_special_tokens(special_tokens_dict, input_ids)
        self.assertEqual(result, [1, 2, 3, 0])

    def test_prepare_for_tokenization(self):
        tokenizer = self.get_tokenizer()
        text, kwargs = tokenizer.prepare_for_tokenization("Hello world")
        self.assertEqual(text, "Hello world")
        self.assertEqual(kwargs, {})

        # Test with add_prefix_space=True
        tokenizer.add_prefix_space = True
        text, kwargs = tokenizer.prepare_for_tokenization("Hello world")
        self.assertEqual(text, " Hello world")
        self.assertEqual(kwargs, {})

    def test_save_and_load_tokenizer(self):
        # We want to verify that we can save and load a tokenizer
        tokenizer = self.get_tokenizer()
        tokenizer.save_pretrained(self.tmpdirname)

        # Load the tokenizer
        loaded_tokenizer = GPTOSSVisionTokenizer.from_pretrained(self.tmpdirname)
        self.assertEqual(tokenizer.get_vocab(), loaded_tokenizer.get_vocab())

        # Test encoding/decoding
        text = "Hello world"
        ids = tokenizer.encode(text)
        loaded_ids = loaded_tokenizer.encode(text)
        self.assertListEqual(ids, loaded_ids)

        decoded_text = tokenizer.decode(ids)
        loaded_decoded_text = loaded_tokenizer.decode(loaded_ids)
        self.assertEqual(decoded_text, loaded_decoded_text)
