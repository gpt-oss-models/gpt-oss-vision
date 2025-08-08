# coding=utf-8
# Copyright 2025 Dustin Loring
# 
# Based on the original GPT-OSS configuration from Hugging Face & OpenAI's GPT-OSS.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes:
# - Renamed to GPTOSSVisionConfig for multimodal (text + image) support
# - Added vision encoder parameters
# - Added NoPE (No Positional Embedding in periodic layers) configuration
# - Updated default settings for multimodal pretraining
# - Contact: Dustin Loring <Dustinwloring1988@gmail.com>
"""GPT-OSS-Vision model init"""

from typing import TYPE_CHECKING

from ...utils import _LazyModule
from ...utils.import_utils import define_import_structure


if TYPE_CHECKING:
    from .configuration_gpt_oss_vision import *
    from .image_processing_gpt_oss_vision import *
    from .modeling_gpt_oss_vision import *
    from .processing_gpt_oss_vision import *
    from .tokenization_gpt_oss_vision import *
    from .tokenization_gpt_oss_vision_fast import *
else:
    import sys

    _file = globals()["__file__"]
    sys.modules[__name__] = _LazyModule(__name__, _file, define_import_structure(_file), module_spec=__spec__)
