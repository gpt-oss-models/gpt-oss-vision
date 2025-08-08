#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 Dustin Loring
# 
# Based on the original GPT-OSS setup from Hugging Face & OpenAI's GPT-OSS.
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Changes:
# - Adapted for GPT-OSS-Vision multimodal support
# - Added vision dependencies
# - Contact: Dustin Loring <Dustinwloring1988@gmail.com>
"""Setup script for GPT-OSS-Vision."""

import os
import re
import sys
from pathlib import Path

from setuptools import find_packages, setup

# Ensure we're running from the right directory
setup_dir = Path(__file__).resolve().parent
os.chdir(setup_dir)

# Read the README file
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read the requirements file
def read_requirements(filename):
    with open(filename, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

# Get version from __init__.py
def get_version():
    version_file = setup_dir / "model" / "gpt_oss_vision" / "__init__.py"
    with open(version_file, "r", encoding="utf-8") as f:
        version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", f.read(), re.M)
        if version_match:
            return version_match.group(1)
    raise RuntimeError("Unable to find version string.")

# Define the package
setup(
    name="gpt-oss-vision",
    version=get_version(),
    author="Dustin Loring",
    author_email="Dustinwloring1988@gmail.com",
    description="GPT-OSS-Vision: Multimodal Mixture-of-Experts model with NoPE support",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gpt-oss-vision/gpt-oss-vision",
    project_urls={
        "Bug Reports": "https://github.com/gpt-oss-vision/gpt-oss-vision/issues",
        "Source": "https://github.com/gpt-oss-vision/gpt-oss-vision",
        "Documentation": "https://huggingface.co/docs/transformers/model_doc/gpt-oss-vision",
    },
    packages=find_packages(include=["model*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "transformers>=4.36.0",
        "torch>=2.0.0",
        "numpy>=1.21.0",
        "Pillow>=8.0.0",
        "tokenizers>=0.13.0",
        "regex>=2021.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=22.0.0",
            "isort>=5.0.0",
            "flake8>=4.0.0",
            "mypy>=0.950",
        ],
        "vision": [
            "torchvision>=0.15.0",
            "opencv-python>=4.5.0",
        ],
        "all": [
            "accelerate>=0.20.0",
            "datasets>=2.0.0",
            "evaluate>=0.4.0",
            "sentencepiece>=0.1.0",
            "protobuf>=3.20.0",
        ],
    },
    include_package_data=True,
    zip_safe=False,
    keywords="machine learning, transformers, vision, multimodal, mixture of experts, nope",
    license="Apache License 2.0",
)
