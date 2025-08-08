<!--
Copyright 2025 Dustin Loring.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0
-->

<div style="float: right;">
    <div class="flex flex-wrap space-x-1">
        <img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-DE3412?style=flat&logo=pytorch&logoColor=white">
        <img alt="FlashAttention" src="https://img.shields.io/badge/%E2%9A%A1%EF%B8%8E%20FlashAttention-eae0c8?style=flat">
        <img alt="SDPA" src="https://img.shields.io/badge/SDPA-DE3412?style=flat&logo=pytorch&logoColor=white">
    </div>
</div>

# GPT-OSS-Vision-20B

## Overview

**GPT-OSS-Vision-20B** extends the open-source [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b) model by adding two major capabilities:  

1. **Vision Input** via a lightweight ViT-based adapter, enabling the model to accept and process image data alongside text.  
2. **NoPE (No Positional Encoding in periodic layers)** for improved long-context stability, inspired by [SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3) methodology.

The base GPT-OSS-20B architecture is a large Mixture-of-Experts (MoE) causal language model developed and released by [OpenAI](https://huggingface.co/openai). Our work builds directly upon their publicly released weights and configuration, preserving full compatibility for text-only use while introducing optional multimodal and long-context-aware features.

---

### **Key Features**
- **Multimodal**: Combines text and vision streams into a unified sequence for joint processing.
- **Configurable NoPE**: Neutralizes rotary position embeddings in every _n_-th transformer layer.
- **Expert Routing Bias**: MoE router includes a learned bias for visual tokens, allowing specialization.
- **Weight Compatibility**: Can load weights from `openai/gpt-oss-20b` with `strict=False` for immediate use.

---

## Acknowledgements

This work derives from and extends:
- **Base architecture**: [OpenAI’s GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)  
- **NoPE inspiration**: [HuggingFaceTB/SmolLM3](https://huggingface.co/HuggingFaceTB/SmolLM3)

Special thanks to the Hugging Face team and the open-source contributors behind the GPT-OSS and Transformers projects.

---

## Paper References

- *[OpenAI GPT-OSS Technical Report]* — _link to be added when available_
- *[SmolLM3: Efficient Long-Context LMs]* — https://huggingface.co/HuggingFaceTB/SmolLM3

---

## Author & Contact

Maintained by **Dustin Loring**  
📧 Email: [Dustinwloring1988@gmail.com](mailto:Dustinwloring1988@gmail.com)  
📂 Repo: [AIGym/gpt_oss_vision_20B](https://github.com/AIGym/gpt_oss_vision_20B)  

---

## GPTOSSVisionConfig

[[autodoc]] GPTOSSVisionConfig

## GPTOSSVisionModel

[[autodoc]] GPTOSSVisionModel
    - forward

## GPTOSSVisionForCausalLM

[[autodoc]] GPTOSSVisionForCausalLM
    - forward
