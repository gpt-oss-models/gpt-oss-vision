# GPT-OSS-Vision

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Transformers](https://img.shields.io/badge/Transformers-4.36+-blue.svg)](https://huggingface.co/transformers)

**GPT-OSS-Vision** is a high-capacity Mixture-of-Experts (MoE) causal language model with multimodal vision capabilities and NoPE (No Positional Embedding) support for efficient long-context processing.

## 🚀 Features

### 🔍 **Multimodal Vision Support**
- **Vision Adapter**: Lightweight ViT-based encoder for image processing
- **Joint Text-Image Processing**: Seamless integration of text and visual inputs
- **Expert Specialization**: MoE experts can specialize on visual vs text tokens

### 📏 **NoPE (No Positional Embedding)**
- **Configurable Layer-wise NoPE**: Selectively disable rotary positional encodings in periodic layers
- **Long Context Performance**: Improved retention over extended input contexts
- **SmolLM3 Methodology**: Consistent with proven long-context techniques

### 🧠 **Mixture of Experts (MoE)**
- **128 Local Experts**: High-capacity model with efficient routing
- **Vision-Aware Routing**: Router bias for vision tokens to enable expert specialization
- **Load Balancing**: Optimized expert utilization across modalities

### ⚡ **Performance Optimizations**
- **Flash Attention**: Support for efficient attention mechanisms
- **Gradient Checkpointing**: Memory-efficient training
- **Flexible Attention**: Support for sliding window and full attention modes

## 📦 Installation

### From Source
```bash
git clone https://github.com/gpt-oss-vision/gpt-oss-vision.git
cd gpt-oss-vision
pip install -e .
```

### With Vision Dependencies
```bash
pip install -e .[vision]
```

### For Development
```bash
pip install -e .[dev]
```

## 🎯 Quick Start

### Basic Text Generation
```python
from transformers import GPTOSSVisionForCausalLM, GPTOSSVisionTokenizer

# Load model and tokenizer
model = GPTOSSVisionForCausalLM.from_pretrained("AIGym/gpt_oss_vision_20B")
tokenizer = GPTOSSVisionTokenizer.from_pretrained("AIGym/gpt_oss_vision_20B")

# Generate text
text = "The future of artificial intelligence is"
inputs = tokenizer(text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Multimodal Generation
```python
from transformers import GPTOSSVisionProcessor
from PIL import Image

# Load processor (combines tokenizer and image processor)
processor = GPTOSSVisionProcessor.from_pretrained("AIGym/gpt_oss_vision_20B")

# Load image
image = Image.open("path/to/image.jpg")

# Process text and image
text = "Describe this image in detail:"
inputs = processor(text=text, images=image, return_tensors="pt")

# Generate multimodal response
outputs = model.generate(**inputs, max_new_tokens=100)
response = processor.decode(outputs[0], skip_special_tokens=True)
print(response)
```

### Long Context with NoPE
```python
from transformers import GPTOSSVisionConfig, GPTOSSVisionForCausalLM

# Configure NoPE for long context
config = GPTOSSVisionConfig.from_pretrained("AIGym/gpt_oss_vision_20B")
config.use_nope = True
config.nope_stride = 4  # Disable RoPE every 4th layer

# Load model with NoPE configuration
model = GPTOSSVisionForCausalLM.from_pretrained(
    "AIGym/gpt_oss_vision_20B", 
    config=config
)

# Process long context
long_text = "..." # Your long context here
inputs = tokenizer(long_text, return_tensors="pt")
outputs = model.generate(**inputs, max_new_tokens=50)
```

## 🔧 Configuration

### Vision Parameters
```python
config = GPTOSSVisionConfig(
    # Vision encoder settings
    vision_embed_dim=1024,
    vision_patch_size=14,
    vision_num_channels=3,
    vision_num_layers=24,
    vision_num_heads=16,
    
    # Enable vision processing
    use_vision=True,
)
```

### NoPE Parameters
```python
config = GPTOSSVisionConfig(
    # Enable NoPE
    use_nope=True,
    nope_stride=4,  # Disable RoPE every 4th layer
)
```

### MoE Parameters
```python
config = GPTOSSVisionConfig(
    # Expert configuration
    num_local_experts=128,
    num_experts_per_tok=4,
    router_aux_loss_coef=0.9,
    
    # Model architecture
    hidden_size=2880,
    num_hidden_layers=36,
    num_attention_heads=64,
)
```

## 📊 Model Architecture

### Vision Adapter
- **ViT Encoder**: 24-layer Vision Transformer
- **Patch Size**: 14x14 pixels
- **Embedding Dimension**: 1024 → 2880 (model hidden size)
- **Pool Token**: Optional learnable prefix token

### NoPE Implementation
- **Layer-wise Control**: Configurable stride for RoPE neutralization
- **Positional Encoding**: Replaces (cos, sin) with (1, 0) in periodic layers
- **Backward Compatibility**: Maintains standard RoPE when disabled

### MoE Architecture
- **128 Local Experts**: High-capacity feed-forward networks
- **Top-K Routing**: Routes tokens to 4 experts per token
- **Vision Bias**: Learned bias for vision token routing
- **Load Balancing**: Auxiliary loss for expert utilization

## 🧪 Testing

Run the test suite:
```bash
pytest tests/
```

Run specific test categories:
```bash
# Model tests
pytest tests/gpt_oss_vision/test_modeling_gpt_oss_vision.py

# Tokenizer tests
pytest tests/gpt_oss_vision/test_tokenization_gpt_oss_vision.py

# Integration tests
pytest tests/gpt_oss_vision/ -m "slow"
```

## 📈 Performance

### Long Context Benchmarks
- **16K Tokens**: Improved retention vs baseline
- **NoPE Impact**: ≤1% regression on short-context tasks
- **Memory Efficiency**: Optimized for extended sequences

### Multimodal Capabilities
- **Image Understanding**: Robust visual comprehension
- **Text Generation**: High-quality multimodal responses
- **Expert Utilization**: Balanced routing across modalities

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/gpt-oss-vision/gpt-oss-vision.git
cd gpt-oss-vision
pip install -e .[dev]
pre-commit install
```

## 📄 License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Based on the original GPT-OSS from Hugging Face & OpenAI
- NoPE methodology inspired by SmolLM3
- Vision integration patterns from Gemma3/Llama4
- MoE architecture from Mixtral and related work

## 📞 Contact

- **Author**: Dustin Loring
- **Email**: Dustinwloring1988@gmail.com
- **GitHub**: [@gpt-oss-vision](https://github.com/gpt-oss-vision)
- **Hugging Face**: [AIGym/gpt_oss_vision_20B](https://huggingface.co/AIGym/gpt_oss_vision_20B)

## 🔗 Links

- [Documentation](https://huggingface.co/docs/transformers/model_doc/gpt-oss-vision)
- [Model Hub](https://huggingface.co/AIGym/gpt_oss_vision_20B)
- [Paper](https://arxiv.org/abs/XXXX.XXXXX) (Coming Soon)
- [Discussions](https://github.com/gpt-oss-vision/gpt-oss-vision/discussions)
