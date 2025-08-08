# GPT-OSS-Vision Fine-tuning Guide

This guide explains how to fine-tune the GPT-OSS-Vision model for your specific use cases, including both text-only and multimodal (text + image) fine-tuning.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Dataset Preparation](#dataset-preparation)
- [Fine-tuning Options](#fine-tuning-options)
- [Advanced Configuration](#advanced-configuration)
- [Troubleshooting](#troubleshooting)
- [Examples](#examples)

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- Hugging Face account and access token
- Weights & Biases account (optional, for experiment tracking)

## Installation

1. Install the required dependencies:

```bash
pip install -r requirements.txt
```

2. Set up your Hugging Face token:

```bash
export HF_TOKEN="your-huggingface-token"
```

3. (Optional) Set up Weights & Biases:

```bash
wandb login
```

## Quick Start

### 1. Text-only Fine-tuning

```bash
python finetune_gpt_oss_vision.py \
    --model_name_or_path "path/to/your/base/model" \
    --dataset_path "path/to/your/dataset.json" \
    --dataset_type "text_only" \
    --output_dir "./finetuned_model" \
    --hf_repo_id "your-username/your-model-name" \
    --hf_token "your-hf-token" \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-5 \
    --num_train_epochs 3
```

### 2. Multimodal Fine-tuning

```bash
python finetune_gpt_oss_vision.py \
    --model_name_or_path "path/to/your/base/model" \
    --dataset_path "path/to/your/multimodal/dataset.json" \
    --dataset_type "multimodal" \
    --output_dir "./finetuned_multimodal_model" \
    --hf_repo_id "your-username/your-multimodal-model" \
    --hf_token "your-hf-token" \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --image_size 224
```

## Dataset Preparation

### Text-only Dataset Format

Create a JSON file with the following structure:

```json
[
    {"text": "Your first training example text here."},
    {"text": "Your second training example text here."},
    {"text": "Your third training example text here."}
]
```

### Multimodal Dataset Format

Create a JSON file with the following structure:

```json
[
    {
        "text": "A description of the image content.",
        "image": "path/to/image1.jpg"
    },
    {
        "text": "Another description of a different image.",
        "image": "path/to/image2.jpg"
    }
]
```

### Dataset Loading Options

The script supports multiple dataset formats:

- **Local JSON/JSONL files**: `--dataset_path "path/to/dataset.json"`
- **Hugging Face datasets**: `--dataset_path "huggingface/dataset-name"`
- **Custom dataset splits**: Add `--split "train"` for specific splits

## Fine-tuning Options

### Basic Parameters

| Parameter | Description | Default | Recommended Range |
|-----------|-------------|---------|-------------------|
| `--batch_size` | Training batch size per device | 4 | 1-8 (depending on GPU memory) |
| `--gradient_accumulation_steps` | Steps for gradient accumulation | 4 | 4-32 |
| `--learning_rate` | Learning rate | 5e-5 | 1e-5 to 1e-4 |
| `--num_train_epochs` | Number of training epochs | 3 | 1-10 |
| `--max_length` | Maximum sequence length | 2048 | 512-4096 |

### Memory Optimization

For limited GPU memory, use these options:

```bash
# 4-bit quantization (recommended for memory efficiency)
--load_in_4bit

# 8-bit quantization
--load_in_8bit

# Reduce batch size and increase gradient accumulation
--batch_size 1 --gradient_accumulation_steps 16
```

### Advanced Training Options

```bash
# Enable gradient checkpointing (saves memory)
--gradient_checkpointing

# Use mixed precision training
--fp16

# Custom warmup steps
--warmup_steps 100

# Weight decay for regularization
--weight_decay 0.01
```

## Advanced Configuration

### Custom Model Configuration

You can modify the model configuration by editing the `GPTOSSVisionConfig` class in `model/gpt_oss_vision/configuration_gpt_oss_vision.py`:

```python
config = GPTOSSVisionConfig(
    num_hidden_layers=36,
    hidden_size=2880,
    vision_embed_dim=1024,
    use_nope=True,  # Enable NoPE for certain layers
    nope_stride=4,
)
```

### Custom Training Loop

For advanced use cases, you can extend the `GPTOSSVisionFineTuner` class:

```python
from finetune_gpt_oss_vision import GPTOSSVisionFineTuner

class CustomFineTuner(GPTOSSVisionFineTuner):
    def custom_training_step(self, batch):
        # Implement custom training logic
        pass

# Use your custom fine-tuner
fine_tuner = CustomFineTuner(
    model_name_or_path="your-model",
    output_dir="./output"
)
```

## Troubleshooting

### Common Issues

1. **Out of Memory (OOM)**
   - Reduce batch size: `--batch_size 1`
   - Enable quantization: `--load_in_4bit`
   - Increase gradient accumulation: `--gradient_accumulation_steps 16`
   - Enable gradient checkpointing: `--gradient_checkpointing`

2. **Slow Training**
   - Increase batch size if memory allows
   - Reduce gradient accumulation steps
   - Use mixed precision: `--fp16`
   - Optimize data loading: `--dataloader_num_workers 4`

3. **Poor Convergence**
   - Adjust learning rate
   - Increase training epochs
   - Check dataset quality
   - Monitor loss curves in WandB

4. **Import Errors**
   - Ensure all dependencies are installed: `pip install -r requirements.txt`
   - Check Python version compatibility
   - Verify model path is correct

### Performance Tips

1. **Memory Management**
   - Use gradient checkpointing for large models
   - Enable mixed precision training
   - Consider quantization for inference

2. **Training Efficiency**
   - Use appropriate batch sizes for your hardware
   - Monitor GPU utilization
   - Use WandB for experiment tracking

3. **Data Pipeline**
   - Preprocess data offline when possible
   - Use efficient data loading
   - Cache processed datasets

## Examples

### Example 1: Domain-specific Fine-tuning

```bash
python finetune_gpt_oss_vision.py \
    --model_name_or_path "your-base-model" \
    --dataset_path "medical_texts.json" \
    --dataset_type "text_only" \
    --output_dir "./medical_model" \
    --hf_repo_id "your-username/medical-gpt-oss-vision" \
    --hf_token "your-hf-token" \
    --batch_size 2 \
    --gradient_accumulation_steps 8 \
    --learning_rate 3e-5 \
    --num_train_epochs 5 \
    --max_length 4096
```

### Example 2: Multimodal Instruction Following

```bash
python finetune_gpt_oss_vision.py \
    --model_name_or_path "your-base-model" \
    --dataset_path "instruction_dataset.json" \
    --dataset_type "multimodal" \
    --output_dir "./instruction_model" \
    --hf_repo_id "your-username/instruction-gpt-oss-vision" \
    --hf_token "your-hf-token" \
    --batch_size 1 \
    --gradient_accumulation_steps 16 \
    --learning_rate 2e-5 \
    --num_train_epochs 3 \
    --image_size 224 \
    --text_column "instruction" \
    --image_column "image_path"
```

### Example 3: Memory-efficient Fine-tuning

```bash
python finetune_gpt_oss_vision.py \
    --model_name_or_path "your-base-model" \
    --dataset_path "large_dataset.json" \
    --dataset_type "text_only" \
    --output_dir "./efficient_model" \
    --hf_repo_id "your-username/efficient-gpt-oss-vision" \
    --hf_token "your-hf-token" \
    --load_in_4bit \
    --batch_size 1 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --gradient_checkpointing \
    --fp16
```

## Monitoring and Logging

### WandB Integration

The script automatically logs training metrics to Weights & Biases:

- Training loss
- Learning rate
- Gradient norms
- Memory usage
- Training speed

### Local Logging

Training logs are saved locally in the output directory:
- `trainer_state.json`: Training state and metrics
- `training_args.bin`: Training configuration
- `pytorch_model.bin`: Model weights

## Uploading to Hugging Face

The script automatically uploads your fine-tuned model to Hugging Face Hub when you provide:

- `--hf_repo_id`: Your repository ID (e.g., "username/model-name")
- `--hf_token`: Your Hugging Face access token

The uploaded model will include:
- Model weights
- Tokenizer
- Configuration
- Training arguments
- Model card

## Support

For issues and questions:

1. Check the troubleshooting section above
2. Review the example scripts in `example_finetune.py`
3. Check the main README.md for general model information
4. Open an issue on the project repository

## License

This fine-tuning script is licensed under the Apache License 2.0, same as the main GPT-OSS-Vision model.
