#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 Dustin Loring
# 
# Example fine-tuning script for GPT-OSS-Vision model
# This script demonstrates how to use the finetune_gpt_oss_vision.py script
# with different configurations for text-only and multimodal fine-tuning.

import os
import subprocess
import sys
from pathlib import Path

def run_finetune_example():
    """Run example fine-tuning commands."""
    
    # Example 1: Text-only fine-tuning
    print("=== Example 1: Text-only Fine-tuning ===")
    text_only_cmd = [
        "python", "finetune_gpt_oss_vision.py",
        "--model_name_or_path", "your-base-model-path",  # Replace with your model path
        "--dataset_path", "path/to/your/text/dataset.json",  # Replace with your dataset
        "--dataset_type", "text_only",
        "--output_dir", "./finetuned_text_model",
        "--batch_size", "2",
        "--gradient_accumulation_steps", "8",
        "--learning_rate", "5e-5",
        "--num_train_epochs", "3",
        "--max_length", "2048",
        "--hf_repo_id", "your-username/gpt-oss-vision-finetuned-text",  # Replace with your repo
        "--hf_token", "your-hf-token",  # Replace with your token
        "--use_wandb",
        "--seed", "42"
    ]
    
    print("Command:", " ".join(text_only_cmd))
    print("Note: Uncomment the line below to run this example")
    # subprocess.run(text_only_cmd)
    
    print("\n" + "="*50 + "\n")
    
    # Example 2: Multimodal fine-tuning
    print("=== Example 2: Multimodal Fine-tuning ===")
    multimodal_cmd = [
        "python", "finetune_gpt_oss_vision.py",
        "--model_name_or_path", "your-base-model-path",  # Replace with your model path
        "--dataset_path", "path/to/your/multimodal/dataset.json",  # Replace with your dataset
        "--dataset_type", "multimodal",
        "--output_dir", "./finetuned_multimodal_model",
        "--batch_size", "1",  # Smaller batch size for multimodal
        "--gradient_accumulation_steps", "16",
        "--learning_rate", "3e-5",
        "--num_train_epochs", "5",
        "--max_length", "2048",
        "--image_size", "224",
        "--text_column", "text",
        "--image_column", "image_path",
        "--hf_repo_id", "your-username/gpt-oss-vision-finetuned-multimodal",  # Replace with your repo
        "--hf_token", "your-hf-token",  # Replace with your token
        "--use_wandb",
        "--seed", "42"
    ]
    
    print("Command:", " ".join(multimodal_cmd))
    print("Note: Uncomment the line below to run this example")
    # subprocess.run(multimodal_cmd)
    
    print("\n" + "="*50 + "\n")
    
    # Example 3: Quantized fine-tuning (for memory efficiency)
    print("=== Example 3: Quantized Fine-tuning (4-bit) ===")
    quantized_cmd = [
        "python", "finetune_gpt_oss_vision.py",
        "--model_name_or_path", "your-base-model-path",  # Replace with your model path
        "--dataset_path", "path/to/your/text/dataset.json",  # Replace with your dataset
        "--dataset_type", "text_only",
        "--output_dir", "./finetuned_quantized_model",
        "--load_in_4bit",  # Enable 4-bit quantization
        "--batch_size", "4",
        "--gradient_accumulation_steps", "4",
        "--learning_rate", "5e-5",
        "--num_train_epochs", "3",
        "--max_length", "2048",
        "--hf_repo_id", "your-username/gpt-oss-vision-finetuned-quantized",  # Replace with your repo
        "--hf_token", "your-hf-token",  # Replace with your token
        "--use_wandb",
        "--seed", "42"
    ]
    
    print("Command:", " ".join(quantized_cmd))
    print("Note: Uncomment the line below to run this example")
    # subprocess.run(quantized_cmd)

def create_sample_dataset():
    """Create a sample dataset for testing."""
    
    # Sample text dataset
    sample_text_data = [
        {"text": "The GPT-OSS-Vision model is a powerful multimodal language model that can process both text and images."},
        {"text": "This model uses advanced attention mechanisms and can handle complex reasoning tasks."},
        {"text": "Fine-tuning allows us to adapt the model to specific domains and tasks."},
        {"text": "The model architecture includes vision encoders and text decoders for multimodal understanding."},
        {"text": "Training with gradient checkpointing helps manage memory usage during fine-tuning."},
    ]
    
    # Save sample dataset
    import json
    with open("sample_text_dataset.json", "w") as f:
        json.dump(sample_text_data, f, indent=2)
    
    print("Created sample_text_dataset.json")
    print("You can use this file to test the fine-tuning script")

def main():
    """Main function to demonstrate fine-tuning usage."""
    
    print("GPT-OSS-Vision Fine-tuning Examples")
    print("=" * 50)
    
    # Create sample dataset
    create_sample_dataset()
    
    print("\n" + "="*50)
    print("FINE-TUNING EXAMPLES")
    print("="*50)
    
    # Show examples
    run_finetune_example()
    
    print("\n" + "="*50)
    print("USAGE INSTRUCTIONS")
    print("="*50)
    print("1. Replace 'your-base-model-path' with the path to your pretrained model")
    print("2. Replace 'path/to/your/dataset.json' with your actual dataset path")
    print("3. Replace 'your-username' with your Hugging Face username")
    print("4. Replace 'your-hf-token' with your Hugging Face access token")
    print("5. Adjust hyperparameters based on your hardware and requirements")
    print("6. Uncomment the subprocess.run() lines to execute the commands")
    
    print("\n" + "="*50)
    print("DATASET FORMAT")
    print("="*50)
    print("Text-only dataset (JSON format):")
    print('[{"text": "Your text here"}, {"text": "Another text"}, ...]')
    print("\nMultimodal dataset (JSON format):")
    print('[{"text": "Description", "image": "path/to/image.jpg"}, ...]')
    
    print("\n" + "="*50)
    print("IMPORTANT NOTES")
    print("="*50)
    print("- Ensure you have sufficient GPU memory for your chosen batch size")
    print("- Use gradient accumulation to simulate larger batch sizes")
    print("- Consider using quantization (--load_in_4bit) for memory efficiency")
    print("- Monitor training with WandB for better insights")
    print("- The model will be automatically uploaded to Hugging Face if hf_repo_id is provided")

if __name__ == "__main__":
    main()
