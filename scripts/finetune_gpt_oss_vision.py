#!/usr/bin/env python3
# coding=utf-8
# Copyright 2025 Dustin Loring
# 
# Fine-tuning script for GPT-OSS-Vision model
# Licensed under the Apache License, Version 2.0 (the "License");
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
#
# This script provides fine-tuning capabilities for the GPT-OSS-Vision model
# with support for both text and vision inputs, and automatic upload to Hugging Face.
# Contact: Dustin Loring <Dustinwloring1988@gmail.com>

import os
import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
import warnings

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    get_linear_schedule_with_warmup,
    set_seed,
)
from datasets import Dataset, load_dataset
from accelerate import Accelerator
from huggingface_hub import HfApi, login
import wandb
from PIL import Image
import numpy as np

# Import your custom model components
from model.gpt_oss_vision import (
    GPTOSSVisionConfig,
    GPTOSSVisionForCausalLM,
    GPTOSSVisionTokenizer,
    GPTOSSVisionImageProcessor,
    GPTOSSVisionProcessor,
)

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore")


class GPTOSSVisionFineTuner:
    """
    Fine-tuning class for GPT-OSS-Vision model with support for:
    - Text-only fine-tuning
    - Multimodal (text + image) fine-tuning
    - Automatic Hugging Face upload
    - WandB integration
    - Gradient checkpointing and mixed precision
    """
    
    def __init__(
        self,
        model_name_or_path: str,
        output_dir: str,
        hf_repo_id: Optional[str] = None,
        hf_token: Optional[str] = None,
        use_wandb: bool = True,
        seed: int = 42,
    ):
        self.model_name_or_path = model_name_or_path
        self.output_dir = Path(output_dir)
        self.hf_repo_id = hf_repo_id
        self.hf_token = hf_token
        self.use_wandb = use_wandb
        self.seed = seed
        
        # Set seed for reproducibility
        set_seed(seed)
        
        # Initialize accelerator for distributed training
        self.accelerator = Accelerator(
            mixed_precision="fp16",
            gradient_accumulation_steps=1,
        )
        
        # Initialize components
        self.config = None
        self.tokenizer = None
        self.image_processor = None
        self.processor = None
        self.model = None
        self.trainer = None
        
        # Setup Hugging Face
        if hf_token:
            login(hf_token)
            self.hf_api = HfApi()
        else:
            self.hf_api = None
            
        # Setup WandB
        if use_wandb:
            wandb.init(project="gpt-oss-vision-finetune")
    
    def load_model_and_tokenizer(self, load_in_8bit: bool = False, load_in_4bit: bool = False):
        """Load the model, tokenizer, and processors."""
        logger.info(f"Loading model and tokenizer from {self.model_name_or_path}")
        
        # Load configuration
        self.config = GPTOSSVisionConfig.from_pretrained(self.model_name_or_path)
        
        # Load tokenizer
        try:
            self.tokenizer = GPTOSSVisionTokenizer.from_pretrained(self.model_name_or_path)
        except:
            logger.warning("Custom tokenizer not found, falling back to AutoTokenizer")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name_or_path)
        
        # Ensure tokenizer has padding token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load image processor
        try:
            self.image_processor = GPTOSSVisionImageProcessor.from_pretrained(self.model_name_or_path)
        except:
            logger.warning("Custom image processor not found, using default")
            self.image_processor = None
        
        # Load processor (combines tokenizer and image processor)
        try:
            self.processor = GPTOSSVisionProcessor.from_pretrained(self.model_name_or_path)
        except:
            logger.warning("Custom processor not found, will use separate tokenizer and image processor")
            self.processor = None
        
        # Load model
        if load_in_8bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            self.model = GPTOSSVisionForCausalLM.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        elif load_in_4bit:
            from transformers import BitsAndBytesConfig
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
            self.model = GPTOSSVisionForCausalLM.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                quantization_config=quantization_config,
                device_map="auto",
                torch_dtype=torch.float16,
            )
        else:
            self.model = GPTOSSVisionForCausalLM.from_pretrained(
                self.model_name_or_path,
                config=self.config,
                torch_dtype=torch.float16,
            )
        
        logger.info(f"Model loaded with {self.model.num_parameters():,} parameters")
    
    def prepare_dataset(
        self,
        dataset_path: str,
        dataset_type: str = "text_only",  # "text_only" or "multimodal"
        max_length: int = 2048,
        image_size: int = 224,
        text_column: str = "text",
        image_column: str = "image",
        split: str = "train",
    ) -> Dataset:
        """Prepare dataset for fine-tuning."""
        logger.info(f"Loading dataset from {dataset_path}")
        
        # Load dataset
        if dataset_path.endswith(('.json', '.jsonl')):
            dataset = load_dataset('json', data_files=dataset_path, split=split)
        else:
            dataset = load_dataset(dataset_path, split=split)
        
        if dataset_type == "text_only":
            return self._prepare_text_dataset(dataset, max_length, text_column)
        elif dataset_type == "multimodal":
            return self._prepare_multimodal_dataset(dataset, max_length, image_size, text_column, image_column)
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}")
    
    def _prepare_text_dataset(self, dataset: Dataset, max_length: int, text_column: str) -> Dataset:
        """Prepare text-only dataset."""
        def tokenize_function(examples):
            # Tokenize text
            tokenized = self.tokenizer(
                examples[text_column],
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None,
            )
            
            # Add labels (same as input_ids for causal LM)
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Apply tokenization
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Tokenizing text dataset",
        )
        
        return tokenized_dataset
    
    def _prepare_multimodal_dataset(
        self, 
        dataset: Dataset, 
        max_length: int, 
        image_size: int, 
        text_column: str, 
        image_column: str
    ) -> Dataset:
        """Prepare multimodal dataset with text and images."""
        def process_multimodal(examples):
            # Process images
            images = []
            for image_path in examples[image_column]:
                if isinstance(image_path, str):
                    image = Image.open(image_path).convert('RGB')
                else:
                    image = image_path
                
                # Resize image
                image = image.resize((image_size, image_size))
                images.append(image)
            
            # Process text
            texts = examples[text_column]
            
            # Use processor if available, otherwise process separately
            if self.processor:
                processed = self.processor(
                    text=texts,
                    images=images,
                    truncation=True,
                    padding=False,
                    max_length=max_length,
                    return_tensors=None,
                )
            else:
                # Process text
                text_processed = self.tokenizer(
                    texts,
                    truncation=True,
                    padding=False,
                    max_length=max_length,
                    return_tensors=None,
                )
                
                # Process images
                if self.image_processor:
                    image_processed = self.image_processor(
                        images,
                        return_tensors=None,
                    )
                else:
                    # Convert to tensors manually
                    image_processed = {
                        "pixel_values": [np.array(img) for img in images]
                    }
                
                # Combine
                processed = {**text_processed, **image_processed}
            
            # Add labels
            processed["labels"] = processed["input_ids"].copy()
            
            return processed
        
        # Apply processing
        processed_dataset = dataset.map(
            process_multimodal,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Processing multimodal dataset",
        )
        
        return processed_dataset
    
    def setup_training(
        self,
        dataset: Dataset,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        learning_rate: float = 5e-5,
        num_train_epochs: int = 3,
        warmup_steps: int = 100,
        weight_decay: float = 0.01,
        max_grad_norm: float = 1.0,
        save_steps: int = 500,
        eval_steps: int = 500,
        logging_steps: int = 10,
        save_total_limit: int = 3,
        dataloader_pin_memory: bool = False,
        gradient_checkpointing: bool = True,
        fp16: bool = True,
        dataloader_num_workers: int = 4,
    ):
        """Setup training configuration and trainer."""
        logger.info("Setting up training configuration")
        
        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(self.output_dir),
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            learning_rate=learning_rate,
            num_train_epochs=num_train_epochs,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            max_grad_norm=max_grad_norm,
            save_steps=save_steps,
            eval_steps=eval_steps,
            logging_steps=logging_steps,
            save_total_limit=save_total_limit,
            dataloader_pin_memory=dataloader_pin_memory,
            gradient_checkpointing=gradient_checkpointing,
            fp16=fp16,
            dataloader_num_workers=dataloader_num_workers,
            remove_unused_columns=False,
            report_to="wandb" if self.use_wandb else None,
            run_name=f"gpt-oss-vision-finetune-{self.seed}",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            push_to_hub=False,  # We'll handle this manually
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
        )
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        
        # Prepare with accelerator
        self.trainer.model, self.trainer.optimizer, self.trainer.lr_scheduler = self.accelerator.prepare(
            self.trainer.model, self.trainer.optimizer, self.trainer.lr_scheduler
        )
    
    def train(self):
        """Start training."""
        if self.trainer is None:
            raise ValueError("Trainer not set up. Call setup_training() first.")
        
        logger.info("Starting training...")
        
        # Train the model
        train_result = self.trainer.train()
        
        # Save the model
        self.trainer.save_model()
        
        # Save training metrics
        metrics = train_result.metrics
        self.trainer.log_metrics("train", metrics)
        self.trainer.save_metrics("train", metrics)
        self.trainer.save_state()
        
        logger.info(f"Training completed. Metrics: {metrics}")
        
        return train_result
    
    def upload_to_huggingface(self, commit_message: str = "Add fine-tuned GPT-OSS-Vision model"):
        """Upload the fine-tuned model to Hugging Face Hub."""
        if not self.hf_repo_id or not self.hf_api:
            logger.warning("Hugging Face repo ID or API not configured. Skipping upload.")
            return
        
        logger.info(f"Uploading model to {self.hf_repo_id}")
        
        try:
            # Push to hub
            self.trainer.push_to_hub(
                repo_id=self.hf_repo_id,
                commit_message=commit_message,
                private=False,  # Set to True if you want a private repo
            )
            
            logger.info(f"Successfully uploaded model to {self.hf_repo_id}")
            
        except Exception as e:
            logger.error(f"Failed to upload model: {e}")
            raise
    
    def cleanup(self):
        """Clean up resources."""
        if self.use_wandb:
            wandb.finish()


def main():
    parser = argparse.ArgumentParser(description="Fine-tune GPT-OSS-Vision model")
    
    # Model and data arguments
    parser.add_argument("--model_name_or_path", type=str, required=True,
                       help="Path to pretrained model or model identifier from huggingface.co/models")
    parser.add_argument("--dataset_path", type=str, required=True,
                       help="Path to dataset file or dataset name from huggingface.co/datasets")
    parser.add_argument("--dataset_type", type=str, default="text_only",
                       choices=["text_only", "multimodal"],
                       help="Type of dataset (text_only or multimodal)")
    parser.add_argument("--output_dir", type=str, default="./finetuned_model",
                       help="Output directory for the fine-tuned model")
    
    # Hugging Face arguments
    parser.add_argument("--hf_repo_id", type=str, default=None,
                       help="Hugging Face repository ID for uploading the model")
    parser.add_argument("--hf_token", type=str, default=None,
                       help="Hugging Face token for authentication")
    
    # Training arguments
    parser.add_argument("--batch_size", type=int, default=4,
                       help="Training batch size per device")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4,
                       help="Number of steps for gradient accumulation")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                       help="Learning rate")
    parser.add_argument("--num_train_epochs", type=int, default=3,
                       help="Number of training epochs")
    parser.add_argument("--max_length", type=int, default=2048,
                       help="Maximum sequence length")
    parser.add_argument("--image_size", type=int, default=224,
                       help="Image size for multimodal training")
    
    # Model loading arguments
    parser.add_argument("--load_in_8bit", action="store_true",
                       help="Load model in 8-bit precision")
    parser.add_argument("--load_in_4bit", action="store_true",
                       help="Load model in 4-bit precision")
    
    # Other arguments
    parser.add_argument("--use_wandb", action="store_true", default=True,
                       help="Use Weights & Biases for logging")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")
    parser.add_argument("--text_column", type=str, default="text",
                       help="Column name for text data")
    parser.add_argument("--image_column", type=str, default="image",
                       help="Column name for image data")
    
    args = parser.parse_args()
    
    # Initialize fine-tuner
    fine_tuner = GPTOSSVisionFineTuner(
        model_name_or_path=args.model_name_or_path,
        output_dir=args.output_dir,
        hf_repo_id=args.hf_repo_id,
        hf_token=args.hf_token,
        use_wandb=args.use_wandb,
        seed=args.seed,
    )
    
    try:
        # Load model and tokenizer
        fine_tuner.load_model_and_tokenizer(
            load_in_8bit=args.load_in_8bit,
            load_in_4bit=args.load_in_4bit,
        )
        
        # Prepare dataset
        dataset = fine_tuner.prepare_dataset(
            dataset_path=args.dataset_path,
            dataset_type=args.dataset_type,
            max_length=args.max_length,
            image_size=args.image_size,
            text_column=args.text_column,
            image_column=args.image_column,
        )
        
        # Setup training
        fine_tuner.setup_training(
            dataset=dataset,
            batch_size=args.batch_size,
            gradient_accumulation_steps=args.gradient_accumulation_steps,
            learning_rate=args.learning_rate,
            num_train_epochs=args.num_train_epochs,
        )
        
        # Train
        train_result = fine_tuner.train()
        
        # Upload to Hugging Face
        if args.hf_repo_id:
            fine_tuner.upload_to_huggingface()
        
        logger.info("Fine-tuning completed successfully!")
        
    except Exception as e:
        logger.error(f"Fine-tuning failed: {e}")
        raise
    finally:
        fine_tuner.cleanup()


if __name__ == "__main__":
    main()
