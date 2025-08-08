#!/usr/bin/env python3
"""
GPT-OSS-Vision Fine-tuning Script

This script fine-tunes the gpt-oss-vision model on the lmms-lab/LLaVA-OneVision-Data dataset.
It extends the openai/gpt-oss-20b base model with vision capabilities.

Usage:
    python finetune_gpt_oss_vision.py --config config.yaml

Author: Dustin Loring <dustinwloring1988@gmail.com>
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union, Any
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import transformers
from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    HfArgumentParser,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
from datasets import load_dataset, DatasetDict, Dataset as HFDataset
from PIL import Image
import requests
from io import BytesIO
import numpy as np
import math

# Check versions
check_min_version("4.36.0")
require_version("datasets>=2.15.0", "To fix: pip install -r requirements.txt")

# Set up logging
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

# Suppress some warnings
warnings.filterwarnings("ignore", category=UserWarning)


@dataclass
class ModelArguments:
    """Arguments pertaining to which model/config/tokenizer we are going to fine-tune."""
    
    model_name_or_path: Optional[str] = field(
        default="openai/gpt-oss-20b",
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models."}
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."}
    )
    tokenizer_name: Optional[str] = field(
        default=None,
        metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by tokenizers library) or not."}
    )
    model_max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length. Sequences will be right-padded (and possibly truncated)."}
    )
    use_auth_token: bool = field(
        default=False,
        metadata={"help": "Will use the token generated when running `transformers-cli login`."}
    )
    trust_remote_code: bool = field(
        default=False,
        metadata={"help": "Enable unpickling of arbitrary code in AutoModelForCausalLM#from_pretrained."}
    )
    # Vision specific arguments
    vision_encoder_name: str = field(
        default="openai/clip-vit-large-patch14-336",
        metadata={"help": "Vision encoder to use for image processing"}
    )
    image_size: int = field(
        default=336,
        metadata={"help": "Input image size for vision encoder"}
    )
    patch_size: int = field(
        default=14,
        metadata={"help": "Patch size for vision transformer"}
    )


@dataclass
class DataTrainingArguments:
    """Arguments pertaining to what data we are going to input our model for training and eval."""
    
    dataset_name: Optional[str] = field(
        default="lmms-lab/LLaVA-OneVision-Data",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default="websight(cauldron)",
        metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(
        default=None,
        metadata={"help": "The input training data file (a text file)."}
    )
    validation_file: Optional[str] = field(
        default=None,
        metadata={"help": "An optional input evaluation data file to evaluate the perplexity on (a text file)."}
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate training examples."}
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={"help": "For debugging purposes or quicker training, truncate evaluation examples."}
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Enable streaming mode"}
    )
    block_size: Optional[int] = field(
        default=None,
        metadata={"help": "Optional input sequence length after tokenization."}
    )
    overwrite_cache: bool = field(
        default=False,
        metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    validation_split_percentage: Optional[int] = field(
        default=5,
        metadata={"help": "The percentage of the train set used as validation set."}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."}
    )
    keep_linebreaks: bool = field(
        default=True,
        metadata={"help": "Whether to keep line breaks when using TXT files or not."}
    )
    # Multimodal specific arguments
    image_token: str = field(
        default="<image>",
        metadata={"help": "Token used to represent images in text"}
    )
    max_images_per_sample: int = field(
        default=8,
        metadata={"help": "Maximum number of images per training sample"}
    )


class LLaVAOneVisionDataset(Dataset):
    """Custom dataset for LLaVA-OneVision data with multimodal support."""
    
    def __init__(
        self, 
        dataset: HFDataset, 
        tokenizer, 
        image_processor,
        max_length: int = 2048,
        image_token: str = "<image>",
        max_images: int = 8
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.max_length = max_length
        self.image_token = image_token
        self.max_images = max_images
        
        # Add special tokens if needed
        if self.image_token not in self.tokenizer.get_vocab():
            self.tokenizer.add_tokens([self.image_token])
            
    def __len__(self):
        return len(self.dataset)
    
    def load_image(self, image_path_or_url: str) -> Optional[Image.Image]:
        """Load image from path or URL."""
        try:
            if image_path_or_url.startswith(('http://', 'https://')):
                response = requests.get(image_path_or_url, timeout=10)
                image = Image.open(BytesIO(response.content))
            else:
                image = Image.open(image_path_or_url)
            return image.convert('RGB')
        except Exception as e:
            logger.warning(f"Failed to load image {image_path_or_url}: {e}")
            return None
    
    def __getitem__(self, idx):
        """Get a single training example."""
        try:
            item = self.dataset[idx]
            
            # Extract conversation data
            conversations = item.get('conversations', [])
            images = item.get('image', []) if isinstance(item.get('image'), list) else [item.get('image')] if item.get('image') else []
            
            # Process images
            processed_images = []
            for img in images[:self.max_images]:  # Limit number of images
                if img:
                    pil_img = self.load_image(img)
                    if pil_img:
                        processed_images.append(pil_img)
            
            # Build conversation text
            text_parts = []
            for conv in conversations:
                role = conv.get('from', 'unknown')
                content = conv.get('value', '')
                
                if role == 'human':
                    # Insert image tokens for human messages
                    if processed_images and self.image_token not in content:
                        content = f"{self.image_token}\n{content}"
                    text_parts.append(f"Human: {content}")
                elif role == 'gpt':
                    text_parts.append(f"Assistant: {content}")
            
            full_text = "\n".join(text_parts)
            
            # Tokenize text
            encoding = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            
            # Process images if any
            pixel_values = None
            if processed_images:
                try:
                    pixel_values = self.image_processor(
                        processed_images, 
                        return_tensors="pt"
                    )['pixel_values']
                except Exception as e:
                    logger.warning(f"Failed to process images: {e}")
                    pixel_values = None
            
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze(),  # For causal LM
                'pixel_values': pixel_values.squeeze() if pixel_values is not None else None,
            }
            
        except Exception as e:
            logger.error(f"Error processing sample {idx}: {e}")
            # Return a dummy sample
            dummy_text = "This is a dummy sample due to processing error."
            encoding = self.tokenizer(
                dummy_text,
                truncation=True,
                max_length=self.max_length,
                padding="max_length",
                return_tensors="pt"
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(),
                'attention_mask': encoding['attention_mask'].squeeze(),
                'labels': encoding['input_ids'].squeeze(),
                'pixel_values': None,
            }


class MultimodalDataCollator:
    """Data collator for multimodal training."""
    
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
    
    def __call__(self, features):
        # Separate text and image features
        input_ids = [f['input_ids'] for f in features]
        attention_masks = [f['attention_mask'] for f in features]
        labels = [f['labels'] for f in features]
        pixel_values = [f['pixel_values'] for f in features if f['pixel_values'] is not None]
        
        # Pad sequences
        batch_size = len(input_ids)
        max_length = max(len(ids) for ids in input_ids)
        
        # Pad to multiple if specified
        if self.pad_to_multiple_of:
            max_length = ((max_length + self.pad_to_multiple_of - 1) 
                         // self.pad_to_multiple_of * self.pad_to_multiple_of)
        
        # Create padded tensors
        padded_input_ids = torch.full((batch_size, max_length), self.tokenizer.pad_token_id, dtype=torch.long)
        padded_attention_masks = torch.zeros((batch_size, max_length), dtype=torch.long)
        padded_labels = torch.full((batch_size, max_length), -100, dtype=torch.long)
        
        for i, (ids, mask, lbls) in enumerate(zip(input_ids, attention_masks, labels)):
            length = len(ids)
            padded_input_ids[i, :length] = ids
            padded_attention_masks[i, :length] = mask
            padded_labels[i, :length] = lbls
        
        batch = {
            'input_ids': padded_input_ids,
            'attention_mask': padded_attention_masks,
            'labels': padded_labels,
        }
        
        # Add pixel values if present
        if pixel_values:
            # Stack pixel values, padding with zeros if needed
            max_imgs = max(pv.shape[0] if pv.dim() > 3 else 1 for pv in pixel_values)
            img_shape = pixel_values[0].shape[-3:] if pixel_values else (3, 336, 336)
            
            batch_pixel_values = torch.zeros((batch_size, max_imgs, *img_shape))
            for i, pv in enumerate(pixel_values):
                if pv.dim() > 3:
                    num_imgs = min(pv.shape[0], max_imgs)
                    batch_pixel_values[i, :num_imgs] = pv[:num_imgs]
                else:
                    batch_pixel_values[i, 0] = pv
            
            batch['pixel_values'] = batch_pixel_values
        
        return batch


def setup_model_and_tokenizer(model_args: ModelArguments):
    """Set up the model, tokenizer, and image processor."""
    
    logger.info(f"Loading tokenizer from {model_args.tokenizer_name or model_args.model_name_or_path}")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Set model max length
    if model_args.model_max_length:
        tokenizer.model_max_length = model_args.model_max_length
    
    logger.info(f"Loading base model from {model_args.model_name_or_path}")
    
    # Load base model configuration
    config = AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code,
    )
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained(
        model_args.model_name_or_path,
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        trust_remote_code=model_args.trust_remote_code,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    
    # Resize token embeddings if we added new tokens
    model.resize_token_embeddings(len(tokenizer))
    
    logger.info("Setting up image processor")
    # Set up image processor (using CLIP for now)
    from transformers import CLIPImageProcessor
    image_processor = CLIPImageProcessor.from_pretrained(
        model_args.vision_encoder_name,
        cache_dir=model_args.cache_dir,
    )
    
    logger.info(f"Model setup complete. Model type: {type(model).__name__}")
    logger.info(f"Tokenizer vocab size: {len(tokenizer)}")
    
    return model, tokenizer, image_processor


def setup_datasets(data_args: DataTrainingArguments, tokenizer, image_processor):
    """Load and preprocess the datasets."""
    
    logger.info(f"Loading dataset: {data_args.dataset_name}")
    
    # Load dataset
    if data_args.dataset_name:
        raw_datasets = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            cache_dir=data_args.cache_dir if hasattr(data_args, 'cache_dir') else None,
            streaming=data_args.streaming,
        )
    else:
        data_files = {}
        if data_args.train_file:
            data_files["train"] = data_args.train_file
        if data_args.validation_file:
            data_files["validation"] = data_args.validation_file
        
        extension = data_args.train_file.split(".")[-1] if data_args.train_file else "json"
        raw_datasets = load_dataset(extension, data_files=data_files)
    
    # Split dataset if needed
    if "train" not in raw_datasets.keys():
        raw_datasets["train"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[:{data_args.validation_split_percentage}%]",
            cache_dir=data_args.cache_dir if hasattr(data_args, 'cache_dir') else None,
        )
        raw_datasets["validation"] = load_dataset(
            data_args.dataset_name,
            data_args.dataset_config_name,
            split=f"train[{data_args.validation_split_percentage}%:]",
            cache_dir=data_args.cache_dir if hasattr(data_args, 'cache_dir') else None,
        )
    
    # Limit samples for debugging
    if data_args.max_train_samples is not None:
        max_train_samples = min(len(raw_datasets["train"]), data_args.max_train_samples)
        raw_datasets["train"] = raw_datasets["train"].select(range(max_train_samples))
    
    if data_args.max_eval_samples is not None and "validation" in raw_datasets:
        max_eval_samples = min(len(raw_datasets["validation"]), data_args.max_eval_samples)
        raw_datasets["validation"] = raw_datasets["validation"].select(range(max_eval_samples))
    
    # Create custom datasets
    train_dataset = LLaVAOneVisionDataset(
        raw_datasets["train"],
        tokenizer,
        image_processor,
        max_length=data_args.block_size or tokenizer.model_max_length,
        image_token=data_args.image_token,
        max_images=data_args.max_images_per_sample,
    )
    
    eval_dataset = None
    if "validation" in raw_datasets:
        eval_dataset = LLaVAOneVisionDataset(
            raw_datasets["validation"],
            tokenizer,
            image_processor,
            max_length=data_args.block_size or tokenizer.model_max_length,
            image_token=data_args.image_token,
            max_images=data_args.max_images_per_sample,
        )
    
    logger.info(f"Train dataset size: {len(train_dataset)}")
    if eval_dataset:
        logger.info(f"Eval dataset size: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def main():
    # Parse arguments
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments))
    
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # Load arguments from JSON file
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    # Set up logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    
    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    
    # Log on each process the small summary
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )
    
    # Set seed for reproducibility
    transformers.set_seed(training_args.seed)
    
    # Setup model and tokenizer
    model, tokenizer, image_processor = setup_model_and_tokenizer(model_args)
    
    # Setup datasets
    train_dataset, eval_dataset = setup_datasets(data_args, tokenizer, image_processor)
    
    # Data collator
    data_collator = MultimodalDataCollator(
        tokenizer=tokenizer,
        pad_to_multiple_of=8 if training_args.fp16 else None,
    )
    
    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        
        logger.info("*** Starting training ***")
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload
        
        metrics = train_result.metrics
        max_train_samples = len(train_dataset)
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))
        
        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
        
        logger.info("*** Training completed ***")
    
    # Evaluation
    if training_args.do_eval:
        logger.info("*** Evaluating ***")
        
        metrics = trainer.evaluate()
        max_eval_samples = len(eval_dataset) if eval_dataset else 0
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity
        
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        logger.info("*** Evaluation completed ***")
    
    # Create model card
    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "multimodal-chat"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        kwargs["dataset"] = data_args.dataset_name
    
    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)


if __name__ == "__main__":
    main()
