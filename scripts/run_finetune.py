#!/usr/bin/env python3
"""
GPT-OSS-Vision Fine-tuning Script for Linux
Author: Adapted from Dustin Loring's Windows batch script
"""

import os
import subprocess
import sys
import venv
import platform


def run(cmd, **kwargs):
    """Run a shell command and raise on error."""
    print(f"\n>> {cmd}")
    subprocess.run(cmd, shell=True, check=True, **kwargs)


def check_python():
    """Ensure Python is available."""
    try:
        subprocess.run(["python3", "--version"], check=True)
    except subprocess.CalledProcessError:
        print("Error: Python is not installed or not in PATH")
        sys.exit(1)


def create_venv():
    """Create virtual environment if it doesn't exist."""
    if not os.path.exists("venv"):
        print("Creating virtual environment...")
        venv.EnvBuilder(with_pip=True).create("venv")


def activate_venv():
    """Activate the virtual environment in current process."""
    if platform.system() == "Windows":
        activate_script = os.path.join("venv", "Scripts", "activate_this.py")
    else:
        activate_script = os.path.join("venv", "bin", "activate_this.py")

    if not os.path.exists(activate_script):
        print("Error: Could not find activation script.")
        sys.exit(1)

    with open(activate_script) as f:
        exec(f.read(), dict(__file__=activate_script))


def install_dependencies():
    """Install all required dependencies."""
    run("python3 -m pip install --upgrade pip")
    run("pip install torch torchvision --index-url https://download.pytorch.org/whl/cu129")

    # Install Linux/Mac-specific deps
    if not sys.platform.startswith("win"):
        run("pip install flash-attn --no-build-isolation")
        run("pip install deepspeed")

    run(
        'pip install transformers datasets accelerate "Pillow>=10.0.1,<=15.0" '
        'numpy requests tqdm safetensors pyyaml wandb tensorboard black hf_xet flake8 '
        'scipy "pytest>=7.2.0"'
    )


def check_cuda():
    """Check CUDA availability."""
    subprocess.run(
        [
            "python3",
            "-c",
            'import torch; print(f"CUDA available: {torch.cuda.is_available()}"); '
            'print(f"CUDA devices: {torch.cuda.device_count()}")',
        ],
        check=True,
    )


def run_training():
    """Run the fine-tuning script."""
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    run("python3 finetune_gpt_oss_vision.py config_gpt_oss_vision.json")


if __name__ == "__main__":
    print("Starting GPT-OSS-Vision Fine-tuning...")
    print("=====================================")

    check_python()
    create_venv()
    activate_venv()
    install_dependencies()
    check_cuda()
    run_training()

    print("\nTraining completed!")
