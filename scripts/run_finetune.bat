@echo off
REM GPT-OSS-Vision Fine-tuning Script for Windows
REM Author: Dustin Loring <dustinwloring1988@gmail.com>

echo Starting GPT-OSS-Vision Fine-tuning...
echo =====================================

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    pause
    exit /b 1
)

REM Check if virtual environment exists, create if not
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
python.exe -m pip install --upgrade pip
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu129

REM Only install flash-attn and deepspeed if NOT on Windows
python -c "import sys; exit(0) if sys.platform.startswith('win') else exit(1)"
if errorlevel 1 (
    pip install flash-attn --no-build-isolation
    pip install deepspeed
)

pip install transformers datasets accelerate "Pillow>=10.0.1,<=15.0" numpy requests tqdm safetensors pyyaml wandb tensorboard black hf_xet flake8 scipy "pytest>=7.2.0"

REM Check for CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA devices: {torch.cuda.device_count()}')"

REM Set environment variables for training
set TOKENIZERS_PARALLELISM=false
set CUDA_VISIBLE_DEVICES=0

REM Run the training script
echo Starting training...
python finetune_gpt_oss_vision.py config_gpt_oss_vision.json

echo Training completed!
pause
