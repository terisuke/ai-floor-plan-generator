#!/bin/bash
# Improved training script for floor plan LoRA

echo "Starting improved LoRA training for floor plans..."
echo "This training is optimized for small datasets with better regularization"
echo ""

# Activate virtual environment if exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

# Check if dataset is prepared
if [ ! -f "prepared_data/train.csv" ]; then
    echo "Dataset not prepared. Running preparation script first..."
    python3 prepare_dataset.py
fi

# Clean up any previous improved training
if [ -d "lora_model_improved" ]; then
    echo "Found existing improved model. Backing up..."
    mv lora_model_improved lora_model_improved_backup_$(date +%Y%m%d_%H%M%S)
fi

# Run improved training
echo "Starting training with improved hyperparameters..."
python3 train_lora_improved.py

echo ""
echo "Training complete! The model is saved in lora_model_improved/"
echo "Use the 'best' checkpoint for best results"