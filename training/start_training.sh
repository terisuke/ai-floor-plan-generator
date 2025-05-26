#!/bin/bash
# LoRA Training Launcher Script

echo "==================================="
echo "Floor Plan LoRA Training"
echo "==================================="

# Check if we're in the training directory
if [ ! -f "train_lora_simple.py" ]; then
    echo "Error: Please run this script from the training directory"
    exit 1
fi

# Check if dataset is prepared
if [ ! -f "prepared_data/train.csv" ]; then
    echo "Preparing dataset..."
    python3 prepare_dataset.py
fi

# Start training
echo "Starting LoRA training..."
echo "This will train for 5 epochs on your floor plan dataset."
echo "Expected time: 1-2 hours on M2 Max"
echo ""

# Run the training script
python3 train_lora_simple.py

echo ""
echo "Training complete! You can now use the LoRA model for inference."
echo "To use it, update the backend to use generate_floor_plan_lora.py"