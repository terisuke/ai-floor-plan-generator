#!/usr/bin/env python3
"""Debug script to test LoRA model output"""
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import numpy as np
import os

# Setup
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load base model
model_id = "stabilityai/stable-diffusion-2-1-base"
print(f"Loading base model: {model_id}")
pipe = StableDiffusionPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    use_safetensors=True
)

# Test base model first
print("\n=== Testing BASE model ===")
pipe = pipe.to(device)
with torch.no_grad():
    base_result = pipe(
        "architectural floor plan, simple room layout",
        num_inference_steps=10,
        guidance_scale=7.5,
        width=256,
        height=256
    )
base_image = base_result.images[0]
base_array = np.array(base_image)
print(f"Base model - Image stats: min={base_array.min()}, max={base_array.max()}, mean={base_array.mean():.2f}")
base_image.save("test_base_model.png")

# Now test with LoRA
print("\n=== Testing LoRA model ===")
lora_path = os.path.join(os.path.dirname(__file__), "../training/lora_model/final")
if os.path.exists(lora_path):
    print(f"Loading LoRA from: {lora_path}")
    # Reset pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True
    )
    
    # Load LoRA
    pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    pipe = pipe.to(device)
    
    # Test generation
    with torch.no_grad():
        lora_result = pipe(
            "architectural floor plan, simple room layout",
            num_inference_steps=10,
            guidance_scale=7.5,
            width=256,
            height=256
        )
    
    lora_image = lora_result.images[0]
    lora_array = np.array(lora_image)
    print(f"LoRA model - Image stats: min={lora_array.min()}, max={lora_array.max()}, mean={lora_array.mean():.2f}")
    lora_image.save("test_lora_model.png")
    
    # Check for NaN in latents
    print("\nChecking internal tensors...")
    
else:
    print("LoRA path not found!")

print("\nDone!")