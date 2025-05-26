#!/usr/bin/env python3
"""
Floor plan generation with LoRA-finetuned model
"""
import argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel
from PIL import Image
import os
import sys
import json
import numpy as np

def generate_floor_plan_with_lora(width: int, height: int, output_path: str, lora_path: str = None):
    """Generate a floor plan image using Stable Diffusion with optional LoRA weights"""
    
    try:
        # Check if MPS is available
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA for acceleration")
        else:
            device = torch.device("cpu")
            print("Using CPU (this will be slow)")
        
        # Load the base model
        model_id = "stabilityai/stable-diffusion-2-1-base"
        print(f"Loading model: {model_id}")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
            use_safetensors=True
        )
        
        # Load LoRA weights if available
        if lora_path and os.path.exists(lora_path):
            print(f"Loading LoRA weights from: {lora_path}")
            # Load the LoRA weights into the UNet
            pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
            print("LoRA weights loaded successfully!")
        else:
            print("No LoRA weights found, using base model")
        
        # Move pipeline to device
        pipe = pipe.to(device)
        
        # Use optimized scheduler
        pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        
        # Create optimized prompt for floor plan generation
        prompt = f"architectural floor plan, {width}x{height} room grid, technical blueprint, black lines on white background, room layout with walls and doors, professional CAD style drawing, clean vector lines, orthogonal top view"
        negative_prompt = "3d render, perspective, furniture, people, text labels, shadows, gradient, colors, dark background, complex details, realistic photo"
        
        print(f"Generating floor plan with prompt: {prompt}")
        
        # Calculate image dimensions
        pixel_width = min(width * 80, 1024)
        pixel_height = min(height * 80, 1024)
        
        # Ensure dimensions are multiples of 8
        pixel_width = (pixel_width // 8) * 8
        pixel_height = (pixel_height // 8) * 8
        
        # Generate image
        generator = torch.Generator(device=device).manual_seed(42)
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=pixel_width,
            height=pixel_height,
            num_inference_steps=30,
            guidance_scale=7.5,
            generator=generator
        ).images[0]
        
        # Post-process to ensure white background
        img_array = np.array(image)
        
        # Check if image is mostly dark
        if np.mean(img_array) < 50:
            print("Warning: Generated image is too dark, inverting colors")
            img_array = 255 - img_array
            image = Image.fromarray(img_array.astype('uint8'))
        
        # Save the image
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        print(f"Floor plan saved to: {output_path}")
        
        return True
        
    except Exception as e:
        print(f"Error generating floor plan: {str(e)}", file=sys.stderr)
        return False

def main():
    parser = argparse.ArgumentParser(description='Generate floor plan using Stable Diffusion with LoRA')
    parser.add_argument('--width', type=int, required=True, help='Grid width (1-50)')
    parser.add_argument('--height', type=int, required=True, help='Grid height (1-50)')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    parser.add_argument('--lora', type=str, help='Path to LoRA weights directory')
    
    args = parser.parse_args()
    
    # Validate dimensions
    if not (1 <= args.width <= 50 and 1 <= args.height <= 50):
        print("Error: Width and height must be between 1 and 50", file=sys.stderr)
        sys.exit(1)
    
    # Default LoRA path if not specified
    if not args.lora:
        default_lora = os.path.join(os.path.dirname(__file__), "../training/lora_model/final")
        if os.path.exists(default_lora):
            args.lora = default_lora
    
    # Generate the floor plan
    success = generate_floor_plan_with_lora(args.width, args.height, args.output, args.lora)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()