#!/usr/bin/env python3
"""
Floor plan generation with LoRA-finetuned model
"""
import argparse
import torch
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
from peft import PeftModel, LoraConfig
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
            try:
                # Load LoRA weights using PEFT
                pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
                print("LoRA weights loaded successfully using PEFT!")
                # Don't merge - keep LoRA as adapter
                # pipe.unet = pipe.unet.merge_and_unload()
                # print("LoRA weights merged into base model")
            except Exception as e:
                print(f"Error loading LoRA weights with PEFT: {e}")
                print("Continuing with base model")
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
        
        with torch.no_grad():
            result = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                width=pixel_width,
                height=pixel_height,
                num_inference_steps=30,
                guidance_scale=7.5,
                generator=generator
            )
        
        image = result.images[0]
        
        # Convert PIL image to numpy array for analysis
        img_array = np.array(image)
        
        # Check if the image has valid data
        if np.isnan(img_array).any() or np.isinf(img_array).any():
            print("ERROR: Image contains NaN or Inf values!")
            # Try to salvage the image
            img_array = np.nan_to_num(img_array, nan=128.0, posinf=255.0, neginf=0.0)
            
        # Check the actual value range
        if img_array.max() <= 1.0 and img_array.min() >= 0.0:
            print("Debug - Image appears to be in [0,1] range, converting to [0,255]")
            img_array = (img_array * 255).astype(np.uint8)
        else:
            # Ensure values are in valid range [0, 255]
            img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # Check if image is mostly uniform color
        std_dev = np.std(img_array)
        
        if std_dev < 5:
            print(f"ERROR: Image has very low variance (std={std_dev}), might be blank!")
            
            # Fallback: Try generating without LoRA
            if lora_path and os.path.exists(lora_path):
                print("Falling back to base model without LoRA...")
                
                # Reload base model
                pipe = StableDiffusionPipeline.from_pretrained(
                    model_id,
                    torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
                    use_safetensors=True
                )
                pipe = pipe.to(device)
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
                
                if hasattr(pipe, "enable_attention_slicing"):
                    pipe.enable_attention_slicing()
                
                # Generate with base model
                with torch.no_grad():
                    result = pipe(
                        prompt=prompt,
                        negative_prompt=negative_prompt,
                        width=pixel_width,
                        height=pixel_height,
                        num_inference_steps=30,
                        guidance_scale=8.0,  # Slightly higher for base model
                        generator=generator
                    )
                
                image = result.images[0]
                img_array = np.array(image)
                print(f"Fallback - Image stats: min={img_array.min()}, max={img_array.max()}, mean={img_array.mean():.2f}")
                
                # Check if fallback image needs inversion
                if np.mean(img_array) < 50:
                    print("Fallback image is dark, inverting colors")
                    img_array = 255 - img_array
                    image = Image.fromarray(img_array.astype('uint8'))
        
        # Check if image needs inversion
        mean_value = np.mean(img_array)
        if mean_value < 50:
            print(f"Warning: Generated image is too dark (mean={mean_value}), inverting colors")
            img_array = 255 - img_array
        elif mean_value > 240:
            print(f"Warning: Generated image is very bright (mean={mean_value}), checking if it's blank")
            # Check if it's actually a blank white image
            if std_dev < 10:
                print("ERROR: Image appears to be blank white!")
        
        # Convert back to PIL Image
        image = Image.fromarray(img_array)
        
        # Save the image
        print(f"Attempting to save image to: {output_path}")
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        image.save(output_path)
        print(f"Floor plan saved to: {output_path}")
        
        return True
        
    except Exception as e:
        import traceback
        print(f"Error generating floor plan: {str(e)}", file=sys.stderr)
        print(f"Full error traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
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
        # Try improved model first
        improved_lora = os.path.join(os.path.dirname(__file__), "../training/lora_model_improved/best")
        default_lora = os.path.join(os.path.dirname(__file__), "../training/lora_model/final")
        
        if os.path.exists(improved_lora):
            args.lora = improved_lora
            print(f"Using improved LoRA model from: {improved_lora}")
        elif os.path.exists(default_lora):
            args.lora = default_lora
            print(f"Using default LoRA model from: {default_lora}")
    
    # Generate the floor plan
    success = generate_floor_plan_with_lora(args.width, args.height, args.output, args.lora)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()