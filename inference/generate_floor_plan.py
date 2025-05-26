#!/usr/bin/env python3
import argparse
import torch
from diffusers import StableDiffusionPipeline
from PIL import Image
import os
import sys

def generate_floor_plan(width: int, height: int, output_path: str):
    """Generate a floor plan image using Stable Diffusion"""
    
    try:
        # Check if MPS is available (for M2 Max)
        if torch.backends.mps.is_available():
            device = torch.device("mps")
            print("Using MPS (Metal Performance Shaders) for acceleration")
        elif torch.cuda.is_available():
            device = torch.device("cuda")
            print("Using CUDA for acceleration")
        else:
            device = torch.device("cpu")
            print("Using CPU (this will be slow)")
        
        # Load the base Stable Diffusion model
        model_id = "stabilityai/stable-diffusion-2-1-base"
        print(f"Loading model: {model_id}")
        
        pipe = StableDiffusionPipeline.from_pretrained(
            model_id,
            torch_dtype=torch.float16 if device.type != "cpu" else torch.float32,
            use_safetensors=True
        )
        pipe = pipe.to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        
        # Create prompt based on dimensions
        prompt = f"simple line drawing floor plan, {width}x{height} grid of rooms, black lines on white paper, architectural blueprint, walls and doors only, minimalist design, orthogonal top view, technical drawing"
        negative_prompt = "photo, realistic, 3d, perspective, shadows, colors, gradient, dark, black background, furniture, text, complex"
        
        print(f"Generating floor plan with prompt: {prompt}")
        
        # Generate image
        # Scale dimensions to pixels - use larger size for better quality
        pixel_width = min(width * 80, 1200)  # Increased from 40 to 80
        pixel_height = min(height * 80, 1000)  # Increased from 40 to 80
        
        # Ensure dimensions are multiples of 8 for stable diffusion
        pixel_width = (pixel_width // 8) * 8
        pixel_height = (pixel_height // 8) * 8
        
        # Set a seed for reproducibility
        generator = torch.Generator(device=device).manual_seed(42)
        
        image = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            width=pixel_width,
            height=pixel_height,
            num_inference_steps=30,
            guidance_scale=8.0,
            generator=generator
        ).images[0]
        
        # Post-process to ensure white background
        import numpy as np
        img_array = np.array(image)
        
        # Check if image is mostly dark (common issue)
        if np.mean(img_array) < 50:  # If average pixel value is very dark
            print("Warning: Generated image is too dark, inverting colors")
            # Invert the image
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
    parser = argparse.ArgumentParser(description='Generate floor plan using Stable Diffusion')
    parser.add_argument('--width', type=int, required=True, help='Grid width (1-50)')
    parser.add_argument('--height', type=int, required=True, help='Grid height (1-50)')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    
    args = parser.parse_args()
    
    # Validate dimensions
    if not (1 <= args.width <= 50 and 1 <= args.height <= 50):
        print("Error: Width and height must be between 1 and 50", file=sys.stderr)
        sys.exit(1)
    
    # Generate the floor plan
    success = generate_floor_plan(args.width, args.height, args.output)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()