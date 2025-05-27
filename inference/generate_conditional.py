#!/usr/bin/env python3
"""
Conditional floor plan generation with structural element control
"""
import torch
from diffusers import StableDiffusionPipeline
from peft import PeftModel
import json
import argparse
from pathlib import Path

def generate_conditional_floor_plan(
    prompt_base="architectural floor plan drawing",
    floor_type="1F",
    required_elements=None,
    layout_size=(8, 9),
    output_path="generated_plan.png",
    model_path="runwayml/stable-diffusion-v1-5",
    lora_path=None,
    num_inference_steps=50,
    guidance_scale=7.5,
    seed=42
):
    """
    Generate floor plan with specific structural requirements
    
    Args:
        prompt_base: Base description
        floor_type: "1F" or "2F"
        required_elements: List of required elements ["stair", "entrance", "balcony"]
        layout_size: (width_grids, height_grids)
        output_path: Output file path
        model_path: Base model path
        lora_path: Path to LoRA weights
        num_inference_steps: Number of denoising steps
        guidance_scale: Classifier-free guidance scale
        seed: Random seed for reproducibility
    """
    
    if required_elements is None:
        required_elements = []
    
    # Build conditional prompt
    conditions = [f"floor_type:{floor_type}"]
    conditions.append(f"grid_size:{layout_size[0]}x{layout_size[1]}")
    
    element_descriptions = []
    for element in required_elements:
        conditions.append(f"has_{element}")
        if element == "stair":
            element_descriptions.append("staircase for vertical circulation")
        elif element == "entrance":
            element_descriptions.append("main entrance with foyer")
        elif element == "balcony":
            element_descriptions.append("outdoor balcony space")
    
    # Combine into full prompt
    full_prompt = f"{prompt_base}, {floor_type} floor"
    if element_descriptions:
        full_prompt += f", with {', '.join(element_descriptions)}"
    full_prompt += f", grid layout {layout_size[0]}x{layout_size[1]} cells"
    full_prompt += ", technical blueprint, black lines on white background"
    full_prompt += ", professional CAD style drawing"
    full_prompt += f", {' '.join(conditions)}"
    
    print(f"Generating with prompt: {full_prompt}")
    
    # Check for CUDA/MPS availability
    if torch.cuda.is_available():
        device = "cuda"
        torch_dtype = torch.float16
    elif hasattr(torch.backends.mps, 'is_available') and torch.backends.mps.is_available():
        device = "mps"
        torch_dtype = torch.float32  # MPS doesn't support float16 well
    else:
        device = "cpu"
        torch_dtype = torch.float32
    
    print(f"Using device: {device}")
    
    # Load pipeline
    pipe = StableDiffusionPipeline.from_pretrained(
        model_path,
        torch_dtype=torch_dtype,
        safety_checker=None,
        requires_safety_checker=False
    )
    
    # Load LoRA weights if provided
    if lora_path and Path(lora_path).exists():
        print(f"Loading LoRA weights from: {lora_path}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, lora_path)
    
    pipe = pipe.to(device)
    
    # Generate image
    generator = torch.Generator(device=device).manual_seed(seed)
    
    # Calculate image dimensions based on grid size
    image_width = layout_size[0] * 40  # 40px per grid unit
    image_height = layout_size[1] * 40
    
    # Ensure dimensions are multiples of 8
    image_width = (image_width // 8) * 8
    image_height = (image_height // 8) * 8
    
    # Cap at maximum size
    image_width = min(image_width, 1024)
    image_height = min(image_height, 1024)
    
    print(f"Generating {image_width}x{image_height} image...")
    
    image = pipe(
        prompt=full_prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
        width=image_width,
        height=image_height
    ).images[0]
    
    # Save image
    image.save(output_path)
    print(f"Generated image saved to: {output_path}")
    
    # Save generation metadata
    metadata = {
        "prompt": full_prompt,
        "floor_type": floor_type,
        "required_elements": required_elements,
        "layout_size": layout_size,
        "conditions": conditions,
        "model_path": model_path,
        "lora_path": lora_path,
        "num_inference_steps": num_inference_steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
        "image_dimensions": [image_width, image_height]
    }
    
    metadata_path = Path(output_path).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved to: {metadata_path}")
    
    return full_prompt

def main():
    parser = argparse.ArgumentParser(description="Generate floor plans with structural element control")
    parser.add_argument("--floor", type=str, default="1F", choices=["1F", "2F"],
                        help="Floor type (1F or 2F)")
    parser.add_argument("--width", type=int, default=8,
                        help="Grid width (default: 8)")
    parser.add_argument("--height", type=int, default=9,
                        help="Grid height (default: 9)")
    parser.add_argument("--elements", nargs="+", 
                        choices=["stair", "entrance", "balcony"],
                        help="Required structural elements")
    parser.add_argument("--output", type=str, default="conditional_plan.png",
                        help="Output file path")
    parser.add_argument("--lora-path", type=str, 
                        default="../training/lora_model_improved/best",
                        help="Path to LoRA weights")
    parser.add_argument("--steps", type=int, default=50,
                        help="Number of inference steps")
    parser.add_argument("--guidance", type=float, default=7.5,
                        help="Guidance scale")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    
    args = parser.parse_args()
    
    generate_conditional_floor_plan(
        floor_type=args.floor,
        required_elements=args.elements or [],
        layout_size=(args.width, args.height),
        output_path=args.output,
        lora_path=args.lora_path,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed
    )

if __name__ == "__main__":
    main()