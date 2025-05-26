#!/usr/bin/env python3
"""
Simple floor plan generator that creates basic grid layouts
This is a temporary solution while we fine-tune the AI model
"""
import argparse
from PIL import Image, ImageDraw
import random
import os

def generate_simple_floor_plan(width: int, height: int, output_path: str):
    """Generate a simple floor plan with grid layout"""
    
    # Calculate image size (80px per grid unit)
    cell_size = 80
    img_width = width * cell_size
    img_height = height * cell_size
    
    # Create white background
    image = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(image)
    
    # Draw outer walls (thick black lines)
    wall_thickness = 3
    draw.rectangle([0, 0, img_width-1, img_height-1], 
                   outline='black', width=wall_thickness)
    
    # Draw interior walls (grid)
    for i in range(1, width):
        x = i * cell_size
        # Add some random door openings
        if random.random() > 0.3:  # 70% chance of wall
            # Draw wall with door opening
            door_width = cell_size // 3
            door_pos = random.randint(0, height - 1) * cell_size + cell_size // 2
            draw.line([(x, 0), (x, door_pos - door_width // 2)], 
                     fill='black', width=2)
            draw.line([(x, door_pos + door_width // 2), (x, img_height)], 
                     fill='black', width=2)
        else:
            # Full wall
            draw.line([(x, 0), (x, img_height)], fill='black', width=2)
    
    for i in range(1, height):
        y = i * cell_size
        # Add some random door openings
        if random.random() > 0.3:  # 70% chance of wall
            # Draw wall with door opening
            door_width = cell_size // 3
            door_pos = random.randint(0, width - 1) * cell_size + cell_size // 2
            draw.line([(0, y), (door_pos - door_width // 2, y)], 
                     fill='black', width=2)
            draw.line([(door_pos + door_width // 2, y), (img_width, y)], 
                     fill='black', width=2)
        else:
            # Full wall
            draw.line([(0, y), (img_width, y)], fill='black', width=2)
    
    # Add some random door arcs for style
    for _ in range(width * height // 4):
        x = random.randint(0, width - 1) * cell_size
        y = random.randint(0, height - 1) * cell_size
        # Draw door arc
        arc_size = cell_size // 3
        draw.arc([x + 10, y + 10, x + 10 + arc_size, y + 10 + arc_size], 
                 0, 90, fill='black', width=1)
    
    # Save the image
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    image.save(output_path)
    print(f"Simple floor plan saved to: {output_path}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate simple floor plan')
    parser.add_argument('--width', type=int, required=True, help='Grid width (1-50)')
    parser.add_argument('--height', type=int, required=True, help='Grid height (1-50)')
    parser.add_argument('--output', type=str, required=True, help='Output file path')
    
    args = parser.parse_args()
    
    # Validate dimensions
    if not (1 <= args.width <= 50 and 1 <= args.height <= 50):
        print("Error: Width and height must be between 1 and 50")
        exit(1)
    
    # Generate the floor plan
    success = generate_simple_floor_plan(args.width, args.height, args.output)
    exit(0 if success else 1)

if __name__ == "__main__":
    main()