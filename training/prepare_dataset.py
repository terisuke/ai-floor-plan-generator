#!/usr/bin/env python3
"""
Prepare floor plan dataset for LoRA training
Creates a metadata file with image paths and captions
"""
import json
import os
from pathlib import Path
import pandas as pd

def prepare_dataset():
    dataset_dir = Path("../dataset")
    output_dir = Path("prepared_data")
    output_dir.mkdir(exist_ok=True)
    
    # Collect all floor plan images
    metadata = []
    
    for png_file in sorted(dataset_dir.glob("*.png")):
        json_file = dataset_dir / f"{png_file.stem}.json"
        
        if json_file.exists():
            # Read JSON metadata
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Extract floor number from filename
            floor_match = "1f" if "1f" in png_file.name else "2f" if "2f" in png_file.name else "unknown"
            
            # Create detailed caption for the image
            caption = f"architectural floor plan drawing, {floor_match} floor, technical blueprint, black lines on white background, room layout with walls and doors, professional CAD style drawing, grid size {data.get('grid_mm', 910)}mm"
            
            metadata.append({
                "file_name": str(png_file.absolute()),
                "text": caption,
                "floor": data.get("floor", floor_match.upper()),
                "grid_px": data.get("grid_px", 107.5),
                "grid_mm": data.get("grid_mm", 910)
            })
    
    # Save metadata as CSV
    df = pd.DataFrame(metadata)
    metadata_path = output_dir / "metadata.csv"
    df.to_csv(metadata_path, index=False)
    print(f"Created metadata file with {len(metadata)} entries: {metadata_path}")
    
    # Also save as JSON for flexibility
    json_path = output_dir / "metadata.json"
    with open(json_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Also saved as JSON: {json_path}")
    
    # Create train/validation split (90/10)
    train_size = int(len(metadata) * 0.9)
    train_df = df[:train_size]
    val_df = df[train_size:]
    
    train_df.to_csv(output_dir / "train.csv", index=False)
    val_df.to_csv(output_dir / "val.csv", index=False)
    
    print(f"Train set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    return metadata_path

if __name__ == "__main__":
    prepare_dataset()