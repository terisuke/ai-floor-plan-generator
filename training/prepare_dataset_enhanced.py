#!/usr/bin/env python3
"""
Enhanced dataset preparation for structural elements-based training
"""
import json
import os
from pathlib import Path
import pandas as pd

def create_detailed_caption(json_data):
    """Create rich captions from structural elements data"""
    
    # Base description
    floor = json_data.get("floor", "unknown")
    width = json_data.get("layout_bounds", {}).get("width_grids", 0)
    height = json_data.get("layout_bounds", {}).get("height_grids", 0)
    
    caption_parts = [
        f"architectural floor plan drawing, {floor} floor",
        f"grid layout {width}x{height} cells",
        "technical blueprint, black lines on white background"
    ]
    
    # Add structural elements information
    elements = json_data.get("structural_elements", [])
    element_summary = json_data.get("element_summary", {})
    
    if elements:
        element_types = []
        if element_summary.get("stair_count", 0) > 0:
            element_types.append(f"{element_summary['stair_count']} staircase")
        if element_summary.get("entrance_count", 0) > 0:
            element_types.append(f"{element_summary['entrance_count']} entrance")
        if element_summary.get("balcony_count", 0) > 0:
            element_types.append(f"{element_summary['balcony_count']} balcony")
        
        if element_types:
            caption_parts.append(f"with {', '.join(element_types)}")
    
    # Add zone information if available
    zones = json_data.get("major_zones", [])
    if zones:
        total_area = sum(zone.get("approximate_grids", 0) for zone in zones)
        caption_parts.append(f"total area {total_area} grid units")
    
    caption_parts.append("professional CAD style drawing")
    
    return ", ".join(caption_parts)

def create_conditional_prompts(json_data):
    """Create conditioning prompts for controlled generation"""
    
    conditions = []
    
    # Floor type condition
    floor = json_data.get("floor", "1F")
    conditions.append(f"floor_type:{floor}")
    
    # Layout dimensions
    bounds = json_data.get("layout_bounds", {})
    if bounds:
        conditions.append(f"grid_size:{bounds.get('width_grids', 0)}x{bounds.get('height_grids', 0)}")
    
    # Element requirements
    element_summary = json_data.get("element_summary", {})
    if element_summary.get("stair_count", 0) > 0:
        conditions.append("has_stairs")
    if element_summary.get("entrance_count", 0) > 0:
        conditions.append("has_entrance")
    if element_summary.get("balcony_count", 0) > 0:
        conditions.append("has_balcony")
    
    # Area constraint
    zones = json_data.get("major_zones", [])
    if zones:
        total_area = sum(zone.get("approximate_grids", 0) for zone in zones)
        conditions.append(f"area:{total_area}")
    
    return "|".join(conditions)

def prepare_enhanced_dataset():
    dataset_dir = Path("../dataset")
    output_dir = Path("prepared_data")
    output_dir.mkdir(exist_ok=True)
    
    metadata = []
    
    for png_file in sorted(dataset_dir.glob("*.png")):
        json_file = dataset_dir / f"{png_file.stem}.json"
        
        if json_file.exists():
            with open(json_file, 'r') as f:
                data = json.load(f)
            
            # Check if this is new format (has structural_elements)
            if "structural_elements" in data:
                # New enhanced format
                caption = create_detailed_caption(data)
                conditions = create_conditional_prompts(data)
                
                metadata.append({
                    "file_name": str(png_file.absolute()),
                    "text": caption,
                    "conditions": conditions,
                    "floor": data.get("floor", "1F"),
                    "grid_px": data.get("grid_px", 107.5),
                    "grid_mm": data.get("grid_mm", 910),
                    "layout_width": data.get("layout_bounds", {}).get("width_grids", 0),
                    "layout_height": data.get("layout_bounds", {}).get("height_grids", 0),
                    "total_elements": data.get("element_summary", {}).get("total_elements", 0),
                    "has_stairs": data.get("element_summary", {}).get("stair_count", 0) > 0,
                    "has_entrance": data.get("element_summary", {}).get("entrance_count", 0) > 0,
                    "has_balcony": data.get("element_summary", {}).get("balcony_count", 0) > 0,
                    "data_version": "v2.0_structural"
                })
            else:
                # Legacy format - create basic caption
                floor_match = "1f" if "1f" in png_file.name else "2f" if "2f" in png_file.name else "unknown"
                caption = f"architectural floor plan drawing, {floor_match} floor, technical blueprint, black lines on white background, professional CAD style drawing"
                
                metadata.append({
                    "file_name": str(png_file.absolute()),
                    "text": caption,
                    "conditions": f"floor_type:{data.get('floor', floor_match.upper())}",
                    "floor": data.get("floor", floor_match.upper()),
                    "grid_px": data.get("grid_px", 107.5),
                    "grid_mm": data.get("grid_mm", 910),
                    "layout_width": 0,
                    "layout_height": 0,
                    "total_elements": 0,
                    "has_stairs": False,
                    "has_entrance": False,
                    "has_balcony": False,
                    "data_version": "v1.0_legacy"
                })
    
    # Save enhanced metadata
    df = pd.DataFrame(metadata)
    
    # Separate new and legacy data
    new_data = df[df['data_version'] == 'v2.0_structural']
    legacy_data = df[df['data_version'] == 'v1.0_legacy']
    
    print(f"Enhanced dataset created:")
    print(f"  - New structural format: {len(new_data)} samples")
    print(f"  - Legacy format: {len(legacy_data)} samples")
    print(f"  - Total: {len(metadata)} samples")
    
    # Save complete dataset
    df.to_csv(output_dir / "metadata_enhanced.csv", index=False)
    
    # Create train/val split prioritizing new data
    if len(new_data) > 0:
        # Use mostly new data for training
        train_ratio = 0.9
        new_train_size = int(len(new_data) * train_ratio)
        
        train_df = pd.concat([
            new_data[:new_train_size],
            legacy_data[:int(len(legacy_data) * 0.3)]  # Limited legacy data
        ])
        
        val_df = pd.concat([
            new_data[new_train_size:],
            legacy_data[int(len(legacy_data) * 0.3):]
        ])
    else:
        # Fallback to legacy data only
        train_size = int(len(df) * 0.9)
        train_df = df[:train_size]
        val_df = df[train_size:]
    
    train_df.to_csv(output_dir / "train_enhanced.csv", index=False)
    val_df.to_csv(output_dir / "val_enhanced.csv", index=False)
    
    print(f"Train set: {len(train_df)} images")
    print(f"Validation set: {len(val_df)} images")
    
    return output_dir / "metadata_enhanced.csv"

if __name__ == "__main__":
    prepare_enhanced_dataset()