#!/usr/bin/env python3
"""
Simplified LoRA training script for floor plan generation
Optimized for M2 Max with reduced memory usage
"""
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import gc

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader

class FloorPlanDataset(Dataset):
    """Custom dataset for floor plan images"""
    def __init__(self, csv_file, size=512):
        self.data = pd.read_csv(csv_file)
        self.size = size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and preprocess image
        image = Image.open(row['file_name']).convert('RGB')
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        return {
            "image": image,
            "text": row['text']
        }

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    return {"images": images, "texts": texts}

def train_lora_simple():
    # Configuration
    output_dir = Path("lora_model")
    output_dir.mkdir(exist_ok=True)
    
    # Training parameters
    num_epochs = 5  # Reduced for faster training
    batch_size = 1
    learning_rate = 5e-5
    save_every = 100
    
    # Set device
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("Using MPS (Metal Performance Shaders)")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    # Load base model
    print("Loading Stable Diffusion model...")
    model_id = "stabilityai/stable-diffusion-2-1-base"
    
    # Load model components separately to save memory
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    
    # Move to device and set to eval/train mode
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    
    text_encoder.eval()
    vae.eval()
    
    # Configure LoRA for UNet only
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=4,  # Very low rank for memory efficiency
        lora_alpha=4,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=0.0,
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Create dataset and dataloader
    print("Loading dataset...")
    train_dataset = FloorPlanDataset("prepared_data/train.csv")
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(unet.parameters(), lr=learning_rate)
    
    # Training loop
    print(f"Starting training for {num_epochs} epochs...")
    global_step = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        epoch_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            images = batch["images"].to(device)
            texts = batch["texts"]
            
            # Encode text
            text_inputs = tokenizer(
                texts,
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            )
            with torch.no_grad():
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
            
            # Encode images to latents
            with torch.no_grad():
                latents = vae.encode(images).latent_dist.sample()
                latents = latents * 0.18215  # Scaling factor
            
            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, 1000, (batch_size,), device=device).long()
            
            # Add noise to latents (forward diffusion)
            noisy_latents = latents * (1 - timesteps.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / 1000) + \
                           noise * (timesteps.float().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1) / 1000)
            
            # Predict noise
            noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
            
            # Calculate loss
            loss = torch.nn.functional.mse_loss(noise_pred, noise)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss.item()
            progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            global_step += 1
            
            # Save checkpoint
            if global_step % save_every == 0:
                checkpoint_path = output_dir / f"checkpoint-{global_step}"
                checkpoint_path.mkdir(exist_ok=True)
                unet.save_pretrained(checkpoint_path)
                print(f"\nSaved checkpoint to {checkpoint_path}")
            
            # Clear cache periodically
            if batch_idx % 10 == 0:
                if torch.backends.mps.is_available():
                    torch.mps.empty_cache()
                gc.collect()
        
        # Print epoch summary
        avg_loss = epoch_loss / len(train_dataloader)
        print(f"Epoch {epoch + 1} - Average Loss: {avg_loss:.4f}")
    
    # Save final model
    final_path = output_dir / "final"
    final_path.mkdir(exist_ok=True)
    unet.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Save adapter config for easy loading
    adapter_config = {
        "model_id": model_id,
        "lora_weights": str(final_path),
        "training_steps": global_step,
        "final_loss": avg_loss
    }
    
    import json
    with open(output_dir / "adapter_config.json", "w") as f:
        json.dump(adapter_config, f, indent=2)

if __name__ == "__main__":
    train_lora_simple()