#!/usr/bin/env python3
"""
Improved LoRA training script for floor plan generation
Optimized for small datasets (72 samples) with better regularization
"""
import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import gc
import random

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel, DDPMScheduler
from diffusers.optimization import get_scheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class FloorPlanDataset(Dataset):
    """Custom dataset for floor plan images with augmentation"""
    def __init__(self, csv_file, size=512, augment=True):
        self.data = pd.read_csv(csv_file)
        self.size = size
        self.augment = augment
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and preprocess image
        image = Image.open(row['file_name']).convert('RGB')
        
        # Light augmentation for training (rotation only for floor plans)
        if self.augment and random.random() < 0.5:
            # Random 90-degree rotations (architectural drawings are rotation-invariant)
            rotation = random.choice([0, 90, 180, 270])
            if rotation > 0:
                image = image.rotate(rotation, fillcolor='white')
        
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Add slight variations to the text prompt
        text = row['text']
        if self.augment and random.random() < 0.3:
            variations = [
                ", detailed architectural drawing",
                ", precise floor plan layout",
                ", clean line drawing",
                ", architectural blueprint design",
                ", technical floor plan illustration"
            ]
            text = text + random.choice(variations)
        
        return {
            "image": image,
            "text": text
        }

def collate_fn(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]
    return {"images": images, "texts": texts}

def train_lora_improved():
    # Configuration optimized for small dataset
    output_dir = Path("lora_model_improved")
    output_dir.mkdir(exist_ok=True)
    
    # Training parameters for small dataset
    num_epochs = 20  # More epochs but with early stopping
    batch_size = 1  # Keep small for memory
    gradient_accumulation_steps = 4  # Effective batch size of 4
    learning_rate = 1e-5  # Lower learning rate for stability
    warmup_steps = 50  # Gentle warmup
    save_every = 200
    
    # LoRA configuration for small dataset
    lora_rank = 8  # Moderate rank (not too low, not too high)
    lora_alpha = 16  # Alpha = 2 * rank for balanced adaptation
    lora_dropout = 0.1  # Add dropout for regularization
    
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
    
    # Load model components
    tokenizer = CLIPTokenizer.from_pretrained(model_id, subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(model_id, subfolder="text_encoder")
    vae = AutoencoderKL.from_pretrained(model_id, subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(model_id, subfolder="unet")
    noise_scheduler = DDPMScheduler.from_pretrained(model_id, subfolder="scheduler")
    
    # Configure noise scheduler for better training
    noise_scheduler.set_timesteps(1000)
    
    # Move to device and set to eval/train mode
    text_encoder.to(device)
    vae.to(device)
    unet.to(device)
    
    text_encoder.eval()
    vae.eval()
    
    # Freeze text encoder and VAE
    text_encoder.requires_grad_(False)
    vae.requires_grad_(False)
    
    # Configure LoRA for UNet
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_alpha,
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=lora_dropout,
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Enable gradient checkpointing for memory efficiency
    unet.enable_gradient_checkpointing()
    
    # Create dataset and dataloader
    print("Loading dataset...")
    train_dataset = FloorPlanDataset("prepared_data/train.csv", augment=True)
    val_dataset = FloorPlanDataset("prepared_data/val.csv", augment=False)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=0
    )
    
    # Optimizer with weight decay for regularization
    optimizer = torch.optim.AdamW(
        unet.parameters(), 
        lr=learning_rate,
        weight_decay=0.01,  # Add weight decay
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler
    lr_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_epochs * len(train_dataloader)
    )
    
    # Training loop with validation
    print(f"Starting training for {num_epochs} epochs...")
    global_step = 0
    best_val_loss = float('inf')
    patience = 5
    patience_counter = 0
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        # Training phase
        unet.train()
        epoch_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Accumulate gradients
            loss_accumulated = 0
            
            for accum_step in range(gradient_accumulation_steps):
                # Get batch data
                if batch_idx * gradient_accumulation_steps + accum_step >= len(train_dataloader):
                    break
                    
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
                    latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                bsz = latents.shape[0]
                
                # Sample random timesteps for each image
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (bsz,), device=device)
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Predict noise
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                loss = loss / gradient_accumulation_steps  # Scale loss
                loss_accumulated += loss.item()
                
                # Backward pass
                loss.backward()
            
            # Update weights
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)  # Gradient clipping
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            # Update metrics
            epoch_loss += loss_accumulated
            progress_bar.set_postfix({
                "loss": f"{loss_accumulated:.4f}",
                "lr": f"{optimizer.param_groups[0]['lr']:.2e}"
            })
            
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
        
        # Validation phase
        unet.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_dataloader, desc="Validation"):
                images = batch["images"].to(device)
                texts = batch["texts"]
                
                # Process similar to training
                text_inputs = tokenizer(texts, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_embeddings = text_encoder(text_inputs.input_ids.to(device))[0]
                
                latents = vae.encode(images).latent_dist.sample() * vae.config.scaling_factor
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=device).long()
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                noise_pred = unet(noisy_latents, timesteps, text_embeddings).sample
                loss = F.mse_loss(noise_pred, noise, reduction="mean")
                val_loss += loss.item()
        
        # Calculate average losses
        avg_train_loss = epoch_loss / len(train_dataloader)
        avg_val_loss = val_loss / len(val_dataloader)
        
        print(f"Epoch {epoch + 1} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            best_path = output_dir / "best"
            best_path.mkdir(exist_ok=True)
            unet.save_pretrained(best_path)
            print(f"New best model saved with val loss: {best_val_loss:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
    
    # Save final model
    final_path = output_dir / "final"
    final_path.mkdir(exist_ok=True)
    unet.save_pretrained(final_path)
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Save training info
    training_info = {
        "model_id": model_id,
        "lora_weights": str(final_path),
        "best_weights": str(output_dir / "best"),
        "training_steps": global_step,
        "final_train_loss": avg_train_loss,
        "best_val_loss": best_val_loss,
        "num_epochs_trained": epoch + 1,
        "hyperparameters": {
            "lora_rank": lora_rank,
            "lora_alpha": lora_alpha,
            "lora_dropout": lora_dropout,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "weight_decay": 0.01
        }
    }
    
    import json
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)
    
    print("\nTraining complete! Use the 'best' checkpoint for inference.")

if __name__ == "__main__":
    train_lora_improved()