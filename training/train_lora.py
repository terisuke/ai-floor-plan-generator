#!/usr/bin/env python3
"""
LoRA training script for floor plan generation
Fine-tunes Stable Diffusion 2.1 on architectural floor plans
"""
import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
from datetime import datetime
from tqdm import tqdm

from diffusers import StableDiffusionPipeline, DDPMScheduler
from transformers import CLIPTextModel, CLIPTokenizer
from peft import LoraConfig, get_peft_model, TaskType
from accelerate import Accelerator
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F

class FloorPlanDataset(Dataset):
    """Custom dataset for floor plan images"""
    def __init__(self, csv_file, tokenizer, size=512):
        self.data = pd.read_csv(csv_file)
        self.tokenizer = tokenizer
        self.size = size
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load and preprocess image
        image = Image.open(row['file_name']).convert('RGB')
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        
        # Convert to tensor and normalize to [-1, 1]
        image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 127.5 - 1.0
        
        # Tokenize caption
        text_inputs = self.tokenizer(
            row['text'],
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        )
        
        return {
            "pixel_values": image,
            "input_ids": text_inputs.input_ids[0]
        }

def train_lora():
    # Configuration
    config = {
        "model_id": "stabilityai/stable-diffusion-2-1-base",
        "output_dir": "lora_model",
        "train_batch_size": 1,  # Small batch size for M2 Max
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 10,
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 100,
        "save_steps": 500,
        "logging_steps": 50,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "lora_rank": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
        "target_modules": ["to_k", "to_q", "to_v", "to_out.0"],
        "seed": 42
    }
    
    # Set up accelerator
    accelerator = Accelerator(
        mixed_precision=config["mixed_precision"],
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
    )
    
    # Set device
    device = accelerator.device
    
    # Load model components
    print("Loading model components...")
    pipeline = StableDiffusionPipeline.from_pretrained(
        config["model_id"],
        torch_dtype=torch.float16 if config["mixed_precision"] == "fp16" else torch.float32,
        use_safetensors=True
    )
    
    # Extract components
    vae = pipeline.vae
    text_encoder = pipeline.text_encoder
    tokenizer = pipeline.tokenizer
    unet = pipeline.unet
    noise_scheduler = DDPMScheduler.from_pretrained(config["model_id"], subfolder="scheduler")
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=config["target_modules"],
        lora_dropout=config["lora_dropout"],
    )
    
    # Add LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if config["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    
    # Load dataset
    print("Loading dataset...")
    train_dataset = FloorPlanDataset("prepared_data/train.csv", tokenizer)
    val_dataset = FloorPlanDataset("prepared_data/val.csv", tokenizer)
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=0  # Set to 0 for MPS
    )
    
    # Set up optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["learning_rate"],
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-08
    )
    
    # Prepare for training
    unet, optimizer, train_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader
    )
    
    # Move other components to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Training loop
    print(f"Starting training for {config['num_train_epochs']} epochs...")
    global_step = 0
    
    for epoch in range(config["num_train_epochs"]):
        unet.train()
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{config['num_train_epochs']}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # Convert images to latent space
                latents = vae.encode(batch["pixel_values"]).latent_dist.sample()
                latents = latents * vae.config.scaling_factor
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (latents.shape[0],), device=latents.device)
                timesteps = timesteps.long()
                
                # Add noise to latents
                noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)
                
                # Get text embeddings
                encoder_hidden_states = text_encoder(batch["input_ids"])[0]
                
                # Predict the noise residual
                model_pred = unet(noisy_latents, timesteps, encoder_hidden_states).sample
                
                # Calculate loss
                loss = F.mse_loss(model_pred.float(), noise.float(), reduction="mean")
                
                # Backward pass
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            global_step += 1
            
            # Save checkpoint
            if global_step % config["save_steps"] == 0:
                save_path = os.path.join(config["output_dir"], f"checkpoint-{global_step}")
                os.makedirs(save_path, exist_ok=True)
                unet.save_pretrained(save_path)
                print(f"Saved checkpoint to {save_path}")
    
    # Save final model
    final_path = os.path.join(config["output_dir"], "final")
    os.makedirs(final_path, exist_ok=True)
    unet.save_pretrained(final_path)
    print(f"Training complete! Model saved to {final_path}")
    
    # Save training config
    import json
    with open(os.path.join(config["output_dir"], "training_config.json"), "w") as f:
        json.dump(config, f, indent=2)

if __name__ == "__main__":
    import numpy as np  # Import here to avoid issues
    train_lora()