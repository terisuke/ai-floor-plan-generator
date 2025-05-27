#!/usr/bin/env python3
"""
Enhanced LoRA training for structural elements-based floor plan generation
"""
import torch
import pandas as pd
import numpy as np
from PIL import Image
import json
import random
from pathlib import Path
import os
from datetime import datetime

from diffusers import StableDiffusionPipeline, AutoencoderKL, UNet2DConditionModel
from transformers import CLIPTextModel, CLIPTokenizer
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from accelerate import Accelerator
from tqdm import tqdm
from peft import LoraConfig, get_peft_model, TaskType

class StructuralFloorPlanDataset(Dataset):
    """Enhanced dataset with structural element conditioning"""
    
    def __init__(self, csv_file, size=512, augment=True):
        self.data = pd.read_csv(csv_file)
        self.size = size
        self.augment = augment
        
        # Filter for structural data when possible
        self.structural_data = self.data[self.data['data_version'] == 'v2.0_structural']
        self.legacy_data = self.data[self.data['data_version'] == 'v1.0_legacy']
        
        print(f"Loaded dataset: {len(self.structural_data)} structural + {len(self.legacy_data)} legacy samples")
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Load image
        image = Image.open(row['file_name']).convert('RGB')
        
        # Augmentation
        if self.augment and random.random() < 0.5:
            rotation = random.choice([0, 90, 180, 270])
            if rotation > 0:
                image = image.rotate(rotation, fillcolor='white')
        
        image = image.resize((self.size, self.size), Image.Resampling.LANCZOS)
        image = np.array(image).astype(np.float32) / 127.5 - 1.0
        image = torch.from_numpy(image).permute(2, 0, 1)
        
        # Create enhanced text prompt
        base_text = row['text']
        conditions = row['conditions']
        
        # Combine base description with conditions
        enhanced_text = f"{base_text}, {conditions}"
        
        # Add random variations for structural elements
        if row['data_version'] == 'v2.0_structural' and self.augment:
            variations = self._get_structural_variations(row)
            if variations:
                enhanced_text = f"{enhanced_text}, {random.choice(variations)}"
        
        return {
            "image": image,
            "text": enhanced_text,
            "conditions": conditions,
            "has_stairs": row.get('has_stairs', False),
            "has_entrance": row.get('has_entrance', False),
            "has_balcony": row.get('has_balcony', False),
            "floor": row.get('floor', '1F'),
            "data_version": row['data_version']
        }
    
    def _get_structural_variations(self, row):
        """Generate text variations based on structural elements"""
        variations = []
        
        if row.get('has_stairs', False):
            variations.extend([
                "staircase centrally located",
                "vertical circulation element",
                "multi-level access point"
            ])
        
        if row.get('has_entrance', False):
            variations.extend([
                "main entrance clearly defined",
                "primary access point",
                "entry foyer area"
            ])
        
        if row.get('has_balcony', False):
            variations.extend([
                "outdoor balcony space",
                "exterior terrace area",
                "extended living space"
            ])
        
        # Layout-specific variations
        area = row.get('layout_width', 0) * row.get('layout_height', 0)
        if area > 50:
            variations.append("spacious layout design")
        elif area > 0 and area < 30:
            variations.append("compact efficient layout")
        
        return variations

def train_structural_lora():
    """Enhanced training with structural element awareness"""
    
    # Training configuration
    config = {
        "model_name": "runwayml/stable-diffusion-v1-5",
        "output_dir": "lora_model_structural",
        "resolution": 512,
        "train_batch_size": 1,
        "gradient_accumulation_steps": 4,
        "num_train_epochs": 100,
        "learning_rate": 1e-4,
        "lr_scheduler": "cosine",
        "lr_warmup_steps": 100,
        "adam_beta1": 0.9,
        "adam_beta2": 0.999,
        "adam_weight_decay": 1e-2,
        "adam_epsilon": 1e-08,
        "max_grad_norm": 1.0,
        "mixed_precision": "fp16",
        "gradient_checkpointing": True,
        "seed": 42,
        "save_steps": 200,
        "validation_steps": 100,
        "lora_rank": 32,
        "lora_alpha": 32,
        "lora_dropout": 0.1,
    }
    
    # Set random seeds
    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])
    random.seed(config["seed"])
    
    # Initialize accelerator
    accelerator = Accelerator(
        gradient_accumulation_steps=config["gradient_accumulation_steps"],
        mixed_precision=config["mixed_precision"] if torch.cuda.is_available() else None,
    )
    
    # Create output directory
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(exist_ok=True)
    
    # Save config
    with open(output_dir / "config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    # Check if enhanced dataset exists
    enhanced_train_path = Path("prepared_data/train_enhanced.csv")
    enhanced_val_path = Path("prepared_data/val_enhanced.csv")
    
    if not enhanced_train_path.exists() or not enhanced_val_path.exists():
        print("Enhanced dataset not found. Running preparation...")
        from prepare_dataset_enhanced import prepare_enhanced_dataset
        prepare_enhanced_dataset()
    
    # Load datasets
    train_dataset = StructuralFloorPlanDataset(enhanced_train_path, size=config["resolution"], augment=True)
    val_dataset = StructuralFloorPlanDataset(enhanced_val_path, size=config["resolution"], augment=False)
    
    if len(train_dataset) == 0:
        print("Warning: Training dataset is empty. Using validation data for training.")
        train_dataset = val_dataset
    
    # Create data loaders
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config["train_batch_size"],
        shuffle=True,
        num_workers=0
    )
    
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    print(f"Training with {len(train_dataset)} train samples, {len(val_dataset)} validation samples")
    
    # Load models
    print("Loading models...")
    
    # Load tokenizer and text encoder
    tokenizer = CLIPTokenizer.from_pretrained(config["model_name"], subfolder="tokenizer")
    text_encoder = CLIPTextModel.from_pretrained(config["model_name"], subfolder="text_encoder")
    
    # Load VAE and UNet
    vae = AutoencoderKL.from_pretrained(config["model_name"], subfolder="vae")
    unet = UNet2DConditionModel.from_pretrained(config["model_name"], subfolder="unet")
    
    # Freeze VAE and text encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    
    # Configure LoRA
    lora_config = LoraConfig(
        r=config["lora_rank"],
        lora_alpha=config["lora_alpha"],
        target_modules=["to_k", "to_q", "to_v", "to_out.0"],
        lora_dropout=config["lora_dropout"],
    )
    
    # Apply LoRA to UNet
    unet = get_peft_model(unet, lora_config)
    unet.print_trainable_parameters()
    
    # Enable gradient checkpointing
    if config["gradient_checkpointing"]:
        unet.enable_gradient_checkpointing()
    
    # Create optimizer
    optimizer = torch.optim.AdamW(
        unet.parameters(),
        lr=config["learning_rate"],
        betas=(config["adam_beta1"], config["adam_beta2"]),
        weight_decay=config["adam_weight_decay"],
        eps=config["adam_epsilon"],
    )
    
    # Prepare for training
    unet, optimizer, train_dataloader, val_dataloader = accelerator.prepare(
        unet, optimizer, train_dataloader, val_dataloader
    )
    
    # Move other models to device
    vae.to(accelerator.device)
    text_encoder.to(accelerator.device)
    
    # Training loop
    global_step = 0
    
    for epoch in range(config["num_train_epochs"]):
        unet.train()
        train_loss = 0
        
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{config['num_train_epochs']}")
        
        for step, batch in enumerate(progress_bar):
            with accelerator.accumulate(unet):
                # Get images and encode
                latents = vae.encode(batch["image"]).latent_dist.sample()
                latents = latents * 0.18215
                
                # Sample noise
                noise = torch.randn_like(latents)
                timesteps = torch.randint(
                    0, 1000, (latents.shape[0],), device=latents.device
                ).long()
                
                # Add noise to latents
                noisy_latents = torch.sqrt(1 - (timesteps / 1000).view(-1, 1, 1, 1) ** 2) * latents + \
                               torch.sqrt((timesteps / 1000).view(-1, 1, 1, 1) ** 2) * noise
                
                # Encode text
                text_inputs = tokenizer(
                    batch["text"],
                    padding="max_length",
                    max_length=tokenizer.model_max_length,
                    truncation=True,
                    return_tensors="pt"
                ).input_ids.to(accelerator.device)
                
                encoder_hidden_states = text_encoder(text_inputs)[0]
                
                # Predict noise
                noise_pred = unet(
                    noisy_latents,
                    timesteps,
                    encoder_hidden_states,
                    return_dict=False
                )[0]
                
                # Calculate loss
                loss = F.mse_loss(noise_pred, noise)
                
                accelerator.backward(loss)
                
                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(unet.parameters(), config["max_grad_norm"])
                
                optimizer.step()
                optimizer.zero_grad()
            
            # Update progress
            if accelerator.sync_gradients:
                train_loss += loss.detach().item()
                global_step += 1
                
                progress_bar.set_postfix({"loss": loss.detach().item()})
                
                # Save checkpoint
                if global_step % config["save_steps"] == 0:
                    save_path = output_dir / f"checkpoint-{global_step}"
                    save_path.mkdir(exist_ok=True)
                    
                    unwrapped_model = accelerator.unwrap_model(unet)
                    unwrapped_model.save_pretrained(save_path)
                    
                    print(f"\nSaved checkpoint to {save_path}")
                
                # Validation
                if global_step % config["validation_steps"] == 0 and len(val_dataset) > 0:
                    val_loss = validate(
                        unet, vae, text_encoder, tokenizer, val_dataloader, accelerator
                    )
                    print(f"\nValidation loss: {val_loss:.4f}")
                    unet.train()
        
        # Save epoch checkpoint
        epoch_path = output_dir / f"epoch-{epoch+1}"
        epoch_path.mkdir(exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(unet)
        unwrapped_model.save_pretrained(epoch_path)
    
    # Save final model
    final_path = output_dir / "final"
    final_path.mkdir(exist_ok=True)
    unwrapped_model = accelerator.unwrap_model(unet)
    unwrapped_model.save_pretrained(final_path)
    
    print(f"\nTraining complete! Final model saved to {final_path}")
    
    # Save training info
    training_info = {
        "config": config,
        "final_step": global_step,
        "timestamp": datetime.now().isoformat(),
        "structural_samples": len(train_dataset.structural_data),
        "legacy_samples": len(train_dataset.legacy_data)
    }
    
    with open(output_dir / "training_info.json", "w") as f:
        json.dump(training_info, f, indent=2)

def validate(unet, vae, text_encoder, tokenizer, val_dataloader, accelerator):
    """Run validation loop"""
    unet.eval()
    val_loss = 0
    
    with torch.no_grad():
        for batch in val_dataloader:
            # Encode images
            latents = vae.encode(batch["image"]).latent_dist.sample()
            latents = latents * 0.18215
            
            # Sample noise
            noise = torch.randn_like(latents)
            timesteps = torch.randint(
                0, 1000, (latents.shape[0],), device=latents.device
            ).long()
            
            # Add noise
            noisy_latents = torch.sqrt(1 - (timesteps / 1000).view(-1, 1, 1, 1) ** 2) * latents + \
                           torch.sqrt((timesteps / 1000).view(-1, 1, 1, 1) ** 2) * noise
            
            # Encode text
            text_inputs = tokenizer(
                batch["text"],
                padding="max_length",
                max_length=tokenizer.model_max_length,
                truncation=True,
                return_tensors="pt"
            ).input_ids.to(accelerator.device)
            
            encoder_hidden_states = text_encoder(text_inputs)[0]
            
            # Predict noise
            noise_pred = unet(
                noisy_latents,
                timesteps,
                encoder_hidden_states,
                return_dict=False
            )[0]
            
            # Calculate loss
            loss = F.mse_loss(noise_pred, noise)
            val_loss += loss.item()
    
    return val_loss / len(val_dataloader)

if __name__ == "__main__":
    train_structural_lora()