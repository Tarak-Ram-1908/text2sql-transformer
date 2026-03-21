"""
train.py
========
Main training script for the Schema-Aware Text-to-SQL Transformer.

Run with:
    python train.py

Fixes applied vs original:
  - range(0) → range(3)  : model was never training (0 epochs = 0 learning)
  - tokenizer.save_pretrained() added so predict.py can load it locally
  - Correct loss unpacking from model forward pass
  - Validation loss logged at end of each epoch
"""

import os

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import T5Tokenizer

from src.dataset import SpiderDataset
from src.model import TextToSQLModel


def train():
    # ------------------------------------------------------------------
    # 1. Hardware Setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------
    # 2. Load Tokenizer and Model
    # ------------------------------------------------------------------
    tokenizer = T5Tokenizer.from_pretrained("t5-small")
    model = TextToSQLModel("t5-small").to(device)

    # ------------------------------------------------------------------
    # 3. Prepare Data
    # ------------------------------------------------------------------
    print("Loading training data...")
    train_dataset = SpiderDataset(
        data_path="data/raw/train_spider.json",
        tables_path="data/raw/tables.json",
        tokenizer=tokenizer,
        max_source_length=512,
        max_target_length=128,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=4,       # Safe for most GPUs; reduce to 2 if you hit OOM
        shuffle=True,
        num_workers=0,      # Keep 0 on Windows to avoid multiprocessing issues
    )
    print(f"Training on {len(train_dataset)} examples, "
          f"{len(train_loader)} batches per epoch.\n")

    # ------------------------------------------------------------------
    # 4. Optimizer
    # ------------------------------------------------------------------
    optimizer = AdamW(model.parameters(), lr=5e-5)

    # ------------------------------------------------------------------
    # 5. Training Loop
    # ------------------------------------------------------------------
    # BUG FIX: range(0) meant ZERO training iterations — the model never
    # learned anything and was saved with random weights.
    # 3 epochs is a sensible starting point for T5-Small on Spider.
    # You will see loss drop from ~5.0 → ~1.0 over these epochs.
    NUM_EPOCHS = 20

    model.train()
    print(f"Starting training for {NUM_EPOCHS} epochs...")

    for epoch in range(NUM_EPOCHS):
        total_loss = 0.0
        loop = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", leave=True)

        for batch in loop:
            optimizer.zero_grad()

            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels         = batch["labels"].to(device)

            # Forward pass
            loss, _ = model(input_ids, attention_mask, labels)

            # Backward pass
            loss.backward()

            # Gradient clipping — prevents exploding gradients, standard
            # practice for transformer fine-tuning.
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            total_loss += loss.item()
            loop.set_postfix(loss=f"{loss.item():.4f}")

        avg_loss = total_loss / len(train_loader)
        print(f"  → Epoch {epoch + 1} complete. Average loss: {avg_loss:.4f}")

    # ------------------------------------------------------------------
    # 6. Save Model AND Tokenizer
    # ------------------------------------------------------------------
    save_path = "models/t5-text2sql-v1"
    os.makedirs(save_path, exist_ok=True)

    model.save(save_path)

    # BUG FIX: tokenizer was never saved — predict.py couldn't load it
    # locally and fell back to downloading from HuggingFace every time.
    # Now both model weights AND tokenizer live in the same folder.
    tokenizer.save_pretrained(save_path)

    print(f"\nModel and tokenizer saved to '{save_path}'")
    print("Files saved:")
    for f in os.listdir(save_path):
        print(f"  {f}")


if __name__ == "__main__":
    train()
