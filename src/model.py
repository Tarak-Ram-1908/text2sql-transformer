"""
src/model.py
============
The "Brain" of the Text-to-SQL Transformer.
"""

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer
from pathlib import Path

class TextToSQLModel(nn.Module):
    def __init__(self, model_name: str = "t5-small"):
        super().__init__()
        # Load the pre-trained T5 model
        self.model = T5ForConditionalGeneration.from_pretrained(model_name)
        
    def forward(self, input_ids, attention_mask, labels=None):
        """
        Standard forward pass for training. 
        If labels are provided, it returns the Loss.
        """
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        return outputs.loss, outputs.logits

    def generate_sql(self, input_ids, attention_mask, tokenizer):
        """
        Uses BEAM SEARCH to find the most logical SQL query.
        """
        generated_ids = self.model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_length=128,
            num_beams=5,      # Search 5 paths at once for better accuracy
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True
        )
        return [tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True) for g in generated_ids]

    def save(self, path: str):
        self.model.save_pretrained(path)

if __name__ == "__main__":
    # SANITY CHECK
    print("Testing Model Initialization...")
    my_model = TextToSQLModel()
    
    # Create fake data (1 example, 10 tokens long)
    fake_input = torch.randint(0, 32000, (1, 10))
    fake_mask = torch.ones((1, 10))
    
    loss, logits = my_model(fake_input, fake_mask, labels=fake_input)
    print(f"Model successfully loaded. Logit shape: {logits.shape}")
    print("✓ Model Phase Sanity Check Complete.")