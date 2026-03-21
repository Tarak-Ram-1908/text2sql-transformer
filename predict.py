"""
predict.py
==========
Interactive inference script for the Schema-Aware Text-to-SQL Transformer.

Run with:
    python predict.py

The user provides a plain-English question and a schema in the format:
    table1: col1, col2 | table2: col1, col2

This script converts that input into the exact format the model was
trained on, then decodes the model's output back into SQL.

Fixes applied vs original:
  - model.model.from_pretrained() → T5ForConditionalGeneration.from_pretrained()
    (the original line silently created and discarded a new object; weights
    were never actually loaded into the model used for inference)
  - Schema input is now parsed and re-serialized into the training format
    so the model sees the same token distribution it learned from
  - Tokenizer is loaded from the saved checkpoint folder (local), not HF Hub
"""

import torch
from transformers import T5Tokenizer, T5ForConditionalGeneration


# ---------------------------------------------------------------------------
# Schema parsing helper
# ---------------------------------------------------------------------------

def parse_user_schema(raw_schema: str) -> str:
    """
    Convert the user-friendly schema string into the exact format the model
    was trained on.

    User input format  (natural, easy to type):
        "table: students | columns: student_id, name, age"
        or multi-table:
        "table: students | columns: id, name | table: courses | columns: id, title"

    Model training format  (from src/dataset.py → build_model_input):
        "students : student_id , name , age"
        or multi-table:
        "students : id , name | courses : id , title"

    WHY do this conversion?
    -----------------------
    The model was fine-tuned on 7,000+ examples where the schema always
    looked like "tablename : col , col | tablename : col".  At inference
    time, if we feed "table: students | columns: name, age" instead, the
    model sees an input distribution it has NEVER seen during training.
    Its attention patterns, which learned to treat ' : ' as "table→column
    separator" and ' | ' as "table boundary", get confused — producing
    output like "students | columns" instead of SQL.

    This function bridges the gap between a user-friendly CLI format and
    the model's learned input distribution.

    Args:
        raw_schema: User-typed schema string.

    Returns:
        Schema string in the model's training format.
    """
    raw_schema = raw_schema.strip()

    # Strategy: split on 'table:' boundaries, then extract columns
    # Handle both 'table: X | columns: Y' and direct 'X : col, col | Y : col' formats.

    # If the user already typed in training format (contains ' : '), pass through.
    # Heuristic: if 'table:' doesn't appear, assume it's already correct format.
    if "table:" not in raw_schema.lower():
        return raw_schema

    # Split by 'table:' to get individual table blocks
    # e.g. "table: students | columns: id, name | table: courses | columns: id"
    # → we need to find paired (table_name, columns) groups

    segments = []
    # Normalize spacing around pipe
    parts = [p.strip() for p in raw_schema.split("|")]

    i = 0
    while i < len(parts):
        part = parts[i].strip()

        if part.lower().startswith("table:"):
            table_name = part[len("table:"):].strip()

            # Look ahead for 'columns:' in the next part
            if i + 1 < len(parts) and parts[i + 1].strip().lower().startswith("columns:"):
                cols_raw = parts[i + 1][len("columns:"):].strip()
                # Convert "col1, col2, col3" → "col1 , col2 , col3"
                cols = " , ".join(c.strip() for c in cols_raw.split(","))
                segments.append(f"{table_name} : {cols}")
                i += 2          # consumed both 'table:' and 'columns:' parts
            else:
                # No columns listed — just add the table name
                segments.append(f"{table_name} :")
                i += 1
        else:
            # Unknown format — pass through as-is
            segments.append(part)
            i += 1

    return " | ".join(segments)


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def predict():
    # ------------------------------------------------------------------
    # 1. Device setup
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------------------------------------------------------------
    # 2. Load tokenizer and model
    # ------------------------------------------------------------------
    model_path = "models/t5-text2sql-v1"
    print(f"Loading model from {model_path}...")

    # BUG FIX: Load tokenizer from the saved checkpoint folder.
    # train.py now saves the tokenizer there via tokenizer.save_pretrained().
    # Previously this fell back to HuggingFace Hub (unauthenticated warning).
    tokenizer = T5Tokenizer.from_pretrained(model_path)

    # BUG FIX: The original code called model.model.from_pretrained() as an
    # *instance method*, which:
    #   (a) is not how from_pretrained works — it's a CLASS method
    #   (b) returned a new object that was immediately discarded
    #   (c) left the model with random untrained weights
    #
    # The correct pattern is to load directly via T5ForConditionalGeneration,
    # which is the underlying HuggingFace model that TextToSQLModel wraps.
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()

    print(f"Model loaded on {device}.")
    print("\n--- Ready! Type 'quit' to exit. ---")
    print("Schema format: 'table: students | columns: student_id, name, age'")
    print("               (multi-table: add '| table: courses | columns: id, title')\n")

    # ------------------------------------------------------------------
    # 3. Interactive inference loop
    # ------------------------------------------------------------------
    while True:
        question = input("Enter Question: ").strip()
        if question.lower() == "quit":
            break
        if not question:
            print("  (question cannot be empty)\n")
            continue

        raw_schema = input("Enter Schema: ").strip()
        if not raw_schema:
            print("  (schema cannot be empty)\n")
            continue

        # BUG FIX: Parse and re-serialize the schema into the training format.
        # Original code passed raw user input directly, causing distribution
        # mismatch between training and inference.
        serialized_schema = parse_user_schema(raw_schema)
        input_text = f"question: {question} | context: {serialized_schema}"

        print(f"\n  [DEBUG] Model input: {input_text}")  # Remove after confirming it works

        # Tokenize
        inputs = tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        ).to(device)

        # Generate SQL
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_length=128,
                num_beams=4,
                early_stopping=True,
                # Note: repetition_penalty was masking the real bug.
                # With correct weights + correct input format, the model
                # won't echo the schema. Removing it gives cleaner SQL.
            )

        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"\n  Predicted SQL: {prediction}\n")


if __name__ == "__main__":
    predict()
