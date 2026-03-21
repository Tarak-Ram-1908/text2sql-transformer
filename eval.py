"""
eval.py
=======
Evaluation script for the Schema-Aware Text-to-SQL Transformer.
Computes Exact Match (EM) accuracy on the Spider development set.

Exact Match Accuracy is the PRIMARY metric used in all Text-to-SQL
research papers. It measures what percentage of generated SQL queries
match the gold SQL exactly (after normalization).

Run with:
    python eval.py

Results are saved to: eval_results.json

Author : [Your Name]
Project: Schema-Aware Text-to-SQL Transformer (Spider)
"""

import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import T5ForConditionalGeneration, T5Tokenizer

from src.dataset import build_model_input, build_schema_lookup, serialize_schema


# ===========================================================================
# PART 1 — SQL NORMALIZATION
# ===========================================================================

def normalize_sql(sql: str) -> str:
    """
    Normalize a SQL string for fair exact-match comparison.

    WHY normalize?
    --------------
    "SELECT name FROM singer" and "select name from singer" are logically
    identical but won't match as raw strings. Normalization removes these
    superficial differences so we measure semantic correctness, not
    formatting style.

    Normalization steps:
      1. Lowercase everything
      2. Collapse multiple whitespace into single space
      3. Strip leading/trailing whitespace
      4. Remove spaces around parentheses  e.g. "count( * )" -> "count(*)"
      5. Normalize spacing around commas   e.g. "a ,b" -> "a, b"
    """
    sql = sql.lower().strip()
    sql = re.sub(r'\s+', ' ', sql)
    sql = re.sub(r'\(\s+', '(', sql)
    sql = re.sub(r'\s+\)', ')', sql)
    sql = re.sub(r'\s*,\s*', ', ', sql)
    sql = re.sub(r'\s*=\s*', ' = ', sql)
    sql = re.sub(r'\s*>\s*', ' > ', sql)
    sql = re.sub(r'\s*<\s*', ' < ', sql)
    sql = sql.replace('"', "'")   # normalize double quotes to single
    return sql.strip()


def exact_match(predicted: str, gold: str) -> bool:
    """Returns True if predicted SQL matches gold SQL after normalization."""
    return normalize_sql(predicted) == normalize_sql(gold)


# ===========================================================================
# PART 2 — EVALUATION LOOP
# ===========================================================================

def evaluate(
    model_path: str       = "models/t5-text2sql-v1",
    dev_path: str         = "data/raw/dev.json",
    tables_path: str      = "data/raw/tables.json",
    output_path: str      = "eval_results.json",
    max_examples: int     = None,
    batch_size: int       = 8,
    max_source_length: int = 512,
    max_target_length: int = 128,
):
    """
    Run evaluation on the Spider dev set and report Exact Match accuracy.

    Args:
        model_path       : Path to saved model + tokenizer directory.
        dev_path         : Path to Spider dev.json.
        tables_path      : Path to Spider tables.json.
        output_path      : Where to save per-example results as JSON.
        max_examples     : If set, only evaluate first N examples (quick test).
        batch_size       : Number of examples processed at once.
        max_source_length: Max input token length.
        max_target_length: Max output token length.
    """

    # ------------------------------------------------------------------
    # 1. Device + Model
    # ------------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice       : {device}")
    print(f"Model path   : {model_path}")

    print("Loading tokenizer and model...")
    tokenizer = T5Tokenizer.from_pretrained(model_path)
    model     = T5ForConditionalGeneration.from_pretrained(model_path)
    model.to(device)
    model.eval()
    print("Model loaded.\n")

    # ------------------------------------------------------------------
    # 2. Load data
    # ------------------------------------------------------------------
    print(f"Loading dev set from {dev_path}...")
    with open(dev_path, "r", encoding="utf-8") as f:
        dev_data = json.load(f)

    with open(tables_path, "r", encoding="utf-8") as f:
        tables_data = json.load(f)

    schema_lookup = build_schema_lookup(tables_data)

    if max_examples:
        dev_data = dev_data[:max_examples]
        print(f"Running quick eval on first {max_examples} examples.")

    total = len(dev_data)
    print(f"Evaluating on {total} examples...\n")

    # ------------------------------------------------------------------
    # 3. Evaluation loop (batched for speed)
    # ------------------------------------------------------------------
    results    = []
    correct    = 0
    start_time = time.time()

    for batch_start in tqdm(range(0, total, batch_size), desc="Evaluating"):
        batch = dev_data[batch_start: batch_start + batch_size]

        input_texts, gold_sqls, questions, db_ids = [], [], [], []

        for item in batch:
            try:
                db_id    = item["db_id"]
                question = item["question"]
                gold_sql = item["query"]

                serialized = serialize_schema(db_id, schema_lookup)
                input_text = build_model_input(question, serialized)

                input_texts.append(input_text)
                gold_sqls.append(gold_sql)
                questions.append(question)
                db_ids.append(db_id)
            except KeyError:
                continue

        if not input_texts:
            continue

        # Tokenize batch
        encodings = tokenizer(
            input_texts,
            max_length=max_source_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(device)

        # Generate SQL
        with torch.no_grad():
            output_ids = model.generate(
                encodings.input_ids,
                attention_mask=encodings.attention_mask,
                max_length=max_target_length,
                num_beams=4,
                early_stopping=True,
            )

        # Score each example
        for i, output in enumerate(output_ids):
            predicted_sql = tokenizer.decode(output, skip_special_tokens=True)
            gold_sql      = gold_sqls[i]
            is_correct    = exact_match(predicted_sql, gold_sql)

            if is_correct:
                correct += 1

            results.append({
                "question"     : questions[i],
                "db_id"        : db_ids[i],
                "gold_sql"     : gold_sql,
                "predicted_sql": predicted_sql,
                "exact_match"  : is_correct,
            })

    # ------------------------------------------------------------------
    # 4. Final metrics
    # ------------------------------------------------------------------
    elapsed  = time.time() - start_time
    accuracy = correct / total * 100

    correct_examples   = [r for r in results if r["exact_match"]]
    incorrect_examples = [r for r in results if not r["exact_match"]]

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS — Spider Dev Set")
    print("=" * 60)
    print(f"  Total examples     : {total}")
    print(f"  Correct            : {correct}")
    print(f"  Incorrect          : {total - correct}")
    print(f"  Exact Match (EM)   : {accuracy:.2f}%")
    print(f"  Time taken         : {elapsed:.1f}s  ({elapsed/total:.2f}s/example)")
    print("=" * 60)

    print("\n--- 5 CORRECT PREDICTIONS ---")
    for r in correct_examples[:5]:
        print(f"  Q   : {r['question']}")
        print(f"  Gold: {r['gold_sql']}")
        print(f"  Pred: {r['predicted_sql']}")
        print()

    print("--- 5 INCORRECT PREDICTIONS (error analysis) ---")
    for r in incorrect_examples[:5]:
        print(f"  Q   : {r['question']}")
        print(f"  Gold: {r['gold_sql']}")
        print(f"  Pred: {r['predicted_sql']}")
        print()

    # ------------------------------------------------------------------
    # 5. Save results
    # ------------------------------------------------------------------
    summary = {
        "model_path"      : model_path,
        "total_examples"  : total,
        "correct"         : correct,
        "exact_match_pct" : round(accuracy, 2),
        "time_seconds"    : round(elapsed, 1),
        "per_example"     : results,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"Full results saved to '{output_path}'")
    print(f"\n>>> FINAL SCORE: {accuracy:.2f}% Exact Match on Spider Dev Set <<<\n")

    return accuracy


# ===========================================================================
# ENTRY POINT
# ===========================================================================

if __name__ == "__main__":
    # Running on first 50 examples first to verify setup quickly.
    # On CPU: 50 examples takes ~2 minutes, full 1034 takes ~45 minutes.
    # On GPU (Colab): full 1034 takes ~3 minutes.
    #
    # Once you confirm the output looks correct, either:
    #   (a) Change max_examples=None here and rerun locally (slow on CPU)
    #   (b) Run full eval on Colab GPU (recommended — see instructions below)

    print("=" * 60)
    print("  Schema-Aware Text-to-SQL — Spider Evaluation")
    print("=" * 60)
    print("\nRunning quick check on 50 examples first.")
    print("Change max_examples=None for full Spider dev evaluation.\n")

    score = evaluate(
        model_path        = "models/t5-text2sql-v1",
        dev_path          = "data/raw/dev.json",
        tables_path       = "data/raw/tables.json",
        output_path       = "eval_results.json",
        max_examples      = None,   # Change to None for full evaluation
        batch_size        = 8,
        max_source_length = 512,
        max_target_length = 128,
    )
