"""
src/dataset.py  (Updated — Schema Linking integrated)
======================================================
Changes from original:
  - SpiderDataset now accepts use_schema_linking=True parameter
  - When enabled, uses build_linked_input() from schema_linker.py
    instead of build_model_input()
  - Schema raw data (table_names, column_names) is passed through
    to the linker for each example
  - Backward compatible — use_schema_linking=False gives original behavior
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from transformers import T5Tokenizer

logger = logging.getLogger(__name__)


# ===========================================================================
# SCHEMA SERIALIZATION
# ===========================================================================

def build_schema_lookup(tables_data: List[Dict]) -> Dict[str, Dict]:
    """Pre-process tables.json into a dict keyed by db_id."""
    return {entry["db_id"]: entry for entry in tables_data}


def serialize_schema(db_id: str, schema_lookup: Dict[str, Dict]) -> str:
    """
    Convert a database schema into a flat model-readable string.
    Format: "table1 : col1 , col2 | table2 : col1"
    """
    if db_id not in schema_lookup:
        raise KeyError(f"db_id '{db_id}' not found in tables.json.")

    schema       = schema_lookup[db_id]
    table_names  = schema["table_names_original"]
    column_names = schema["column_names_original"]

    table_to_columns: Dict[int, List[str]] = {
        i: [] for i in range(len(table_names))
    }
    for table_idx, col_name in column_names:
        if table_idx == -1:
            continue
        table_to_columns[table_idx].append(col_name)

    table_segments = []
    for table_idx, table_name in enumerate(table_names):
        cols    = table_to_columns.get(table_idx, [])
        col_str = " , ".join(cols)
        table_segments.append(f"{table_name} : {col_str}")

    return " | ".join(table_segments)


def build_model_input(question: str, serialized_schema: str) -> str:
    """
    Basic input fusion without schema linking.
    Format: "question: <question> | context: <schema>"
    """
    return f"question: {question.strip()} | context: {serialized_schema}"


# ===========================================================================
# PYTORCH DATASET
# ===========================================================================

class SpiderDataset(Dataset):
    """
    PyTorch Dataset for Spider Text-to-SQL.

    Args:
        data_path            : Path to train.json or dev.json.
        tables_path          : Path to tables.json.
        tokenizer            : Pre-loaded T5Tokenizer instance.
        max_source_length    : Max tokens for input  (default 512).
        max_target_length    : Max tokens for target (default 128).
        use_schema_linking   : If True, uses schema_linker.py to highlight
                               matched tokens in input. Default: True.
    """

    def __init__(
        self,
        data_path: str | Path,
        tables_path: str | Path,
        tokenizer: T5Tokenizer,
        max_source_length: int = 512,
        max_target_length: int = 128,
        use_schema_linking: bool = True,
    ) -> None:
        super().__init__()

        self.tokenizer          = tokenizer
        self.max_source_length  = max_source_length
        self.max_target_length  = max_target_length
        self.use_schema_linking = use_schema_linking

        if use_schema_linking:
            try:
                from src.schema_linker import build_linked_input
            except ModuleNotFoundError:
                from schema_linker import build_linked_input
            self._build_linked_input = build_linked_input
            logger.info("Schema linking ENABLED.")
        else:
            logger.info("Schema linking DISABLED.")

        logger.info("Loading Spider data from %s", data_path)
        with open(data_path, "r", encoding="utf-8") as f:
            raw_data: List[Dict] = json.load(f)

        logger.info("Loading schema lookup from %s", tables_path)
        with open(tables_path, "r", encoding="utf-8") as f:
            tables_data: List[Dict] = json.load(f)

        self.schema_lookup = build_schema_lookup(tables_data)

        self.examples: List[Dict[str, str]] = []
        skipped = 0

        for item in raw_data:
            try:
                db_id      = item["db_id"]
                question   = item["question"]
                target_sql = item["query"]

                schema_entry = self.schema_lookup[db_id]
                serialized   = serialize_schema(db_id, self.schema_lookup)

                if use_schema_linking:
                    input_text = self._build_linked_input(
                        question,
                        serialized,
                        schema_entry["table_names_original"],
                        schema_entry["column_names_original"],
                    )
                else:
                    input_text = build_model_input(question, serialized)

                self.examples.append({
                    "input_text" : input_text,
                    "target_sql" : target_sql,
                    "db_id"      : db_id,
                    "question"   : question,
                })

            except KeyError as e:
                logger.warning("Skipping example due to missing key: %s", e)
                skipped += 1

        logger.info(
            "Dataset ready: %d examples loaded, %d skipped.",
            len(self.examples), skipped
        )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]

        input_encoding = self.tokenizer(
            example["input_text"],
            max_length=self.max_source_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        target_encoding = self.tokenizer(
            example["target_sql"],
            max_length=self.max_target_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )

        labels = target_encoding["input_ids"].squeeze(0).clone()
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids"     : input_encoding["input_ids"].squeeze(0),
            "attention_mask": input_encoding["attention_mask"].squeeze(0),
            "labels"        : labels,
        }

    def get_raw(self, idx: int) -> Dict[str, str]:
        return self.examples[idx]

    def decode_labels(self, labels: torch.Tensor) -> str:
        valid_ids = labels[labels != -100]
        return self.tokenizer.decode(valid_ids, skip_special_tokens=True)


# ===========================================================================
# SANITY CHECK
# ===========================================================================

if __name__ == "__main__":
    import sys
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    DATA_DIR    = Path("data/raw")
    TRAIN_PATH  = DATA_DIR / "train_spider.json"
    TABLES_PATH = DATA_DIR / "tables.json"
    MODEL_NAME  = "t5-small"

    for p in [TRAIN_PATH, TABLES_PATH]:
        if not p.exists():
            logger.error("File not found: %s", p)
            sys.exit(1)

    print("\n" + "=" * 65)
    print("  DATASET SANITY CHECK — Schema Linking ON vs OFF")
    print("=" * 65)

    tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)

    print("\n[1/2] Dataset WITHOUT schema linking:")
    ds_basic = SpiderDataset(
        TRAIN_PATH, TABLES_PATH, tokenizer,
        use_schema_linking=False
    )
    print(f"  Input: {ds_basic.get_raw(0)['input_text'][:120]}...")

    print("\n[2/2] Dataset WITH schema linking:")
    ds_linked = SpiderDataset(
        TRAIN_PATH, TABLES_PATH, tokenizer,
        use_schema_linking=True
    )
    print(f"  Input: {ds_linked.get_raw(0)['input_text'][:120]}...")

    print("\n" + "=" * 65)
    print("  Dataset with schema linking ready.")
    print("=" * 65)
