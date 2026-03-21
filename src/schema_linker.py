"""
src/schema_linker.py
====================
Schema Linking for the Schema-Aware Text-to-SQL Transformer.

WHAT IS SCHEMA LINKING?
-----------------------
Schema linking is the process of identifying which words in the natural
language question correspond to tables and columns in the database schema.

Example:
    Question : "What is the average age of students older than 20?"
    Schema   : students : student_id , name , age , gpa

    Schema Links Found:
        "age"      → students.age      (column match)
        "students" → table:students    (table match)
        "20"       → (no match)

    Highlighted Input:
        "question: What is the average [age] of [students] older than 20?
         | context: students : student_id , name , [age] , gpa"

WHY DOES THIS HELP?
-------------------
Without schema linking, the model must:
  1. Read the question
  2. Read the schema
  3. Figure out on its own which question words map to which schema tokens
  4. Generate SQL using the correct schema tokens

This is hard. The model often gets step 3 wrong, generating:
  - Wrong casing: "Age" instead of "age"
  - Wrong column names: "student_name" instead of "name"
  - Hallucinated columns that don't exist in the schema

WITH schema linking, we do step 3 FOR the model before it even starts.
We explicitly mark the matched tokens with brackets, so the model sees:
  "average [age] of [students]" → it knows EXACTLY which schema tokens
  to use when generating SQL.

This is Layer 1 of the 3-layer schema linking approach from the
architecture design (Phase 1). It requires no change to model architecture
— it's purely a pre-processing improvement.

Author : [Your Name]
Project: Schema-Aware Text-to-SQL Transformer (Spider)
"""

from __future__ import annotations

import re
import string
from typing import Dict, List, Set, Tuple

# ===========================================================================
# PART 1 — TEXT NORMALIZATION UTILITIES
# ===========================================================================

# SQL keywords to ignore during matching — we don't want "name" in
# "column_name" to match the word "name" in every question
SQL_KEYWORDS: Set[str] = {
    "select", "from", "where", "and", "or", "not", "in", "is", "null",
    "like", "between", "exists", "join", "on", "as", "by", "order",
    "group", "having", "limit", "offset", "union", "all", "distinct",
    "count", "sum", "avg", "max", "min", "the", "a", "an", "of", "to",
    "for", "with", "what", "which", "who", "how", "many", "much",
    "show", "find", "list", "get", "give", "tell", "are", "is", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does",
    "did", "will", "would", "could", "should", "may", "might", "than",
    "that", "this", "these", "those", "each", "every", "any", "some",
    "no", "more", "most", "least", "less", "above", "below", "between",
}

# Minimum token length to consider for matching
# Short tokens like "id", "no" cause too many false positives
MIN_MATCH_LENGTH = 3


def normalize_token(token: str) -> str:
    """
    Normalize a single token for comparison.

    Converts to lowercase and removes punctuation so that:
        "Students" == "students"
        "age,"     == "age"
        "name?"    == "name"

    Args:
        token: Raw token string.

    Returns:
        Normalized token string.
    """
    token = token.lower().strip()
    token = token.translate(str.maketrans("", "", string.punctuation))
    return token


def tokenize_question(question: str) -> List[str]:
    """
    Split a question into individual word tokens.

    We preserve the original tokens (with their position) so we can
    reconstruct the highlighted question later.

    Args:
        question: Raw natural language question string.

    Returns:
        List of word tokens.

    Example:
        >>> tokenize_question("What is the average age of students?")
        ['What', 'is', 'the', 'average', 'age', 'of', 'students?']
    """
    return question.split()


# ===========================================================================
# PART 2 — SCHEMA MATCH FINDING
# ===========================================================================

def find_schema_matches(
    question: str,
    table_names: List[str],
    column_names: List[Tuple[int, str]],
) -> Dict[str, str]:
    """
    Find all question tokens that match table or column names in the schema.

    This is the core of schema linking. We do exact string matching after
    normalization. For a research extension, this could be replaced with
    fuzzy matching or embedding-based similarity.

    Matching Rules:
    ---------------
    1. A question token matches a COLUMN if:
       - normalize(token) == normalize(column_name)
       - token length >= MIN_MATCH_LENGTH
       - token is not a SQL keyword

    2. A question token matches a TABLE if:
       - normalize(token) == normalize(table_name)
       - token length >= MIN_MATCH_LENGTH
       - token is not a SQL keyword

    3. Multi-word table/column names (e.g., "song_name") are split on
       underscores and matched against question bigrams/unigrams.

    Args:
        question     : Raw natural language question.
        table_names  : List of table name strings from tables.json.
        column_names : List of (table_idx, col_name) tuples from tables.json.

    Returns:
        Dictionary mapping matched question token → schema reference.
        e.g., {"age": "students.age", "students": "table:students"}

    Example:
        >>> find_schema_matches(
        ...     "What is the average age of students?",
        ...     ["students"],
        ...     [(-1, "*"), (0, "student_id"), (0, "name"), (0, "age")]
        ... )
        {'age': 'students.age', 'students': 'table:students'}
    """
    matches: Dict[str, str] = {}

    # Build normalized lookup sets for fast matching
    # Table lookup: normalized_name → original_name
    table_lookup: Dict[str, str] = {}
    for t_name in table_names:
        norm = normalize_token(t_name)
        if len(norm) >= MIN_MATCH_LENGTH:
            table_lookup[norm] = t_name

    # Column lookup: normalized_name → (table_idx, original_name)
    col_lookup: Dict[str, Tuple[int, str]] = {}
    for t_idx, c_name in column_names:
        if t_idx == -1:      # skip wildcard *
            continue
        norm = normalize_token(c_name)
        if len(norm) >= MIN_MATCH_LENGTH and norm not in SQL_KEYWORDS:
            col_lookup[norm] = (t_idx, c_name)

        # Also index individual words from underscore-separated names
        # "song_name" → also index "song" and "name" separately
        parts = c_name.lower().split("_")
        for part in parts:
            part_norm = normalize_token(part)
            if (len(part_norm) >= MIN_MATCH_LENGTH
                    and part_norm not in SQL_KEYWORDS
                    and part_norm not in col_lookup):
                col_lookup[part_norm] = (t_idx, c_name)

    # Scan question tokens for matches
    tokens = tokenize_question(question)
    for token in tokens:
        norm = normalize_token(token)

        if len(norm) < MIN_MATCH_LENGTH or norm in SQL_KEYWORDS:
            continue

        # Check column match first (more specific)
        if norm in col_lookup:
            t_idx, c_name = col_lookup[norm]
            # Get the parent table name for the reference string
            if 0 <= t_idx < len(table_names):
                table_ref = table_names[t_idx]
                matches[token] = f"{table_ref}.{c_name}"
            else:
                matches[token] = c_name

        # Check table match
        elif norm in table_lookup:
            orig_table = table_lookup[norm]
            matches[token] = f"table:{orig_table}"

    return matches


# ===========================================================================
# PART 3 — INPUT HIGHLIGHTING
# ===========================================================================

def highlight_question(question: str, matches: Dict[str, str]) -> str:
    """
    Wrap matched tokens in the question with square brackets.

    WHY SQUARE BRACKETS?
    --------------------
    T5's SentencePiece tokenizer treats '[' and ']' as separate sub-word
    tokens. When we write "[age]", the model sees three tokens: '[', 'age',
    ']'. After training on thousands of examples where matched schema tokens
    always appear between '[' and ']', the model learns:

        "[token]" → "this exact token should appear in the output SQL"

    This is a soft copy signal — we're telling the model "pay extra
    attention to this token" without modifying the model architecture.

    Args:
        question : Raw natural language question.
        matches  : Output of find_schema_matches().

    Returns:
        Question string with matched tokens wrapped in brackets.

    Example:
        >>> highlight_question(
        ...     "What is the average age of students?",
        ...     {"age": "students.age", "students": "table:students"}
        ... )
        'What is the average [age] of [students]?'
    """
    if not matches:
        return question

    tokens = tokenize_question(question)
    highlighted_tokens = []

    for token in tokens:
        # Check if this token (stripped of punctuation) is a match
        norm = normalize_token(token)
        # Find if any match key normalizes to the same thing
        is_matched = any(
            normalize_token(match_key) == norm
            for match_key in matches.keys()
        )

        if is_matched:
            # Preserve any trailing punctuation outside the brackets
            # e.g., "students?" → "[students]?"
            stripped = token.rstrip(string.punctuation)
            suffix   = token[len(stripped):]
            highlighted_tokens.append(f"[{stripped}]{suffix}")
        else:
            highlighted_tokens.append(token)

    return " ".join(highlighted_tokens)


def highlight_schema(
    serialized_schema: str,
    matches: Dict[str, str],
) -> str:
    """
    Wrap matched column names in the serialized schema with brackets.

    This reinforces the linking signal on BOTH sides:
    - Question side: "[age] of [students]"
    - Schema side:   "students : student_id , name , [age] , gpa"

    When the model sees the same bracketed token in both the question
    AND the schema, it learns a very strong association between the
    question context and the schema token. This directly addresses the
    hallucination problem — the model copies from the schema rather
    than generating from memory.

    Args:
        serialized_schema : Output of serialize_schema() from dataset.py.
        matches           : Output of find_schema_matches().

    Returns:
        Schema string with matched column/table names wrapped in brackets.
    """
    if not matches:
        return serialized_schema

    # Collect all column and table names that were matched
    matched_schema_tokens: Set[str] = set()
    for schema_ref in matches.values():
        if schema_ref.startswith("table:"):
            matched_schema_tokens.add(schema_ref[6:].lower())  # table name
        else:
            # "students.age" → add "age"
            parts = schema_ref.split(".")
            if len(parts) == 2:
                matched_schema_tokens.add(parts[1].lower())    # column name
                matched_schema_tokens.add(parts[0].lower())    # table name too

    # Tokenize the serialized schema and highlight matched tokens
    # The schema format is: "table1 : col1 , col2 | table2 : col1"
    # We split on spaces to get individual tokens
    schema_tokens = serialized_schema.split(" ")
    highlighted = []

    for token in schema_tokens:
        # Skip structural tokens
        if token in (":", "|", ","):
            highlighted.append(token)
            continue

        norm = normalize_token(token)
        if norm in matched_schema_tokens and len(norm) >= MIN_MATCH_LENGTH:
            highlighted.append(f"[{token}]")
        else:
            highlighted.append(token)

    return " ".join(highlighted)


# ===========================================================================
# PART 4 — MAIN PUBLIC API
# ===========================================================================

def build_linked_input(
    question: str,
    serialized_schema: str,
    table_names: List[str],
    column_names: List[Tuple[int, str]],
) -> str:
    """
    Build the complete schema-linked model input string.

    This is the DROP-IN REPLACEMENT for build_model_input() in dataset.py.
    It produces a richer input string where matched tokens are highlighted
    on both the question side and the schema side.

    Pipeline:
        1. Find schema matches in the question
        2. Highlight matched tokens in the question
        3. Highlight matched tokens in the schema
        4. Fuse into final input string

    Args:
        question          : Raw natural language question.
        serialized_schema : Output of serialize_schema() from dataset.py.
        table_names       : List of table names from tables.json.
        column_names      : List of (table_idx, col_name) from tables.json.

    Returns:
        Complete model input string with schema linking highlights.

    Example:
        Input:
            question = "What is the average age of students?"
            schema   = "students : student_id , name , age , gpa"

        Output:
            "question: What is the average [age] of [students]?
             | context: [students] : student_id , name , [age] , gpa"
    """
    # Step 1: Find matches
    matches = find_schema_matches(question, table_names, column_names)

    # Step 2: Highlight question
    highlighted_question = highlight_question(question, matches)

    # Step 3: Highlight schema
    highlighted_schema = highlight_schema(serialized_schema, matches)

    # Step 4: Fuse
    return f"question: {highlighted_question.strip()} | context: {highlighted_schema}"


def get_match_stats(matches: Dict[str, str]) -> Dict[str, int]:
    """
    Return statistics about the matches found.
    Useful for debugging and analysis notebooks.

    Args:
        matches: Output of find_schema_matches().

    Returns:
        Dict with counts of table matches and column matches.
    """
    table_matches  = sum(1 for v in matches.values() if v.startswith("table:"))
    column_matches = sum(1 for v in matches.values() if not v.startswith("table:"))
    return {
        "total_matches" : len(matches),
        "table_matches" : table_matches,
        "column_matches": column_matches,
    }


# ===========================================================================
# PART 5 — SANITY CHECK
# ===========================================================================

if __name__ == "__main__":
    print("=" * 65)
    print("  SCHEMA LINKER — Sanity Check")
    print("=" * 65)

    # Simulate a Spider example
    test_cases = [
        {
            "question"    : "What is the average age of students older than 20?",
            "table_names" : ["students"],
            "column_names": [(-1, "*"), (0, "student_id"), (0, "name"),
                             (0, "age"), (0, "gpa")],
            "schema"      : "students : student_id , name , age , gpa",
        },
        {
            "question"    : "How many singers are from France?",
            "table_names" : ["singer", "concert", "stadium"],
            "column_names": [(-1, "*"), (0, "singer_id"), (0, "name"),
                             (0, "country"), (0, "age"),
                             (1, "concert_id"), (1, "concert_name"),
                             (2, "stadium_id"), (2, "location")],
            "schema"      : "singer : singer_id , name , country , age | concert : concert_id , concert_name | stadium : stadium_id , location",
        },
        {
            "question"    : "Find all song names by singers above the average age.",
            "table_names" : ["singer"],
            "column_names": [(-1, "*"), (0, "singer_id"), (0, "name"),
                             (0, "song_name"), (0, "age")],
            "schema"      : "singer : singer_id , name , song_name , age",
        },
    ]

    for i, tc in enumerate(test_cases):
        print(f"\n--- Test Case {i+1} ---")
        print(f"Question : {tc['question']}")

        matches = find_schema_matches(
            tc["question"],
            tc["table_names"],
            tc["column_names"],
        )
        print(f"Matches  : {matches}")
        print(f"Stats    : {get_match_stats(matches)}")

        linked_input = build_linked_input(
            tc["question"],
            tc["schema"],
            tc["table_names"],
            tc["column_names"],
        )
        print(f"Output   :")
        # Word-wrap for readability
        for j in range(0, len(linked_input), 70):
            print(f"  {linked_input[j:j+70]}")

    print("\n" + "=" * 65)
    print("  Schema Linker ready.")
    print("=" * 65)
