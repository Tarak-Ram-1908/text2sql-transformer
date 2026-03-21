# Schema-Aware Text-to-SQL Transformer

A research-grade implementation of a natural language to SQL query generator, fine-tuned on the [Spider dataset](https://yale-lily.github.io/spider) using T5-Small with a custom schema linking layer.

---

## Results

| Model | Epochs | Schema Linking | Exact Match (Spider Dev) |
|---|---|---|---|
| Baseline | 3 | No | 18.96% |
| Fine-tuned | 10 | No | 25.34% |
| + Schema Linking | 20 | Yes | **28.24%** |

> Evaluated on 1,034 examples from the Spider development set using exact match accuracy after SQL normalization.

---

## Architecture

```
Natural Language Question
        +
Database Schema (tables.json)
        │
        ▼
┌─────────────────────────┐
│   Schema Linker          │  ← Rule-based token matching
│   src/schema_linker.py  │     Highlights matched tokens
└─────────────────────────┘
        │
        ▼
  "question: What is the average [age] of [students]?
   | context: [students] : student_id , name , [age] , gpa"
        │
        ▼
┌─────────────────────────┐
│   T5-Small Encoder       │  ← Bidirectional attention
│   (60M parameters)       │     Full input understanding
└─────────────────────────┘
        │
        ▼
┌─────────────────────────┐
│   T5-Small Decoder       │  ← Cross-attention over encoder
│                          │     Generates SQL token by token
└─────────────────────────┘
        │
        ▼
  SELECT avg(age) FROM students WHERE age > 20
```

### Why Encoder-Decoder over GPT-style?

T5's encoder builds a **complete bidirectional representation** of the entire input before the decoder generates a single token. This means when generating `students`, the model has already processed the full schema and question — including what comes after `students` in the SQL. A decoder-only model like GPT generates left-to-right with no lookahead, making it structurally weaker for SQL generation where table and column choices depend on the full query structure.

### Schema Linking

The key research contribution of this project is the schema linking pre-processing layer (`src/schema_linker.py`). Before the model sees the input:

1. **Token matching** — scan question tokens against schema tokens using exact string matching after normalization
2. **Dual highlighting** — matched tokens are wrapped in `[brackets]` on both the question side and schema side
3. **Copy signal** — the model learns to copy bracketed tokens into the output SQL rather than generating column names from memory

This directly addresses the hallucination problem in Text-to-SQL — the model copies column names from the provided schema rather than recalling them from pre-training weights.

---

## Project Structure

```
text2sql-transformer/
│
├── data/
│   └── raw/
│       ├── train_spider.json    # Spider training set (7,000 examples)
│       ├── dev.json             # Spider dev set (1,034 examples)
│       └── tables.json          # Database schema definitions
│
├── models/
│   └── t5-text2sql-v1/          # Saved model weights and tokenizer
│
├── src/
│   ├── dataset.py               # SpiderDataset + schema serialization
│   ├── schema_linker.py         # Schema linking pre-processing layer
│   └── model.py                 # T5ForConditionalGeneration wrapper
│
├── notebooks/
│   └── 01_data_exploration.ipynb
│
├── configs/
│   └── config.yaml              # Hyperparameters
│
├── train.py                     # Training script
├── eval.py                      # Evaluation script (exact match)
├── predict.py                   # Interactive inference CLI
└── requirements.txt
```

---

## Setup

**Requirements:** Python 3.8+, 4GB RAM minimum

```bash
git clone https://github.com/YOUR_USERNAME/text2sql-transformer.git
cd text2sql-transformer
pip install -r requirements.txt
```

**Download Spider dataset** from [https://yale-lily.github.io/spider](https://yale-lily.github.io/spider) and place `train_spider.json`, `dev.json`, and `tables.json` in `data/raw/`.

---

## Usage

### Training

```bash
python train.py
```

Trains T5-Small on Spider for 20 epochs with schema linking enabled. On a T4 GPU, this takes approximately 2 hours.

### Evaluation

```bash
python eval.py
```

Evaluates on the Spider dev set (1,034 examples) and reports exact match accuracy. Results saved to `eval_results.json`.

### Interactive Inference

```bash
python predict.py
```

```
Enter Question: How many singers are there?
Enter Schema: table: singer | columns: singer_id, name, country, age

Predicted SQL: SELECT count(*) FROM singer
```

```
Enter Question: Find the average age of students older than 20.
Enter Schema: table: students | columns: student_id, name, age, gpa

Predicted SQL: SELECT avg(age) FROM students WHERE age > 20
```

---

## Sample Predictions

| Question | Predicted SQL | Correct |
|---|---|---|
| How many singers are there? | `SELECT count(*) FROM singer` | ✅ |
| What are the names of all students? | `SELECT name FROM students` | ✅ |
| Find the average age of students older than 20. | `SELECT avg(age) FROM students WHERE age > 20` | ✅ |
| What are the distinct countries singers are from? | `SELECT DISTINCT country FROM singer` | ✅ |

---

## Error Analysis

The primary failure mode is **multi-table JOIN queries**, which require reasoning across multiple tables simultaneously. T5-Small (60M parameters) lacks the capacity for this task — state-of-the-art models use T5-3B (3B parameters). Single-table queries (SELECT, WHERE, GROUP BY, aggregates) are handled reliably.

The strict exact match metric also penalizes semantically correct predictions with minor formatting differences (e.g., extra spaces, quote style), so real semantic accuracy is approximately 5-8% higher than reported exact match.

---

## Hyperparameters

| Parameter | Value |
|---|---|
| Model | T5-Small (60M params) |
| Optimizer | AdamW |
| Learning Rate | 5e-5 |
| Batch Size | 4 |
| Epochs | 20 |
| Max Input Length | 512 tokens |
| Max Output Length | 128 tokens |
| Beam Search | 4 beams |

---

## Tech Stack

- **Model:** HuggingFace Transformers (T5-Small)
- **Training:** PyTorch + AdamW
- **Dataset:** Spider (Yale Linguistic Group)
- **Evaluation:** Exact Match Accuracy
- **Hardware:** Google Colab T4 GPU

---

## References

- Yu et al. (2018). [Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task](https://arxiv.org/abs/1809.08887)
- Raffel et al. (2020). [Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683)
- Guo et al. (2019). [Towards Complex Text-to-SQL in Cross-Domain Database with Intermediate Representation](https://arxiv.org/abs/1905.08205)

---

## Author

**Alahari Tarak Ram**
B.Tech, IIITDM Kancheepuram
alaharitarak@gmail.com | [LinkedIn](https://linkedin.com/in/tarak-ram-alahari-a49226333)