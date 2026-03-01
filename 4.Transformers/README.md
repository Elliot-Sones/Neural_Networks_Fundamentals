# Transformers — From Scratch

<p align="center">
  <img src="https://img.shields.io/badge/PyTorch-2.0+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Gradio-4.0+-ff7c00?logo=gradio&logoColor=white" alt="Gradio">
</p>

[Try the live demo](https://huggingface.co/spaces/Eli181927/Transformer_Demo)

A from-scratch implementation of the Transformer architecture from ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762). Three progressively complex projects — encoder-only, decoder-only, and full encoder-decoder — each built from first principles.

<table align="center">
  <tr>
    <td align="center" width="50%">
      <img src="../Encoder.gif" width="100%">
      <br>Emotion Analysis (Encoder)
    </td>
    <td align="center" width="50%">
      <img src="../Decoder.gif" width="100%">
      <br>Shakespeare Generator (Decoder)
    </td>
  </tr>
</table>

<p align="center">
  <img src="../Machine_Translation.gif" width="60%">
  <br>Machine Translation EN→FR (Seq2Seq)
</p>

---

## Table of Contents

- [What is a Transformer?](#what-is-a-transformer)
- [Key Concepts](#key-concepts)
- [Model Architecture](#model-architecture)
- [Encoder](#encoder-component-words--context)
- [Decoder](#decoder-component-context--words)
- [Projects](#projects)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)

---

## What is a Transformer?

Transformers are neural networks that use self-attention to take input data (like text), model relationships between elements, and generate meaningful outputs like translations or classifications.

Before Transformers, sequence models relied on recurrent (RNN) or convolutional (CNN) architectures that processed data sequentially — one token at a time. This sequential nature made them hard to scale and caused them to struggle with long-range dependencies.

<p align="center">
  <img src="assets/RNN.png" width="60%" alt="Recurrent Network for language processing" />
  <br><em>Recurrent network for language processing</em>
</p>

The 2017 paper "Attention Is All You Need" introduced the Transformer, which replaces recurrence entirely with attention mechanisms. This allows the model to relate every position to every other position in parallel, enabling much better scaling and long-term memory.

<p align="center">
  <img src="assets/transformer.png" width="55%" alt="Transformer architecture" />
  <br><em>Transformer architecture</em>
</p>

---

## Key Concepts

### Attention

The core innovation: allowing the model to focus on different parts of the input when processing each element.

<p align="center">
  <img src="assets/attention.png" width="65%" alt="Attention mechanism" />
  <br><em>Attention with Transformers</em>
</p>

**Self-Attention** — every word in a sequence attends to all other words, learning relationships regardless of distance. For example, learning that "it" refers to "the animal" in:

> "The animal didn't cross because it was too tired."

**Multi-Head Attention** — multiple attention mechanisms running in parallel, each learning different relationship types (semantic, positional, grammatical):

<img src="assets/Multi-head-attention.png" width="25%">

**Scaled Dot-Product** — the mathematical core of each attention head:

$$
\text{Attention}(Q, K, V) = \text{softmax}\left( \frac{QK^T}{\sqrt{d_k}} \right)V
$$

<img src="assets/Scaled-dot-product.png" width="20%">

### Embedding + Positional Encoding

Words are converted to vectors (embeddings) so the model can process them numerically. Since Transformers process all tokens in parallel (no recurrence), positional encodings are added to preserve word order information.

---

## Model Architecture

<p align="center">
  <img src="assets/Machine-translation.png" width="50%" alt="Full transformer architecture" />
  <br><em>Machine translation with the Transformer architecture</em>
</p>

The Transformer has two halves: **Encoder** (left) and **Decoder** (right).

### Encoder Component (Words → Context)

[Detailed README](encoder_transformer/README.md)

Processes the input sequence into rich contextual representations.

- **Self-Attention**: each word attends to all other words in the input (bidirectional)
- **Architecture**: Input → Embedding → Multi-Head Self-Attention → Feed Forward → Output
- **Learns**: grammatical relationships, semantic meaning, long-range dependencies

### Decoder Component (Context → Words)

[Detailed README](decoder_transformer/README.md)

Generates output sequences based on encoder context and previous outputs.

- **Masked Self-Attention**: can only attend to previous positions (causal masking)
- **Cross-Attention**: attends to encoder outputs for source context
- **Architecture**: Input → Embedding → Masked Self-Attention → Cross-Attention → Feed Forward → Output

### How They Work Together

1. **Encoder** processes the source sentence (e.g., English)
2. **Decoder** uses encoder context to generate the target sentence (e.g., French)
3. **Cross-attention** lets the decoder focus on relevant parts of the source
4. **Teacher forcing** during training helps the decoder learn correct patterns

---

## Projects

### 1. Encoder-Only: Emotion Classification & MLM

[Encoder README](encoder_transformer/README.md)

| | |
|---|---|
| **Architecture** | Pre-LN Transformer encoder with multi-head self-attention |
| **Tasks** | 6-class emotion classification (GoEmotions) + masked language modeling |
| **Key files** | [`encode.py`](encoder_transformer/encode.py) — encoder architecture, [`sentiment/train.py`](encoder_transformer/sentiment/train.py) — emotion classifier, [`mlm/train.py`](encoder_transformer/mlm/train.py) — MLM training |

### 2. Decoder-Only: Shakespeare Language Model

[Decoder README](decoder_transformer/README.md)

| | |
|---|---|
| **Architecture** | GPT-style character-level Transformer with causal masking |
| **Task** | Text generation trained on Tiny Shakespeare |
| **Key files** | [`training.py`](decoder_transformer/training.py) — training with EMA + cosine LR, [`sample.py`](decoder_transformer/sample.py) — text generation |

### 3. Full Seq2Seq: English → French Translation

[Machine Translation README](machine_translation/README.md)

| | |
|---|---|
| **Architecture** | Full encoder-decoder Transformer with cross-attention |
| **Task** | English → French translation (WMT14/WMT16/OPUS data) |
| **Key files** | [`mini_transformer.py`](machine_translation/mini_transformer.py) — Seq2Seq model, [`train_mini.py`](machine_translation/train_mini.py) — training loop, [`translate.py`](machine_translation/translate.py) — inference |

---

## Quick Start

### Prerequisites

```bash
pip install -r requirements.txt
# or: pip install torch tokenizers gradio datasets tqdm
```

### Train the Encoder (MLM)

```bash
python encoder_transformer/mlm/train.py
```

### Train the Decoder (Shakespeare)

```bash
python decoder_transformer/training.py
```

### Train Machine Translation

```bash
python machine_translation/setup_data.py --dataset wmt14 --out_dir data/en_fr
python machine_translation/train_mini.py --train_csv data/en_fr/train.csv --val_csv data/en_fr/test.csv
```

### Launch the Gradio Demo

```bash
python app.py
```

### Weights & Checkpoints

Model checkpoints (`.pt`) are not tracked in this repo. To run the demo locally:
- **Train from scratch** using the scripts above, or
- **Download** from the [Hugging Face Space](https://huggingface.co/spaces/Eli181927/Transformer_Demo)

---

## Project Structure

```
4.Transformers/
├── README.md                       # This file
├── app.py                          # 3-tab Gradio demo (Emotion, Shakespeare, Translation)
├── optimize_models.py              # Strip optimizer state from checkpoints
├── requirements.txt                # Transformer-specific deps
├── assets/                         # Architecture diagrams
│
├── encoder_transformer/            # Encoder-only
│   ├── encode.py                   # Encoder architecture (shared by all encoder tasks)
│   ├── README.md
│   ├── assets/
│   ├── mlm/                        # Masked Language Modeling
│   │   ├── train.py, test.py, data.py
│   │   └── models/                 # Saved MLM checkpoints
│   └── sentiment/                  # Emotion Classification
│       ├── train.py, predict.py
│       └── checkpoints/            # Saved emotion model checkpoints
│
├── decoder_transformer/            # Decoder-only (GPT-style)
│   ├── training.py                 # Train on Tiny Shakespeare
│   ├── sample.py                   # Generate text from prompt
│   ├── README.md
│   └── assets/
│
└── machine_translation/            # Full Seq2Seq
    ├── mini_transformer.py         # Encoder-decoder model
    ├── train_mini.py               # Training loop
    ├── setup_data.py               # Download & prepare EN-FR data
    ├── translate.py                # Inference / translation
    ├── train_tokenizer.py          # Train WordPiece tokenizer
    ├── tokenizer_with_words.json   # Pre-trained tokenizer
    └── README.md
```
