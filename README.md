# VGDL Code Generation with LLMs

This repository contains the codebase for a study on **automatic generation of VGDL (Video Game Description Language) code from natural language descriptions** using Large Language Models.

The project explores three complementary strategies: **zero-shot prompting**, **supervised fine-tuning (SFT)** via QLoRA, and **reinforcement learning (RL)**. It includes a fully automated pipeline for dataset construction, model training, structural evaluation, and training curve visualization.

---

## 🧠 Key Features

- 📝 **Automated dataset construction**: 201 (description, VGDL) pairs generated via Claude Sonnet from raw VGDL files
- 🔍 **Zero-shot inference**: baseline experiments across 6 open-source LLMs
- 🎯 **Supervised fine-tuning**: QLoRA-based SFT with Unsloth on Qwen3.5-4B and DeepSeek-LLM 7B
- 🤖 **Reinforcement learning**: RL experiments on Qwen3.5 and DeepSeek (in progress)
- ✅ **Structural evaluation**: VGDL executability check via py-vgdl parser + Jaccard-based similarity score
- 📊 **Training visualization**: interactive notebook for loss, LR schedule, and gradient norm curves

---

## 🛠️ Technologies Used

- **Unsloth** for memory-efficient fine-tuning (QLoRA, gradient checkpointing)
- **TRL** (`SFTTrainer`) for supervised fine-tuning
- **PEFT** (LoRA / QLoRA with NF4 quantization)
- **Hugging Face** `transformers`, `datasets`, `accelerate`
- **py-vgdl** (local fork) for VGDL parsing and executability validation
- **Anthropic API** (`claude-sonnet`) for automated description generation
- **PyTorch 2.10 + CUDA 13.0**, Python, Matplotlib, Jupyter

---

## 🏗️ Pipeline Overview

```
dataset/vgdl_files/          ← 201 raw VGDL game files
        ↓  create_db.py      (Claude Sonnet generates descriptions)
dataset/descriptions/        ← 201 natural language descriptions
        ↓  dataset_hf.py     (train/test split 90/10, HF format)
dataset_hf/                  ← HuggingFace Dataset (arrow format)
        ↓
  ┌─────────────────────────────────────┐
  │          Model Experiments          │
  │                                     │
  │  zero-shot/     → baseline inference│
  │  supervised-learning/ → SFT QLoRA  │
  │  reinforcement-learning/ → RL       │
  └─────────────────────────────────────┘
        ↓
  evaluation/
    check_vgdl_executability.py  ← syntax + semantic validation
    eval_similarity.py           ← Jaccard structural similarity
    plot_training_curves.ipynb   ← training metrics visualization
```

---

## 📂 Project Structure

```
vgdl-reinforcement-learning/
│
├── dataset/
│   ├── vgdl_files/          # 201 raw VGDL game specifications
│   ├── descriptions/        # 201 natural language descriptions (auto-generated)
│   ├── create_db.py         # Generates descriptions via Claude Sonnet API
│   └── dataset_hf.py        # Converts pairs to HuggingFace Dataset format
│
├── dataset_hf/              # HF Dataset (train/test split 90/10, seed=42)
│
├── models/
│   ├── qwen3.5/
│   │   ├── zero-shot/       # Qwen3.5 zero-shot inference + results
│   │   ├── supervised-learning/
│   │   │   ├── finetune.py  # SFT training script (LoRA r=16 α=32)
│   │   │   ├── inference.py # Inference with fine-tuned adapter
│   │   │   └── 16-32-0.05/  # Saved adapter + training metrics
│   │   └── reinforcement-learning/
│   │
│   ├── deepseek/
│   │   ├── zero-shot/       # DeepSeek zero-shot inference + results
│   │   ├── supervised-learning/
│   │   │   ├── finetune.py  # SFT training script (QLoRA 4-bit NF4)
│   │   │   ├── inference.py # Inference with fine-tuned adapter
│   │   │   └── 16-32-0.05-4bit/  # Saved adapter + training metrics + plots
│   │   └── reinforcement-learning/
│   │
│   ├── qwen-2.5/            # Qwen 2.5 zero-shot experiments
│   ├── phi-3.5/             # Phi-3.5 zero-shot experiments
│   ├── mistral/             # Mistral zero-shot experiments
│   ├── minerva7b/           # Minerva 7B zero-shot experiments
│   ├── gpt_neo-2.7/         # GPT-Neo 2.7B zero-shot experiments
│   └── gold-descriptions/   # Reference descriptions for evaluation
│
├── evaluation/
│   ├── check_vgdl_executability.py  # Validates VGDL via py-vgdl parser
│   ├── eval_similarity.py           # Jaccard similarity (sprites/interactions/termination)
│   └── plot_training_curves.ipynb   # Training metrics visualization
│
├── py-vgdl/                 # Local fork of the py-vgdl interpreter
└── requirements.txt
```

---

## 🧪 Setup Instructions

### Main environment (fine-tuning and inference)

```bash
conda create -n rlia python=3.10
conda activate rlia
pip install -r requirements.txt
```

### py-vgdl environment (executability check only)

```bash
cd py-vgdl
pip install -e .
```

> The executability check must be run under the `py-vgdl` environment, while all other scripts use the main `rlia` environment.

### API key (dataset construction)

Create a `.env` file inside `dataset/`:

```
ANTHROPIC_API_KEY=your_key_here
```

---

## 🚀 How to Run

### 1. Build the dataset

```bash
# Step 1 – generate descriptions from VGDL files (requires Anthropic API key)
cd dataset
python create_db.py

# Step 2 – convert to HuggingFace Dataset format
cd ..
python dataset/dataset_hf.py
```

### 2. Zero-shot inference

Each model has its own inference script under `models/<model>/zero-shot/inference/`:

```bash
# Example: DeepSeek zero-shot
python models/deepseek/zero-shot/inference/vgdl_gen_deepseek_zeroshot.py
```

### 3. Supervised fine-tuning

```bash
# Qwen3.5-4B (QLoRA, r=16, α=32, dropout=0.05)
python models/qwen3.5/supervised-learning/finetune.py

# DeepSeek-LLM 7B (QLoRA 4-bit NF4 — required due to VRAM constraints)
python models/deepseek/supervised-learning/finetune.py
```

Training metrics are saved to `training_metrics.json` inside the output directory.

### 4. Inference with fine-tuned model

```bash
python models/deepseek/supervised-learning/inference.py
python models/qwen3.5/supervised-learning/inference.py
```

---

## 📊 Evaluation

### Check VGDL executability

```bash
# Activate py-vgdl environment first
conda activate py-vgdl-env
python evaluation/check_vgdl_executability.py <path_to_vgdl.txt>
```

The script uses the py-vgdl parser to validate syntax, sprite references, and termination conditions.

### Compute structural similarity (Jaccard)

```bash
python evaluation/eval_similarity.py
```

The similarity score is a weighted Jaccard over three components:

| Component      | Weight |
|----------------|--------|
| Sprites        | 25%    |
| Interactions   | 50%    |
| Termination    | 25%    |

### Visualize training curves

Open the notebook and set `METRICS_PATH` to the desired `training_metrics.json`:

```bash
jupyter notebook evaluation/plot_training_curves.ipynb
```

---

## 👥 Contact

- [Mattia Maucioni](https://github.com/mcnmtt) | 📧 mattiamaucioni [at] icloud.com
