"""
GRPO (Group Relative Policy Optimization) training for VGDL generation.

Parte dal modello Qwen3.5-4B fine-tunato con SFT e lo allena con segnali di reward
basati su:
  - Eseguibilità VGDL (parser py-vgdl)
  - Struttura corretta (4 sezioni obbligatorie + BasicGame)
  - Classi sprite valide (ontologia py-vgdl)
  - Effetti di interazione validi (ontologia py-vgdl)
  - Condizioni di terminazione valide (ontologia py-vgdl)
  - Uso di EOS al posto di keyword di bordo non valide

Eseguire dalla root del progetto:
  python models/qwen3.5/reinforcement-learning/grpo_train.py
"""

import sys
import os
os.environ["TORCHDYNAMO_DISABLE"] = "1"
import json
import torch

# Aggiunge py-vgdl al path (relativo alla root del progetto)
_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "py-vgdl"))

from unsloth import FastLanguageModel
from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig
# NON usare PeftModel.from_pretrained: bypassa il pipeline di unsloth e
# causa conflitti con il forward compilato (TorchRuntimeError in rotary emb)

from reward_functions import REWARD_FUNCTIONS  # noqa: E402

# ========================
# Config
# ========================
BASE_MODEL  = "Qwen/Qwen3.5-4B"
SFT_ADAPTER = "models/qwen3.5/supervised-learning/16-32-0.05"
OUTPUT_DIR  = "models/qwen3.5/reinforcement-learning/grpo-output"
MAX_SEQ_LEN = 1024

SYSTEM_PROMPT = (
    "You are an expert in VGDL (Video Game Description Language). "
    "Given a textual description of a game, generate the corresponding "
    "valid VGDL code starting with 'BasicGame'. "
    "Output ONLY raw VGDL code. No explanation, no markdown, no comments."
)


# ========================
# Model Loading
# unsloth from_pretrained con un adapter path carica base model + applica il LoRA SFT
# (lo mantiene come adapter attivo, non lo mergia).
# Non serve FastLanguageModel.get_peft_model: il LoRA SFT è già presente.
# Basta rendere trainable i parametri LoRA per il GRPO.
# ========================
print(f"Loading SFT model from adapter {SFT_ADAPTER}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_ADAPTER,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)
print("SFT model loaded (LoRA adapter attivo).")

# Passa il modello in training mode (il from_pretrained carica in inference mode)
model.train()

# Rende trainable i parametri LoRA esistenti per il GRPO
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad_(True)

# Abilita gradient checkpointing per ridurre memoria
model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


# ========================
# Patch unsloth compiled cache: fix batch-size mismatch in apply_rotary_pos_emb
# Durante GRPO accumulated loss, cos/sin possono avere batch=0 per un chunk vuoto
# mentre q_pass mantiene batch=1 da q. Il torch.cat fallisce senza questo fix.
# ========================
def _patch_unsloth_rotary():
    import re as _re, pathlib as _pl
    cache_file = _pl.Path("unsloth_compiled_cache/unsloth_compiled_module_qwen3_5.py")
    if not cache_file.exists():
        return
    src = cache_file.read_text(encoding="utf-8")
    old = "    q_embed = torch.cat([q_embed, q_pass], dim=-1)\n    k_embed = torch.cat([k_embed, k_pass], dim=-1)"
    new = "    q_embed = torch.cat([q_embed, q_pass[:q_embed.shape[0]]], dim=-1)\n    k_embed = torch.cat([k_embed, k_pass[:k_embed.shape[0]]], dim=-1)"
    if new in src:
        print("Unsloth rotary patch already present.")
        return
    if old not in src:
        return  # unknown state, don't touch
    if old in src:
        cache_file.write_text(src.replace(old, new, 1), encoding="utf-8")
        print("Unsloth rotary patch applied.")
    elif new in src:
        print("Unsloth rotary patch already present.")

_patch_unsloth_rotary()


# ========================
# Fix Qwen3.5-VL rope_deltas stale batch-size bug con GRPO
# Durante la generation GRPO usa batch=num_generations (es. 4). Il modello salva
# rope_deltas.shape[0]=4. Nell'accumulated loss usa batch=1: 1//4=0 → delta vuoto
# → position_ids con batch=0 → cos/sin con batch=0 → mismatch con q/k (batch=1).
# Soluzione: hook pre-forward che azzera rope_deltas prima di ogni training step,
# così compute_3d_position_ids ritorna None e il testo usa cache_position.
# ========================
def _rope_deltas_reset_hook(module, args, kwargs):
    if hasattr(module, "rope_deltas"):
        module.rope_deltas = None

_rope_hook_registered = False
for _name, _mod in model.named_modules():
    if hasattr(_mod, "compute_3d_position_ids") and hasattr(_mod, "rope_deltas"):
        _mod.register_forward_pre_hook(_rope_deltas_reset_hook, with_kwargs=True)
        print(f"rope_deltas reset hook registrato su: {_name}")
        _rope_hook_registered = True
        break

if not _rope_hook_registered:
    print("WARN: rope_deltas reset hook NON registrato (modulo non trovato)")


# ========================
# Dataset
# ========================
print("Loading dataset...")
dataset = load_from_disk("dataset_hf")


def format_prompt(example):
    """
    Formatta l'esempio come prompt per GRPO.
    Il modello genera la completion (codice VGDL) che viene poi valutata
    dalle reward functions.
    """
    prompt = (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['description'].strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n"
    )
    return {"prompt": prompt}


dataset = dataset.map(format_prompt, remove_columns=["vgdl"])
print(f"Train size: {len(dataset['train'])} | Test size: {len(dataset['test'])}")


# ========================
# GRPO Config
# ========================
grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=8,      # effective batch = 8
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    fp16=False,
    # Logging & saving
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    # GRPO-specific
    num_generations=2,          # G: numero di completions per prompt per calcolare reward relativa
    max_prompt_length=400,      # token massimi per il prompt
    max_completion_length=512,  # token massimi per la completion (VGDL generato)
    temperature=0.9,            # deve essere > 0 per l'esplorazione durante training
    beta=0.01,                  # penalità KL (bassa per permettere divergenza dalla policy iniziale)
    use_vllm=True,                   # usa vLLM per generazione efficiente (richiede installazione vllm)
    vllm_gpu_memory_utilization=0.8,  # target utilizzo GPU per vLLM
)


# ========================
# GRPO Trainer
# ========================
trainer = GRPOTrainer(
    model=model,
    reward_funcs=REWARD_FUNCTIONS,
    args=grpo_config,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
)


# ========================
# Training
# ========================
print("Starting GRPO training...")
train_result = trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

metrics = {
    "train": train_result.metrics,
    "history": trainer.state.log_history,
}
os.makedirs(OUTPUT_DIR, exist_ok=True)
metrics_path = os.path.join(OUTPUT_DIR, "grpo_training_metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=2)

print(f"GRPO training completato! Modello salvato in {OUTPUT_DIR}")
print(f"Metriche salvate in {metrics_path}")
