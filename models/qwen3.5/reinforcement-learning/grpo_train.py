"""
GRPO training per generazione VGDL.

Parte dal modello Qwen3.5-4B fine-tunato con SFT e lo allena con reward basate su:
  - Eseguibilità VGDL (parser py-vgdl)
  - Struttura corretta (4 sezioni obbligatorie + BasicGame)
  - Classi sprite, effetti e condizioni di terminazione validi
  - Uso corretto di EOS per i bordi dello schermo

Eseguire dalla root del progetto:
  python models/qwen3.5/reinforcement-learning/grpo_train.py
"""

import sys
import os
# Deve essere impostato PRIMA di qualsiasi import di torch/unsloth,
# altrimenti torch si inizializza senza il flag e inductor tenta di
# compilare kernel Triton cercando un compilatore C (non disponibile in WSL).
os.environ["TORCHDYNAMO_DISABLE"] = "1"
os.environ["TORCHINDUCTOR_DISABLE"] = "1"

from unsloth import FastLanguageModel
import json
import torch
from tqdm import tqdm
from transformers import TrainerCallback

_REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "py-vgdl"))

from datasets import load_from_disk
from trl import GRPOTrainer, GRPOConfig
from reward_functions import REWARD_FUNCTIONS


# ── Configurazione ────────────────────────────────────────────────────────────

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


# ── Caricamento modello ───────────────────────────────────────────────────────

# Carica il modello base con l'adapter LoRA prodotto dall'SFT.
# unsloth gestisce il caricamento in 4-bit e applica automaticamente l'adapter.
print(f"Loading SFT model from {SFT_ADAPTER}...")
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=SFT_ADAPTER,
    max_seq_length=MAX_SEQ_LEN,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

# Attiva la modalità training e sblocca i parametri LoRA per l'aggiornamento GRPO
model.train()
for name, param in model.named_parameters():
    if "lora_" in name:
        param.requires_grad_(True)

# Gradient checkpointing: ricalcola le attivazioni durante il backward
# invece di tenerle in memoria, riducendo il consumo di VRAM
model.enable_input_require_grads()

trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total     = sum(p.numel() for p in model.parameters())
print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")


# ── Patch unsloth: fix rotary embedding ──────────────────────────────────────

# Durante GRPO, la generazione usa batch=num_generations ma il backward usa batch=1.
# Questo causa un mismatch nelle dimensioni di cos/sin nel rotary embedding.
# La patch allinea q_pass e k_pass alla dimensione effettiva del batch.
def _patch_unsloth_rotary():
    import pathlib
    cache_file = pathlib.Path("unsloth_compiled_cache/unsloth_compiled_module_qwen3_5.py")
    if not cache_file.exists():
        return
    src = cache_file.read_text(encoding="utf-8")
    old = "    q_embed = torch.cat([q_embed, q_pass], dim=-1)\n    k_embed = torch.cat([k_embed, k_pass], dim=-1)"
    new = "    q_embed = torch.cat([q_embed, q_pass[:q_embed.shape[0]]], dim=-1)\n    k_embed = torch.cat([k_embed, k_pass[:k_embed.shape[0]]], dim=-1)"
    if new in src:
        print("Unsloth rotary patch: already present.")
        return
    if old not in src:
        return
    cache_file.write_text(src.replace(old, new, 1), encoding="utf-8")
    print("Unsloth rotary patch: applied.")

_patch_unsloth_rotary()


# ── Fix rope_deltas per Qwen3.5 ──────────────────────────────────────────────

# Qwen3.5 salva rope_deltas con batch=num_generations durante la generazione.
# Quando il backward usa batch=1, il calcolo delle position_ids produce un
# tensore vuoto, causando un crash. Questo hook azzera rope_deltas prima di
# ogni forward pass in modo che vengano ricalcolati correttamente.
def _rope_deltas_reset_hook(module, args, kwargs):
    if hasattr(module, "rope_deltas"):
        module.rope_deltas = None

registered = False
for _name, _mod in model.named_modules():
    if hasattr(_mod, "compute_3d_position_ids") and hasattr(_mod, "rope_deltas"):
        _mod.register_forward_pre_hook(_rope_deltas_reset_hook, with_kwargs=True)
        print(f"rope_deltas hook registered on: {_name}")
        registered = True
        break

if not registered:
    print("WARN: rope_deltas hook not registered (module not found)")


# ── Dataset ───────────────────────────────────────────────────────────────────

print("Loading dataset...")
dataset = load_from_disk("dataset_hf")

def format_prompt(example):
    # Costruisce il prompt nel formato ChatML che il modello si aspetta.
    # Il blocco <think> vuoto segue il pattern di Qwen3 per il reasoning.
    return {"prompt": (
        f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n"
        f"<|im_start|>user\n{example['description'].strip()}<|im_end|>\n"
        f"<|im_start|>assistant\n<think>\n\n</think>\n"
    )}

dataset = dataset.map(format_prompt, remove_columns=["vgdl"])
print(f"Train: {len(dataset['train'])} examples | Test: {len(dataset['test'])} examples")


# ── Progress bar ──────────────────────────────────────────────────────────────

class TqdmProgressCallback(TrainerCallback):
    """Mostra una barra di avanzamento con le metriche chiave ad ogni step."""

    def on_train_begin(self, args, state, control, **kwargs):
        self._pbar = tqdm(
            total=state.max_steps,
            desc="GRPO",
            unit="step",
            dynamic_ncols=True,
        )
        self._postfix = {}

    def on_step_end(self, args, state, control, **kwargs):
        self._pbar.update(1)
        self._pbar.set_postfix(self._postfix, refresh=False)

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not logs:
            return
        # Mostra le metriche più utili per capire la qualità del training
        keys = ["loss", "reward", "kl", "grad_norm", "learning_rate"]
        self._postfix = {
            k: f"{logs[k]:.4f}" if isinstance(logs[k], float) else str(logs[k])
            for k in keys if k in logs
        }
        self._pbar.set_postfix(self._postfix, refresh=True)

    def on_train_end(self, args, state, control, **kwargs):
        self._pbar.close()


# ── GRPO Config ───────────────────────────────────────────────────────────────

grpo_config = GRPOConfig(
    output_dir=OUTPUT_DIR,
    # Training
    num_train_epochs=3,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,  # batch effettivo = 1 * 3 generazioni * 4 accum = 12
    learning_rate=5e-6,
    lr_scheduler_type="cosine",
    warmup_steps=10,
    bf16=True,
    fp16=False,
    optim="adamw_8bit",             # ~4x meno VRAM rispetto ad AdamW standard
    # Logging & salvataggio
    logging_steps=5,
    save_strategy="epoch",
    report_to="none",
    # Dataloader asincrono: i worker pre-caricano i batch mentre la GPU lavora
    dataloader_num_workers=2,
    dataloader_pin_memory=True,
    # GRPO: per ogni prompt genera 3 completions, calcola reward relativa e aggiorna
    num_generations=3,
    max_prompt_length=400,
    max_completion_length=384,
    temperature=0.9,
    beta=0.01,                      # peso della penalità KL verso la policy iniziale
    # vLLM disabilitato: Qwen3.5-4B ha un encoder visivo (SigLIP) non supportato
    # da unsloth fast_inference, rendendo impossibile la condivisione dei pesi
    # tra il modello di training e il motore vLLM.
    use_vllm=False,
)


# ── Training ──────────────────────────────────────────────────────────────────

trainer = GRPOTrainer(
    model=model,
    reward_funcs=REWARD_FUNCTIONS,
    args=grpo_config,
    train_dataset=dataset["train"],
    processing_class=tokenizer,
    callbacks=[TqdmProgressCallback()],
)

print("Starting GRPO training...")
train_result = trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

os.makedirs(OUTPUT_DIR, exist_ok=True)
metrics_path = os.path.join(OUTPUT_DIR, "grpo_training_metrics.json")
with open(metrics_path, "w") as f:
    json.dump({"train": train_result.metrics, "history": trainer.state.log_history}, f, indent=2)

print(f"Training complete. Model saved to {OUTPUT_DIR}")
print(f"Metrics saved to {metrics_path}")
