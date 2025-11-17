# ============================================================
# Fine-tuning NLLB-200 - Awajún tag personalizada + entrenamiento ligero (awajun_token_train)
# ============================================================

import os
import time
import math
import pandas as pd
import numpy as np
import torch
import evaluate
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)
import matplotlib.pyplot as plt
from datetime import datetime

# ---------------------------
# Configs y paths
# ---------------------------
TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"
OUTPUT_DIR = "./nllb_awajun_es_finetuned_light"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# Nombre para métricas/plots
MODEL_VERSION = "awajun_token_train"
TIMESTAMP = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
METRICS_DIR = f"./metrics/{MODEL_VERSION}_{TIMESTAMP}"
os.makedirs(METRICS_DIR, exist_ok=True)

# ---------------------------
# Reproducibilidad y device
# ---------------------------
SEED = 42
set_seed(SEED)
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.makedirs("/workspace/hf_cache", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))

# ---------------------------
# Cargar datos (espera columnas src_text, tgt_text)
# ---------------------------
print("Cargando CSVs...")
train_df = pd.read_csv(TRAIN_CSV).dropna()
test_df = pd.read_csv(TEST_CSV).dropna()

# Asegurarnos de columnas correctas
required_cols = {"src_text", "tgt_text"}
if not required_cols.issubset(set(train_df.columns)) or not required_cols.issubset(set(test_df.columns)):
    raise RuntimeError(f"Los CSV deben contener las columnas: {required_cols}. Revisa tus archivos.")

train_df = train_df[["src_text", "tgt_text"]]
test_df = test_df[["src_text", "tgt_text"]]

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# ---------------------------
# Tokenizer + modelo (base)
# ---------------------------
print("Cargando tokenizer y modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/workspace/hf_cache")

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir="/workspace/hf_cache",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

model.config.model_type = getattr(model.config, "model_type", "nllb")
print(f"Modelo cargado. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

# ---------------------------
# Tags personalizados
# ---------------------------
# Definimos las etiquetas EXACTAS del dataset generado
TAG_AWAJUN = ">>agr_Latn<<"   # etiqueta personalizada Awajún
TAG_SPANISH = ">>spa_Latn<<"  # etiqueta español

special_tokens = [TAG_AWAJUN, TAG_SPANISH]

# Agregar tokens si no están en el vocab
tokens_to_add = []
for tok in special_tokens:
    if tok not in tokenizer.get_vocab():
        tokens_to_add.append(tok)

if tokens_to_add:
    print("Añadiendo special tokens al tokenizer:", tokens_to_add)
    tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    model.resize_token_embeddings(len(tokenizer))
else:
    print("Tags ya presentes en tokenizer vocab.")

# Mostrar IDs
print(f"✅ Etiqueta {TAG_AWAJUN} id={tokenizer.convert_tokens_to_ids(TAG_AWAJUN)}")
print(f"✅ Etiqueta {TAG_SPANISH} id={tokenizer.convert_tokens_to_ids(TAG_SPANISH)}")

# ---------------------------
# FIX DEL BUG DE NLLB:
# ---------------------------------------------------
# Obligatorio establecer src_lang y tgt_lang ANTES de tokenizar
# De lo contrario prefix_tokens queda como None → ERROR
# ---------------------------------------------------
tokenizer.src_lang = "agr_Latn"
tokenizer.tgt_lang = "spa_Latn"

# ---------------------------
# Tokenización robusta (NLLB) - Bidireccional
# ---------------------------
MAX_LEN = 96

def tokenize_function(examples):

    inputs  = [str(x) for x in examples["src_text"]]  # ya vienen con tags
    targets = [str(x) for x in examples["tgt_text"]]

    # ------------------------
    # Encoding fuente
    # ------------------------
    model_inputs = tokenizer(
        inputs,
        max_length=MAX_LEN,
        truncation=True,
        padding="max_length",
    )

    # ------------------------
    # Encoding objetivo
    # IMPORTANTE:
    # No usar text_target= (bug de NLLB → NoneType error)
    # ------------------------
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            targets,
            max_length=MAX_LEN,
            truncation=True,
            padding="max_length",
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizando dataset...")

tokenized = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["src_text", "tgt_text"]
)

train_dataset = tokenized["train"]
eval_dataset  = tokenized["test"]
print("Tokenización completada.")

# ---------------------------
# Training args (cortos para pruebas)
# ---------------------------
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 5e-5
num_epochs = 2   # pocas épocas para pruebas

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    bf16=True if torch.cuda.is_available() else False,  # bf16 en GPU si disponible
    fp16=False,
    logging_dir="./logs",
    report_to="none",
    save_strategy="epoch",
    dataloader_num_workers=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# ---------------------------
# Data collator
# ---------------------------
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ---------------------------
# Métricas
# ---------------------------
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    if preds is None or len(preds) == 0:
        return {"bleu": 0.0, "chrf": 0.0, "gen_len": 0.0}
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    try:
        bleu_res = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        chrf_res = chrf.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        gen_len = np.mean([len(p.split()) for p in decoded_preds]) if len(decoded_preds) > 0 else 0.0
    except Exception as e:
        print("Warning compute_metrics failed:", e)
        return {"bleu": 0.0, "chrf": 0.0, "gen_len": 0.0}
    return {"bleu": bleu_res["score"], "chrf": chrf_res["score"], "gen_len": gen_len}

# ---------------------------
# Callbacks (early stopping)
# ---------------------------
callbacks = [EarlyStoppingCallback(early_stopping_patience=2)]

# ---------------------------
# Trainer
# ---------------------------
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=callbacks,
)

# ---------------------------
# Entrenamiento
# ---------------------------
print("Comenzando entrenamiento (bidireccional, etiquetas personalizadas)...")
try:
    trainer.train()
except Exception as e:
    print("ERROR durante training:", e)
    # Guardado parcial
    ckpt_dir = os.path.join(OUTPUT_DIR, "error_checkpoint")
    try:
        trainer.save_model(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        model.config.save_pretrained(ckpt_dir)
        print("Checkpoint parcial guardado en:", ckpt_dir)
    except Exception as ee:
        print("Falló al guardar checkpoint parcial:", ee)
    raise e
else:
    # Guardado final
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.config.save_pretrained(OUTPUT_DIR)
    print("Modelo guardado en", OUTPUT_DIR)

# ---------------------------
# Guardar métricas en CSV + plots por epoch
# ---------------------------
log_hist = trainer.state.log_history
# extraer solo epoch eval entries
rows = []
for entry in log_hist:
    # entries típicas: {'eval_loss':..., 'eval_bleu':..., 'epoch':1.0, ...}
    if "epoch" in entry and ("eval_loss" in entry or "loss" in entry):
        rows.append(entry)

if len(rows) == 0:
    print("No hay entradas de log para guardar métricas (log_history vacío o no hubo evaluación).")
else:
    df_logs = pd.DataFrame(rows)
    csv_path = os.path.join(METRICS_DIR, "training_metrics_log_history.csv")
    df_logs.to_csv(csv_path, index=False)
    print("Saved training log CSV to:", csv_path)

    # Intentar extraer por epoch métricas eval_loss, eval_bleu, eval_chrf
    # ordenamos por epoch
    df_epoch = df_logs.sort_values("epoch").reset_index(drop=True)

    # Plots
    def save_plot(x, y, title, ylabel, fname):
        plt.figure(figsize=(8,4))
        plt.plot(x, y, marker="o")
        plt.title(title)
        plt.xlabel("epoch")
        plt.ylabel(ylabel)
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(METRICS_DIR, fname))
        plt.close()

    epochs = df_epoch["epoch"].tolist()
    if "eval_loss" in df_epoch.columns:
        save_plot(epochs, df_epoch["eval_loss"].tolist(), "Eval Loss por Epoch", "Eval Loss", "eval_loss.png")
    if "eval_bleu" in df_epoch.columns:
        save_plot(epochs, df_epoch["eval_bleu"].tolist(), "Eval BLEU por Epoch", "BLEU", "eval_bleu.png")
    if "eval_chrf" in df_epoch.columns:
        save_plot(epochs, df_epoch["eval_chrf"].tolist(), "Eval ChrF por Epoch", "ChrF", "eval_chrf.png")

    print("Saved plots to:", METRICS_DIR)

print("Entrenamiento completado.")

# ---------------------------
# FIN
# ---------------------------
