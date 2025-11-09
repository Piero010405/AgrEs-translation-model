# ============================================================
# Fine-tuning NLLB-200 (versión ligera, robusta frente a NaNs)
# ============================================================

import os
import random
import math
import time
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

# ---------------------------
# Configs y paths
# ---------------------------
TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"
OUTPUT_DIR = "./nllb_awajun_es_finetuned_light"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

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
# Cargar datos (espera columnas awajun, spanish)
# ---------------------------
print("Cargando CSVs...")
train_df = pd.read_csv(TRAIN_CSV).dropna()
test_df = pd.read_csv(TEST_CSV).dropna()

train_df = train_df[["awajun", "spanish"]]
test_df = test_df[["awajun", "spanish"]]

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})
print(f"Train rows: {len(train_df)}, Test rows: {len(test_df)}")

# ---------------------------
# Tokenizer + modelo (ligero y estable)
# ---------------------------
print("Cargando tokenizer y modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir="/workspace/hf_cache")

# Carga clásica (sin device_map auto al principio) para evitar problemas al guardar config
model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir="/workspace/hf_cache",
    torch_dtype=torch.bfloat16,  # ✅ más estable y soportado por Ampere (RTX 30xx)
    device_map="auto"
)

# Evitar problemas al guardar luego:
model.config.model_type = getattr(model.config, "model_type", "nllb")

print(f"Modelo cargado. Params: {sum(p.numel() for p in model.parameters())/1e6:.2f} M")

# ---------------------------
# Tokenización robusta (NLLB)
# ---------------------------
MAX_LEN = 96
SRC_LANG = "quz_Latn"  # proxy para Awajún
TGT_LANG = "spa_Latn"
# Asignar idiomas al tokenizer si aplica:
setattr(tokenizer, "src_lang", SRC_LANG)
setattr(tokenizer, "tgt_lang", TGT_LANG)

def tokenize_function(examples):
    inputs = [str(x) for x in examples["awajun"]]
    targets = [str(x) for x in examples["spanish"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LEN, truncation=True, padding="max_length")
    # use text_target (nuevo API) para las etiquetas
    labels = tokenizer(text_target=targets, max_length=MAX_LEN, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizando dataset...")
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["awajun", "spanish"])
train_dataset = tokenized["train"]
eval_dataset = tokenized["test"]

# ---------------------------
# Parámetros "ligeros" y robustos para evitar NaNs
# ---------------------------
per_device_train_batch_size = 1
gradient_accumulation_steps = 4
learning_rate = 5e-5
num_epochs = 4

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    evaluation_strategy="epoch",
    per_device_train_batch_size=per_device_train_batch_size,
    per_device_eval_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    learning_rate=learning_rate,
    weight_decay=0.01,
    save_total_limit=2, 
    num_train_epochs=num_epochs,
    predict_with_generate=True,
    bf16=True,       # ✅ usar BF16
    fp16=False,      # ❌ no usar FP16
    logging_dir="./logs",
    report_to="none",
    save_strategy="epoch",
    dataloader_num_workers=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# ---------------------------
# Métricas robustas (evitar crash por preds vacías)
# ---------------------------
bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")

def compute_metrics(eval_pred):
    preds, labels = eval_pred
    if isinstance(preds, tuple):
        preds = preds[0]
    # Si no hay predicciones (fallo), devolver 0s para evitar NaNs
    if preds is None or len(preds) == 0:
        return {"bleu": 0.0, "chrf": 0.0, "gen_len": 0.0}
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    # Protegemos la computación en caso de strings vacíos
    try:
        bleu_res = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        chrf_res = chrf.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        gen_len = np.mean([len(p.split()) for p in decoded_preds]) if len(decoded_preds)>0 else 0.0
    except Exception as e:
        print("Warning: compute_metrics failed:", e)
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
# Entrenamiento con manejo de errores
# ---------------------------
print("Comenzando entrenamiento ligero...")
try:
    trainer.train()
except Exception as e:
    print("ERROR durante training:", e)
    # Intentar guardar estado parcial para debugging
    ckpt_dir = os.path.join(OUTPUT_DIR, "error_checkpoint")
    print("Guardando checkpoint parcial en:", ckpt_dir)
    try:
        trainer.save_model(ckpt_dir)
        tokenizer.save_pretrained(ckpt_dir)
        model.config.save_pretrained(ckpt_dir)
    except Exception as ee:
        print("Falló al guardar checkpoint parcial:", ee)
    raise e
else:
    # Guardado final y forzar config.json completo
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.config.save_pretrained(OUTPUT_DIR)
    print("Modelo guardado en", OUTPUT_DIR)

# ---------------------------
# Guardar métricas finales en CSV
# ---------------------------
final_metrics = trainer.evaluate()
pd.DataFrame([final_metrics]).to_csv("./training_metrics.csv", index=False)
print("Métricas finales:", final_metrics)

# ---------------------------
# Función de prueba rápida local (generación de ejemplo)
# ---------------------------
def quick_test(sentences, max_new_tokens=64):
    # Cargar modelo desde OUTPUT_DIR si quieres reusar guardado (opcional)
    if os.path.isdir(OUTPUT_DIR):
        m = AutoModelForSeq2SeqLM.from_pretrained(OUTPUT_DIR).to(device)
        t = AutoTokenizer.from_pretrained(OUTPUT_DIR)
    else:
        m = model
        t = tokenizer

    m.eval()
    inputs = t(sentences, return_tensors="pt", padding=True, truncation=True).to(device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=max_new_tokens, num_beams=4)
    decoded = t.batch_decode(out, skip_special_tokens=True)
    return decoded

# Ejemplo de uso:
examples = [
    "Tukuy maki ... (ejemplo en awajun u otro texto corto que quieras traducir)",
]
print("Ejemplo de generación:", quick_test(examples))

# ---------------------------
# FIN
# ---------------------------
