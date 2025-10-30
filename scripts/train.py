# =====================================================
# 🦜 Fine-tuning NLLB-200 para traducción Awajún → Español
# Optimizado para GPU RTX con PyTorch + CUDA
# Adaptado para GPU con ~6GB VRAM (RTX 3050).
# =====================================================

# -----------------------------
# 1️⃣ Instalar librerías (ejecuta en terminal si no las tienes)
# -----------------------------

import os
import math
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)

# Paths
TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"
OUTPUT_DIR = "./nllb_awajun_es_finetuned"

# Modelo
MODEL_NAME = "facebook/nllb-200-distilled-600M"

# =====================================================
# 0) GPU check
# =====================================================
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU detectada: {device_name}")
else:
    print("⚠️ GPU no detectada. Se usará CPU, entrenamiento muy lento.")
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 1) Cargar CSVs
# =====================================================
print("Cargando CSVs...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

# asegúrate que las columnas se llamen "awajun" y "spanish"
train_df = train_df[["awajun", "spanish"]].dropna()
test_df = test_df[["awajun", "spanish"]].dropna()

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})

# =====================================================
# 2) Tokenizer y modelo
# =====================================================
print("Cargando tokenizer y modelo...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME).to(device)
model.gradient_checkpointing_enable()  # ahorra memoria activando checkpointing

# Ajustes de idioma (NLLB utiliza tokens especiales de idioma)
SRC_LANG = "ajg_Latn"   # revisa si este es el código correcto, ajg_Latn es una suposición
TGT_LANG = "spa_Latn"

# Intentamos asignar forced_bos_token_id de forma segura
try:
    # Algunos tokenizers de NLLB exponen lang_code_to_id
    if hasattr(tokenizer, "lang_code_to_id") and TGT_LANG in tokenizer.lang_code_to_id:
        model.config.forced_bos_token_id = tokenizer.lang_code_to_id[TGT_LANG]
    else:
        # fallback: intenta buscar token literal (poco común)
        tok = tokenizer.convert_tokens_to_ids(TGT_LANG)
        if tok:
            model.config.forced_bos_token_id = tok
except Exception as e:
    print("No se pudo establecer forced_bos_token_id de forma automática:", e)

# =====================================================
# 3) Tokenización
# =====================================================
MAX_LEN = 96  # corto para ahorrar memoria; aumentar si tu GPU lo permite

def tokenize_function(examples):
    inputs = [str(x) for x in examples["awajun"]]
    targets = [str(x) for x in examples["spanish"]]
    model_inputs = tokenizer(inputs, max_length=MAX_LEN, truncation=True, padding="max_length")
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=MAX_LEN, truncation=True, padding="max_length")
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

print("Tokenizando dataset (esto puede tardar)...")
tokenized = dataset.map(tokenize_function, batched=True, remove_columns=["awajun", "spanish"])

train_dataset = tokenized["train"]
eval_dataset = tokenized["test"]

# =====================================================
# 4) Training params adaptados a 6GB VRAM
# =====================================================
# Recomendaciones (elige una combinación):
# - Opción segura (mínimo VRAM): per_device_train_batch_size=1, gradient_accumulation_steps=8 => eff batch 8
# - Si tienes un poco más de memoria: per_device_train_batch_size=2, gradient_accumulation_steps=4 => eff 8
# - Si te arriesgas: per_device_train_batch_size=4, gradient_accumulation_steps=2 => eff 8 (probablemente OOM)
per_device_train_batch_size = 1
gradient_accumulation_steps = 8

num_epochs = 8
learning_rate = 3e-5

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
    fp16=True,  # requiere soporte CUDA; reduce memoria
    logging_dir="./logs",
    report_to="none",
    save_strategy="epoch",
    dataloader_num_workers=2,
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# =====================================================
# 5) Trainer y start training
# =====================================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

print("Comenzando entrenamiento...")
trainer.train()

# =====================================================
# 6) Guardar modelo finetuneado
# =====================================================
print("Guardando modelo y tokenizer...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"✅ Modelo guardado en: {OUTPUT_DIR}")
