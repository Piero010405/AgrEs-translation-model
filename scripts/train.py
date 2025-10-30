# =====================================================
# 🦜 Fine-tuning NLLB-200 para traducción Awajún → Español
# Optimizado para GPU RTX con PyTorch + CUDA
# =====================================================

# -----------------------------
# 1️⃣ Instalar librerías (ejecuta en terminal si no las tienes)
# -----------------------------
# pip install transformers datasets sacrebleu sentencepiece accelerate torch pandas

from datasets import Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import pandas as pd
import torch
import os

# =====================================================
# 🔹 0. Verificar GPU
# =====================================================
if torch.cuda.is_available():
    device_name = torch.cuda.get_device_name(0)
    print(f"✅ GPU detectada: {device_name}")
else:
    print("⚠️ GPU no detectada. Se usará CPU, entrenamiento muy lento.")
device = "cuda" if torch.cuda.is_available() else "cpu"

# =====================================================
# 🔹 1. Cargar dataset local (CSV)
# =====================================================
filename = "./data/awajun_spanish.csv"
df = pd.read_csv(filename, sep="\,")
df = df[["awajun", "spanish"]].dropna()

# Dividir train/test
train_df = df.sample(frac=0.9, random_state=42)
test_df = df.drop(train_df.index)

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df),
    "test": Dataset.from_pandas(test_df),
})

# =====================================================
# 🔹 2. Cargar modelo y tokenizer
# =====================================================
model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

# Activar gradient checkpointing para ahorrar memoria (opcional)
model.gradient_checkpointing_enable()

SRC_LANG = "ajg_Latn"  # Awajún
TGT_LANG = "spa_Latn"  # Español

tokenizer.src_lang = SRC_LANG
model.config.forced_bos_token_id = tokenizer.convert_tokens_to_ids(TGT_LANG)

# =====================================================
# 🔹 3. Tokenización
# =====================================================
def tokenize_function(examples):
    inputs = [str(text) for text in examples["awajun"]]
    targets = [str(text) for text in examples["spanish"]]

    model_inputs = tokenizer(
        inputs, max_length=128, padding="max_length", truncation=True
    )

    labels = tokenizer(
        targets, max_length=128, padding="max_length", truncation=True
    )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)
train_dataset = tokenized_datasets["train"]
eval_dataset = tokenized_datasets["test"]

# =====================================================
# 🔹 4. Configuración del entrenamiento
# =====================================================
output_dir = "./nllb_awajun_es_finetuned"
os.makedirs(output_dir, exist_ok=True)

training_args = Seq2SeqTrainingArguments(
    output_dir=output_dir,
    evaluation_strategy="epoch",
    learning_rate=3e-5,
    per_device_train_batch_size=4,  # ajusta según memoria GPU
    per_device_eval_batch_size=4,
    weight_decay=0.01,
    save_total_limit=2,
    num_train_epochs=8,
    predict_with_generate=True,
    fp16=True,  # optimizado para GPU
    logging_dir="./logs",
    report_to="none",  # evita errores si no se usa WandB
    save_strategy="epoch"
)

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# =====================================================
# 🔹 5. Entrenamiento
# =====================================================
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()

# =====================================================
# 🔹 6. Guardar modelo finetuneado
# =====================================================
trainer.save_model(output_dir)
tokenizer.save_pretrained(output_dir)

print(f"✅ Modelo guardado en: {output_dir}")
