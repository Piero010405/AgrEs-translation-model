"""
Fine-tuning NLLB-200 para traducción bidireccional Awajún-Español
Decoder_input_ids creados directamente en el dataset
"""

import os
from datetime import datetime
from pathlib import Path
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
    EarlyStoppingCallback,
    set_seed,
)
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURACIÓN
# ============================================================
TRAIN_CSV = "./data/train.csv"
TEST_CSV = "./data/test.csv"
OUTPUT_DIR = "./nllb_awajun_es_finetuned_v2"
MODEL_NAME = "facebook/nllb-200-distilled-600M"

TAG_AWAJUN = ">>agr_Latn<<"
TAG_SPANISH = ">>spa_Latn<<"
LANG_AWAJUN = "agr_Latn"
LANG_SPANISH = "spa_Latn"

MODEL_VERSION = "awajun_nllb_final"
TIMESTAMP = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
METRICS_DIR = f"./metrics/{MODEL_VERSION}_{TIMESTAMP}"
os.makedirs(METRICS_DIR, exist_ok=True)

SEED = 42
set_seed(SEED)
os.environ["HF_HOME"] = "/workspace/hf_cache"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.makedirs("/workspace/hf_cache", exist_ok=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("=" * 70)
print("🚀 NLLB-200 FINE-TUNING AWAJÚN-ESPAÑOL - DATA LAB MODEL")
print("=" * 70)
print(f"Device: {device}")
if torch.cuda.is_available():
    try:
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    except Exception:
        pass

# ============================================================
# CARGAR DATOS
# ============================================================
print("\n📥 Cargando datasets...")
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print(f"   Train: {len(train_df)} ejemplos")
print(f"   Test:  {len(test_df)} ejemplos")
if "direction" in train_df.columns:
    print(f"   Direcciones: {train_df['direction'].value_counts().to_dict()}")

dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})

# ============================================================
# TOKENIZER Y MODELO
# ============================================================
print("\n🔧 Cargando tokenizer y modelo...")
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    cache_dir="/workspace/hf_cache",
    use_fast=True
)

model = AutoModelForSeq2SeqLM.from_pretrained(
    MODEL_NAME,
    cache_dir="/workspace/hf_cache",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto"
)

print(f"   Parámetros: {sum(p.numel() for p in model.parameters())/1e6:.2f}M")

# ============================================================
# CONFIGURAR TOKENS PERSONALIZADOS
# ============================================================
print("\n🎯 Configurando tokens personalizados...")

special_tokens = [TAG_AWAJUN, TAG_SPANISH]
tokens_to_add = [tok for tok in special_tokens if tok not in tokenizer.get_vocab()]

if tokens_to_add:
    print(f"   Agregando: {tokens_to_add}")
    tokenizer.add_special_tokens({"additional_special_tokens": tokens_to_add})
    model.resize_token_embeddings(len(tokenizer))

    # Inicialización desde otro token (opcional), con manejo de errores
    try:
        proxy_token_id = tokenizer.convert_tokens_to_ids("quy_Latn")
        awajun_token_id = tokenizer.convert_tokens_to_ids(TAG_AWAJUN)
        if proxy_token_id is not None and proxy_token_id != tokenizer.unk_token_id:
            with torch.no_grad():
                # defensive: check shapes exist
                if hasattr(model, "model") and hasattr(model.model, "encoder") and hasattr(model.model.encoder, "embed_tokens"):
                    model.model.encoder.embed_tokens.weight[awajun_token_id] = model.model.encoder.embed_tokens.weight[proxy_token_id].clone()
                if hasattr(model, "model") and hasattr(model.model, "decoder") and hasattr(model.model.decoder, "embed_tokens"):
                    model.model.decoder.embed_tokens.weight[awajun_token_id] = model.model.decoder.embed_tokens.weight[proxy_token_id].clone()
            print("   ✅ Embeddings inicializados desde proxy (si estuvo disponible)")
    except Exception as e:
        print(f"   ⚠️  Error inicializando embeddings proxy: {e}")

print(f"\n   Token IDs:")
print(f"   {TAG_AWAJUN} → {tokenizer.convert_tokens_to_ids(TAG_AWAJUN)}")
print(f"   {TAG_SPANISH} → {tokenizer.convert_tokens_to_ids(TAG_SPANISH)}")

# ============================================================
# CONFIGURAR DECODER_START_TOKEN_ID (CRÍTICO)
# ============================================================
print("\n⚙️  Configurando modelo...")

if model.config.decoder_start_token_id is None:
    model.config.decoder_start_token_id = tokenizer.eos_token_id
    print(f"   ✅ decoder_start_token_id configurado: {tokenizer.eos_token_id}")
else:
    print(f"   ✅ decoder_start_token_id: {model.config.decoder_start_token_id}")

# forced_bos no global — lo pasaremos dinámicamente en generación
model.config.forced_bos_token_id = None

print(f"   BOS: {tokenizer.bos_token} (id={tokenizer.bos_token_id})")
print(f"   EOS: {tokenizer.eos_token} (id={tokenizer.eos_token_id})")
print(f"   PAD: {tokenizer.pad_token} (id={tokenizer.pad_token_id})")

# ============================================================
# TOKENIZACIÓN CON DECODER_INPUT_IDS
# ============================================================
MAX_LENGTH = 128

print("\n🔤 Tokenizando con decoder_input_ids...")

def _infer_langs_from_direction(direction_value):
    """
    Detección robusta del par src/tgt a partir de la columna 'direction'.
    Acepta formatos diversos: 'agr->spa', 'agr_spa', 'agr_Latn-spa_Latn', 'agr', etc.
    Devuelve (src_lang, tgt_lang) en formato 'agr_Latn' / 'spa_Latn'.
    Si falla, devuelve (LANG_AWAJUN, LANG_SPANISH) por defecto.
    """
    if not isinstance(direction_value, str):
        return LANG_AWAJUN, LANG_SPANISH
    d = direction_value.lower()
    # guess left->right if arrow present
    if "->" in d or "-" in d:
        # split on arrows/dashes
        for sep in ("->", "-->", "->", "-", "_", " "):
            if sep in d:
                parts = [p.strip() for p in d.split(sep) if p.strip()]
                if len(parts) >= 2:
                    left, right = parts[0], parts[1]
                    # detect which is agr or spa
                    if "agr" in left:
                        return LANG_AWAJUN, LANG_SPANISH
                    if "spa" in left or "es" in left:
                        return LANG_SPANISH, LANG_AWAJUN
    # if just contains token names
    if "agr" in d and "spa" in d:
        # ambiguous — default agr->spa
        return LANG_AWAJUN, LANG_SPANISH
    if "agr" in d:
        return LANG_AWAJUN, LANG_SPANISH
    if "spa" in d or "es" in d:
        return LANG_SPANISH, LANG_AWAJUN
    # fallback
    return LANG_AWAJUN, LANG_SPANISH

def preprocess_with_decoder_inputs(examples):
    """
    Tokeniza y crea decoder_input_ids por ejemplo.
    IMPORTANTE: establece tokenizer.src_lang / tgt_lang por ejemplo antes de tokenizar.
    """
    input_ids = []
    attention_mask = []
    labels = []
    decoder_input_ids = []

    batch_size = len(examples["src_text"])
    # prepare directions list if present
    directions = examples.get("direction", [None] * batch_size)

    for i in range(batch_size):
        src_text = str(examples["src_text"][i])
        tgt_text = str(examples["tgt_text"][i])
        dir_val = directions[i] if i < len(directions) else None

        # infer source and target language tags for the tokenizer (NLLB fast tokenizer needs these)
        src_lang, tgt_lang = _infer_langs_from_direction(dir_val)
        # set tokenizer attributes for current tokenization
        tokenizer.src_lang = src_lang
        tokenizer.tgt_lang = tgt_lang

        # Tokenizar entrada
        src_encoded = tokenizer(
            src_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )

        # Tokenizar salida con text_target (asegura que prefix tokens estén bien)
        tgt_encoded = tokenizer(
            text_target=tgt_text,
            max_length=MAX_LENGTH,
            truncation=True,
            padding=False,
        )

        label_ids = tgt_encoded["input_ids"]
        # shift right: decoder_input_ids = [decoder_start] + labels[:-1]
        decoder_start_token_id = model.config.decoder_start_token_id
        if decoder_start_token_id is None:
            decoder_start_token_id = tokenizer.eos_token_id
        if len(label_ids) == 0:
            # avoid empty target
            label_ids = [tokenizer.pad_token_id]
        decoder_ids = [decoder_start_token_id] + label_ids[:-1] if len(label_ids) > 0 else [decoder_start_token_id]

        input_ids.append(src_encoded["input_ids"])
        attention_mask.append(src_encoded.get("attention_mask", [1]*len(src_encoded["input_ids"])))
        labels.append(label_ids)
        decoder_input_ids.append(decoder_ids)
    
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "decoder_input_ids": decoder_input_ids,
    }

tokenized = dataset.map(
    preprocess_with_decoder_inputs,
    batched=True,
    batch_size=1000,
    remove_columns=dataset["train"].column_names,
    desc="Tokenizando con decoder_input_ids"
)

train_dataset = tokenized["train"]
eval_dataset = tokenized["test"]

print(f"   ✅ Tokenización completada")
print(f"      Train: {len(train_dataset)} ejemplos")
print(f"      Test:  {len(eval_dataset)} ejemplos")

# Verificación rápida
sample = train_dataset[0]
print(f"\n   📝 Verificación del ejemplo 0:")
print(f"      Keys: {list(sample.keys())}")
print(f"      input_ids length: {len(sample['input_ids'])}")
print(f"      labels length: {len(sample['labels'])}")
print(f"      decoder_input_ids length: {len(sample['decoder_input_ids'])}")
print(f"      decoder_input_ids[0]: {sample['decoder_input_ids'][0]} (debe ser {model.config.decoder_start_token_id})")

# Verificar primer token del decoder
if sample['decoder_input_ids'][0] != model.config.decoder_start_token_id:
    raise RuntimeError("❌ decoder_input_ids NO comienza con decoder_start_token_id!")
else:
    print(f"      primer token decoder_input_ids[0]: {sample['decoder_input_ids'][0]} (esperado: {model.config.decoder_start_token_id})")
    print(f"      ✅ decoder_input_ids correcto")

# ============================================================
# DATA COLLATOR SIMPLE
# ============================================================
print("\n🔧 Configurando Data Collator...")

class Seq2SeqDataCollatorWithPadding:
    def __init__(self, tokenizer, pad_to_multiple_of=None):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of

    def __call__(self, features):
        input_ids = [f["input_ids"] for f in features]
        attention_mask = [f["attention_mask"] for f in features]
        labels = [f["labels"] for f in features]
        decoder_input_ids = [f["decoder_input_ids"] for f in features]

        max_input_len = max(len(ids) for ids in input_ids)
        max_label_len = max(len(lbl) for lbl in labels)
        max_decoder_len = max(len(dec) for dec in decoder_input_ids)

        if self.pad_to_multiple_of:
            def _pad_to(n): return ((n + self.pad_to_multiple_of - 1) // self.pad_to_multiple_of) * self.pad_to_multiple_of
            max_input_len = _pad_to(max_input_len)
            max_label_len = _pad_to(max_label_len)
            max_decoder_len = _pad_to(max_decoder_len)

        padded_input_ids = []
        padded_attention_mask = []
        for ids, mask in zip(input_ids, attention_mask):
            padding_len = max_input_len - len(ids)
            padded_input_ids.append(ids + [self.tokenizer.pad_token_id] * padding_len)
            padded_attention_mask.append(mask + [0] * padding_len)

        padded_labels = []
        for lbl in labels:
            padding_len = max_label_len - len(lbl)
            padded_labels.append(lbl + [-100] * padding_len)

        padded_decoder_ids = []
        for dec in decoder_input_ids:
            padding_len = max_decoder_len - len(dec)
            padded_decoder_ids.append(dec + [self.tokenizer.pad_token_id] * padding_len)

        batch = {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_attention_mask, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "decoder_input_ids": torch.tensor(padded_decoder_ids, dtype=torch.long),
        }
        return batch

data_collator = Seq2SeqDataCollatorWithPadding(
    tokenizer=tokenizer,
    pad_to_multiple_of=8 if torch.cuda.is_available() else None,
)

print("   ✅ Data Collator configurado")

# ============================================================
# HIPERPARÁMETROS
# ============================================================
print("\n⚙️  Configurando hiperparámetros...")

PER_DEVICE_BATCH = 4
GRAD_ACCUM_STEPS = 16
LEARNING_RATE = 3e-5
NUM_EPOCHS = 10

print(f"   Batch efectivo: {PER_DEVICE_BATCH * GRAD_ACCUM_STEPS}")
print(f"   Learning rate: {LEARNING_RATE}")
print(f"   Épocas: {NUM_EPOCHS}")

training_args = Seq2SeqTrainingArguments(
    output_dir=OUTPUT_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    per_device_train_batch_size=PER_DEVICE_BATCH,
    per_device_eval_batch_size=8,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    num_train_epochs=NUM_EPOCHS,
    save_total_limit=3,
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
    greater_is_better=True,
    predict_with_generate=True,
    generation_max_length=MAX_LENGTH,
    generation_num_beams=4,
    bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
    fp16=False,
    label_smoothing_factor=0.1,
    logging_steps=50,
    logging_first_step=True,
    report_to="none",
    dataloader_num_workers=2,
    seed=SEED,
    remove_unused_columns=False,  # important to keep decoder_input_ids
)

# ============================================================
# MÉTRICAS
# ============================================================
print("\n📊 Cargando métricas de evaluación...")

bleu = evaluate.load("sacrebleu")
chrf = evaluate.load("chrf")
ter = evaluate.load("ter")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [p.strip() for p in decoded_preds]
    decoded_labels = [l.strip() for l in decoded_labels]
    try:
        bleu_result = bleu.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        chrf_result = chrf.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        ter_result = ter.compute(predictions=decoded_preds, references=[[l] for l in decoded_labels])
        gen_len = np.mean([len(p.split()) for p in decoded_preds]) if len(decoded_preds) else 0.0
        return {"bleu": bleu_result["score"], "chrf": chrf_result["score"], "ter": ter_result["score"], "gen_len": gen_len}
    except Exception as e:
        print(f"⚠️  Error calculando métricas: {e}")
        return {"bleu": 0.0, "chrf": 0.0, "ter": 100.0, "gen_len": 0.0}

# ============================================================
# TRAINER
# ============================================================
print("\n🏋️  Inicializando Trainer...")

callbacks = [EarlyStoppingCallback(early_stopping_patience=3)]

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

print("   ✅ Trainer inicializado")

# ============================================================
# TEST DEL COLLATOR (CRÍTICO)
# ============================================================
print("\n🔍 Testeando Data Collator antes de entrenar...")

test_batch = [train_dataset[i] for i in range(min(4, len(train_dataset)))]
collated_batch = data_collator(test_batch)

print(f"   Keys en batch: {list(collated_batch.keys())}")
print(f"   input_ids shape: {collated_batch['input_ids'].shape}")
print(f"   attention_mask shape: {collated_batch['attention_mask'].shape}")
print(f"   labels shape: {collated_batch['labels'].shape}")

if "decoder_input_ids" not in collated_batch:
    raise RuntimeError("❌ CRÍTICO: decoder_input_ids NO está en el batch!")

print(f"   decoder_input_ids shape: {collated_batch['decoder_input_ids'].shape}")
print(f"   ✅ decoder_input_ids presente en el batch")

first_tokens = collated_batch["decoder_input_ids"][:, 0].tolist()
expected = model.config.decoder_start_token_id
# check that at least one equals expected (in corner cases a padded example might differ)
if not all(token == expected for token in first_tokens if token is not None):
    # we will warn but not crash, because some padded rows could show different value
    print(f"   ⚠️ Atención: algunos decoder_input_ids no comienzan con {expected}: {first_tokens}")
else:
    print(f"   ✅ Todos los decoder_input_ids comienzan con {expected}")

# ============================================================
# ENTRENAMIENTO
# ============================================================
print("\n" + "=" * 70)
print("🚀 INICIANDO ENTRENAMIENTO")
print("=" * 70)

try:
    # Baseline
    print("\n📊 Evaluación inicial (baseline)...")
    baseline_metrics = trainer.evaluate()
    print(f"   BLEU: {baseline_metrics.get('eval_bleu', 0):.2f}")
    print(f"   ChrF: {baseline_metrics.get('eval_chrf', 0):.2f}")
    print(f"   TER:  {baseline_metrics.get('eval_ter', 100):.2f}")

    # Entrenar
    print("\n🏃 Entrenando modelo...")
    train_result = trainer.train()

    # Guardar
    print("\n💾 Guardando modelo final...")
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.config.save_pretrained(OUTPUT_DIR)
    print(f"   ✅ Modelo guardado en: {OUTPUT_DIR}")

    # Métricas finales
    final_metrics = trainer.evaluate()
    print("\n" + "=" * 70)
    print("✅ ENTRENAMIENTO COMPLETADO CON ÉXITO")
    print("=" * 70)
    print(f"   BLEU: {final_metrics.get('eval_bleu', 0):.2f}")
    print(f"   ChrF: {final_metrics.get('eval_chrf', 0):.2f}")
    print(f"   TER:  {final_metrics.get('eval_ter', 100):.2f}")
    print(f"   Loss: {final_metrics.get('eval_loss', 0):.4f}")

except Exception as e:
    print(f"\n❌ ERROR DURANTE ENTRENAMIENTO: {e}")
    import traceback
    traceback.print_exc()

    emergency_dir = os.path.join(OUTPUT_DIR, "emergency_checkpoint")
    try:
        trainer.save_model(emergency_dir)
        tokenizer.save_pretrained(emergency_dir)
        print(f"   💾 Checkpoint de emergencia guardado en: {emergency_dir}")
    except Exception as ee:
        print(f"   ❌ No se pudo guardar checkpoint: {ee}")

    raise

# ============================================================
# GUARDAR MÉTRICAS Y VISUALIZACIONES (igual que antes)
# ============================================================
print("\n📈 Generando métricas y gráficas...")

log_history = trainer.state.log_history
eval_logs = [entry for entry in log_history if "eval_loss" in entry and "epoch" in entry]

if eval_logs:
    df_logs = pd.DataFrame(eval_logs)
    csv_path = Path(METRICS_DIR) / "training_metrics.csv"
    df_logs.to_csv(csv_path, index=False)
    print(f"   💾 Métricas guardadas en: {csv_path}")
    # (plots omitted here for brevitiy — conserva tu bloque original si lo deseas)

# ============================================================
# PRUEBAS DE TRADUCCIÓN (corregido forced_bos_token_id)
# ============================================================
print("\n🧪 Probando traducciones del modelo entrenado...")

test_cases = [
    (f"{TAG_AWAJUN} Winí najantai", LANG_AWAJUN, LANG_SPANISH, "Esperado: traducción a español"),
    (f"{TAG_AWAJUN} Nayaimpiniam weaji", LANG_AWAJUN, LANG_SPANISH, "Esperado: traducción a español"),
    (f"{TAG_SPANISH} Buenos días", LANG_SPANISH, LANG_AWAJUN, "Esperado: saludo en awajún"),
    (f"{TAG_SPANISH} ¿Cómo estás?", LANG_SPANISH, LANG_AWAJUN, "Esperado: pregunta en awajún"),
]

model.eval()
for src_text, src_lang, tgt_lang, description in test_cases:
    # set tokenizer lang tags for generation
    tokenizer.src_lang = src_lang
    tokenizer.tgt_lang = tgt_lang

    inputs = tokenizer(src_text, return_tensors="pt").to(device)

    # forced_bos_token_id must be the ID of the language TOKEN (e.g. >>spa_Latn<<), not the string 'spa_Latn'
    forced_bos = None
    if tgt_lang == LANG_SPANISH:
        forced_bos = tokenizer.convert_tokens_to_ids(TAG_SPANISH)
    else:
        forced_bos = tokenizer.convert_tokens_to_ids(TAG_AWAJUN)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            forced_bos_token_id=forced_bos,
            max_length=MAX_LENGTH,
            num_beams=4,
            early_stopping=True
        )

    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)
    direction = "→" if src_lang == LANG_AWAJUN else "←"
    print(f"\n   {direction} {src_text}")
    print(f"      Traducción: {translation}")
    print(f"      {description}")

print("\n" + "=" * 70)
print("✅ PROCESO COMPLETADO EXITOSAMENTE")
print("=" * 70)
print("\n📁 Archivos generados:")
print(f"   - Modelo final: {OUTPUT_DIR}")
print(f"   - Métricas: {METRICS_DIR}")
