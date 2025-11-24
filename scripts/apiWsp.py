from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# -----------------------------------
# CARGA DEL MODELO
# -----------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./nllb_awajun_es_finetuned_v2"   # Asegúrate de que esta carpeta exista

print(f"Cargando modelo desde {model_path} en {device}...")

tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# Etiquetas de idioma
TAG_ES = ">>spa_Latn<<"
TAG_AW = ">>agr_Latn<<"   # Ajusta si tu modelo usa otra etiqueta


# -----------------------------------
# FUNCIÓN DE TRADUCCIÓN ES → AW
# -----------------------------------
def traducir_espanol_a_awajun(texto: str) -> str:
    entrada = f"{TAG_ES} {texto}"

    inputs = tokenizer(
        entrada,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(TAG_AW)

    output_ids = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        no_repeat_ngram_size=2,
        forced_bos_token_id=forced_bos_token_id
    )

    texto_traducido = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return texto_traducido.strip()


# -----------------------------------
# FASTAPI
# -----------------------------------
app = FastAPI(title="Traductor Español → Awajún")

class Peticion(BaseModel):
    texto: str


@app.post("/traducir")
def traducir(data: Peticion):
    traduccion = traducir_espanol_a_awajun(data.texto)
    return {
        "input_espanol": data.texto,
        "traduccion_awajun": traduccion
    }


@app.get("/")
def root():
    return {
        "mensaje": "API funcionando. Usa POST /traducir para traducir Español → Awajún."
    }
