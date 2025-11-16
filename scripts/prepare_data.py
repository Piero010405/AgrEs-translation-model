# scripts/prepare_data.py
import re
import pandas as pd
from pathlib import Path

# === CONFIG ===
INPUT = Path("data/awajun_spanish.csv")
CLEANED = Path("data/awajun_spanish.cleaned_bidir.csv")
TRAIN = Path("data/train.csv")
TEST = Path("data/test.csv")

# etiquetas NLLB (ajustadas a los idiomas involucrados)
TAG_AWAJUN = ">>agr_Latn<<" # Usamos una etiqueta para Awajún (agr_Latn) personalizada
TAG_SPANISH = ">>spa_Latn<<"

# === REGEX LIMPIEZA ===
PREFIX_RE = re.compile(r'^\s*(?:<nav>)?\s*-?\s*\d+(?:\s*[-–—]\s*\d+)?\s*(?:</nav>)?\s*[:\-\u2013\u2014]?\s*', re.UNICODE)
HTML_TAG_RE = re.compile(r'<[^>]+>')
MULTI_SPACE_RE = re.compile(r'\s{2,}')

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    t = HTML_TAG_RE.sub(" ", t)
    t = PREFIX_RE.sub("", t)
    t = MULTI_SPACE_RE.sub(" ", t)
    t = re.sub(r'^\s*[-–—]+\s*', '', t)
    return t.strip()

def main():
    print(f"📥 Cargando {INPUT}")
    df = pd.read_csv(INPUT)

    # Asegurar columnas esperadas
    expected = ["awajun", "spanish"]
    cols_lower = [c.lower() for c in df.columns]
    if not all(c in cols_lower for c in expected):
        print("Columnas detectadas:", df.columns.tolist())
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["awajun", "spanish"]
        else:
            raise RuntimeError("El CSV debe tener al menos dos columnas (awajun, spanish).")

    # === LIMPIEZA ===
    df["awajun"] = df["awajun"].astype(str).apply(clean_text)
    df["spanish"] = df["spanish"].astype(str).apply(clean_text)

    # Filtrar vacíos o cortos
    df = df.dropna(subset=["awajun", "spanish"])
    df = df[(df["awajun"].str.len() > 1) & (df["spanish"].str.len() > 1)]

    # Remover posibles rangos de versículos tipo 9–10
    multi_verse_re = re.compile(r'\b\d+\s*[-–—]\s*\d+\b')
    mask_multi = df["awajun"].str.contains(multi_verse_re)
    if mask_multi.any():
        print(f"⚠️ {mask_multi.sum()} filas con patrones 'N-N' eliminadas.")
        df = df[~mask_multi]

    # === DUPLICAR PARA BIDIRECCIONAL ===
    print("🔁 Generando dataset bidireccional...")
    df_forward = df.copy()
    df_forward["src_text"] = TAG_AWAJUN + " " + df_forward["awajun"]
    df_forward["tgt_text"] = df_forward["spanish"]

    df_reverse = df.copy()
    df_reverse["src_text"] = TAG_SPANISH + " " + df_reverse["spanish"]
    df_reverse["tgt_text"] = df_reverse["awajun"]

    df_bidir = pd.concat([df_forward[["src_text", "tgt_text"]],
                          df_reverse[["src_text", "tgt_text"]]], ignore_index=True)

    print(f"✅ Dataset combinado: {len(df_bidir)} filas ({len(df)} originales × 2)")

    # === GUARDAR ===
    CLEANED.parent.mkdir(parents=True, exist_ok=True)
    df_bidir.to_csv(CLEANED, index=False, encoding="utf-8-sig")
    print(f"💾 Guardado dataset bidireccional limpio en: {CLEANED}")

    # Split train/test
    df_train = df_bidir.sample(frac=0.9, random_state=42)
    df_test = df_bidir.drop(df_train.index)
    df_train.to_csv(TRAIN, index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST, index=False, encoding="utf-8-sig")

    print(f"🧠 Train: {len(df_train)} | Test: {len(df_test)}")

if __name__ == "__main__":
    main()
