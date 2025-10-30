# scripts/prepare_data.py
import re
import pandas as pd
from pathlib import Path

INPUT = Path("data/awajun_spanish.csv")
CLEANED = Path("data/awajun_spanish.cleaned.csv")
TRAIN = Path("data/train.csv")
TEST = Path("data/test.csv")

# regex para detectar prefijos problemáticos como:
# "9-10 ", "-10 ", "9–10 ", "9 — 10 ", "<nav>9-10<nav>"
PREFIX_RE = re.compile(r'^\s*(?:<nav>)?\s*-?\s*\d+(?:\s*[-–—]\s*\d+)?\s*(?:</nav>)?\s*[:\-\u2013\u2014]?\s*', re.UNICODE)
# limpiar tags HTML residuales
HTML_TAG_RE = re.compile(r'<[^>]+>')
# espacios duplicados
MULTI_SPACE_RE = re.compile(r'\s{2,}')

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    t = text.strip()
    # eliminar tags html si quedaron
    t = HTML_TAG_RE.sub(" ", t)
    # quitar prefijo tipo "9-10", "-10", "9–10", etc.
    t = PREFIX_RE.sub("", t)
    # reemplazar múltiples espacios
    t = MULTI_SPACE_RE.sub(" ", t)
    # limpiar guiones solitarios al inicio
    t = re.sub(r'^\s*[-–—]+\s*', '', t)
    return t.strip()

def main():
    print(f"Loading {INPUT}")
    df = pd.read_csv(INPUT)  # asume separador coma estándar
    # Asegurarse de que columnas esperadas existan (si no, intenta detectar)
    expected = ["awajun", "spanish"]
    cols_lower = [c.lower() for c in df.columns]
    if "awajun" not in cols_lower or "spanish" not in cols_lower:
        # intenta heurística
        print("Column names:", df.columns.tolist())
        # mapear por posición si es necesario
        df.columns = [c.strip() for c in df.columns]
        # si hay exactamente 2 columnas las renombramos
        if df.shape[1] >= 2:
            df = df.iloc[:, :2]
            df.columns = ["awajun", "spanish"]
        else:
            raise RuntimeError("CSV no contiene columnas suficientes. Asegúrate de que tenga al menos 2 columnas.")

    # aplicar limpieza
    df["awajun"] = df["awajun"].astype(str).apply(clean_text)
    df["spanish"] = df["spanish"].astype(str).apply(clean_text)

    # Eliminar filas vacías o con contenido muy corto en cualquiera de los lados
    df = df.dropna(subset=["awajun", "spanish"])
    df = df[(df["awajun"].str.len() > 1) & (df["spanish"].str.len() > 1)]

    # Opcional: eliminar filas donde awajun contiene múltiples versículos detectables (heurística)
    # Por ejemplo si sigue apareciendo "9-10" en medio del texto (poco robusto) se podría filtrar:
    multi_verse_re = re.compile(r'\b\d+\s*[-–—]\s*\d+\b')
    mask_multi = df["awajun"].str.contains(multi_verse_re)
    if mask_multi.any():
        print(f"Advertencia: {mask_multi.sum()} filas tienen patrones 'N-N' en awajun, se marcarán para inspección.")
        # Puedes decidir eliminarlas o guardarlas para revisión. Aquí las eliminamos.
        df = df[~mask_multi]

    # guardar cleaned CSV
    CLEANED.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED, index=False, encoding="utf-8-sig")
    print(f"Saved cleaned CSV: {CLEANED} ({len(df)} rows)")

    # train/test split
    df_train = df.sample(frac=0.9, random_state=42)
    df_test = df.drop(df_train.index)
    df_train.to_csv(TRAIN, index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST, index=False, encoding="utf-8-sig")
    print(f"Saved train ({len(df_train)}) and test ({len(df_test)}) CSVs.")

if __name__ == "__main__":
    main()
