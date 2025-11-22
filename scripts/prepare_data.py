# scripts/prepare_data.py
"""
Preparación de datos bidireccional para Awajún-Español
Optimizado para fine-tuning de NLLB-200 con lengua no pre-entrenada
"""

import re
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# ============================================================
# CONFIGURACIÓN
# ============================================================
INPUT = Path("data/awajun_spanish.csv")
CLEANED = Path("data/awajun_spanish.cleaned_bidir.csv")
TRAIN = Path("data/train.csv")
TEST = Path("data/test.csv")

# Tags personalizados para Awajún (NO existe en NLLB-200)
TAG_AWAJUN = ">>agr_Latn<<"
TAG_SPANISH = ">>spa_Latn<<"

# Idiomas en formato NLLB
LANG_AWAJUN = "agr_Latn"
LANG_SPANISH = "spa_Latn"

# ============================================================
# REGEX PARA LIMPIEZA
# ============================================================
PREFIX_RE = re.compile(
    r'^\s*(?:<nav>)?\s*-?\s*\d+(?:\s*[-–—]\s*\d+)?\s*(?:</nav>)?\s*[:\-\u2013\u2014]?\s*',
    re.UNICODE
)
HTML_TAG_RE = re.compile(r'<[^>]+>')
MULTI_SPACE_RE = re.compile(r'\s{2,}')
MULTI_VERSE_RE = re.compile(r'\b\d+\s*[-–—]\s*\d+\b')


def clean_text(text: str) -> str:
    """
    Limpieza agresiva de texto para NMT.
    
    Args:
        text: Texto crudo
        
    Returns:
        Texto limpio sin HTML, prefijos numéricos, espacios múltiples
    """
    if not isinstance(text, str):
        return ""

    t = text.strip()

    # Remover HTML tags
    t = HTML_TAG_RE.sub(" ", t)

    # Remover prefijos de versículos bíblicos (1:2, - 15, etc.)
    t = PREFIX_RE.sub("", t)

    # Normalizar espacios
    t = MULTI_SPACE_RE.sub(" ", t)

    # Remover guiones iniciales
    t = re.sub(r'^\s*[-–—]+\s*', '', t)

    return t.strip()


def validate_pair(awajun: str, spanish: str, min_len: int = 3, max_len: int = 200) -> bool:
    """
    Valida que un par de frases sea útil para entrenamiento.
    
    Args:
        awajun: Texto en Awajún
        spanish: Texto en Español
        min_len: Longitud mínima de caracteres
        max_len: Longitud máxima de caracteres
        
    Returns:
        True si el par es válido
    """
    # Verificar longitud
    if len(awajun) < min_len or len(spanish) < min_len:
        return False

    if len(awajun) > max_len or len(spanish) > max_len:
        return False

    # Rechazar si contiene rangos de versículos (9-10)
    if MULTI_VERSE_RE.search(awajun) or MULTI_VERSE_RE.search(spanish):
        return False

    # Rechazar si es mayormente números
    if sum(c.isdigit() for c in awajun) / len(awajun) > 0.5:
        return False
    if sum(c.isdigit() for c in spanish) / len(spanish) > 0.5:
        return False

    return True


def create_bidirectional_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Crea dataset bidireccional con tags de idioma y metadata.
    
    IMPORTANTE: Para NLLB, los tags se agregan AL INICIO del texto fuente,
    y el modelo usa forced_bos_token_id para forzar el idioma objetivo.
    
    Args:
        df: DataFrame con columnas 'awajun' y 'spanish'
        
    Returns:
        DataFrame con columnas:
        - src_text: Texto fuente CON tag de idioma
        - tgt_text: Texto objetivo SIN tag
        - src_lang: Código de idioma fuente (agr_Latn o spa_Latn)
        - tgt_lang: Código de idioma objetivo (spa_Latn o agr_Latn)
        - direction: Dirección de traducción (agr2spa o spa2agr)
    """
    records = []

    # Dirección 1: Awajún → Español
    for _, row in df.iterrows():
        records.append({
            'src_text': f"{TAG_AWAJUN} {row['awajun']}",
            'tgt_text': row['spanish'],  # SIN tag en objetivo
            'src_lang': LANG_AWAJUN,
            'tgt_lang': LANG_SPANISH,
            'direction': 'agr2spa'
        })

    # Dirección 2: Español → Awajún
    for _, row in df.iterrows():
        records.append({
            'src_text': f"{TAG_SPANISH} {row['spanish']}",
            'tgt_text': row['awajun'],  # SIN tag en objetivo
            'src_lang': LANG_SPANISH,
            'tgt_lang': LANG_AWAJUN,
            'direction': 'spa2agr'
        })

    return pd.DataFrame(records)


def analyze_dataset(df: pd.DataFrame, name: str = "Dataset"):
    """Imprime estadísticas del dataset."""
    print(f"\n📊 Análisis de {name}")
    print(f"   Total de pares: {len(df)}")

    if 'direction' in df.columns:
        print(f"   Awajún→Español: {(df['direction']=='agr2spa').sum()}")
        print(f"   Español→Awajún: {(df['direction']=='spa2agr').sum()}")

    # Estadísticas de longitud
    src_lens = df['src_text'].str.len()
    tgt_lens = df['tgt_text'].str.len()

    print(f"   Longitud src: min={src_lens.min()}, max={src_lens.max()}, "
          f"mean={src_lens.mean():.1f}, median={src_lens.median():.1f}")
    print(f"   Longitud tgt: min={tgt_lens.min()}, max={tgt_lens.max()}, "
          f"mean={tgt_lens.mean():.1f}, median={tgt_lens.median():.1f}")

"""
Preparación de datos bidireccional Awajún-Español
Optimizado para fine-tuning de NLLB-200 con lengua no pre-entrenada
"""
def main():
    print("=" * 60)
    print("🚀 PREPARACIÓN DE DATOS AWAJÚN-ESPAÑOL BIDIRECCIONAL")
    print("=" * 60)

    # ============================================================
    # 1. CARGAR DATOS
    # ============================================================
    print(f"\n📥 Cargando {INPUT}")

    if not INPUT.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {INPUT}")

    df = pd.read_csv(INPUT)
    print(f"   Filas originales: {len(df)}")
    print(f"   Columnas: {df.columns.tolist()}")

    # Normalizar nombres de columnas
    cols_lower = [c.lower() for c in df.columns]
    expected = ["awajun", "spanish"]

    if not all(c in cols_lower for c in expected):
        if df.shape[1] >= 2:
            print("⚠️  Columnas no estándar detectadas, usando primeras dos columnas")
            df = df.iloc[:, :2]
            df.columns = ["awajun", "spanish"]
        else:
            raise RuntimeError(
                f"El CSV debe tener al menos dos columnas. "
                f"Columnas encontradas: {df.columns.tolist()}"
            )

    # ============================================================
    # 2. LIMPIEZA
    # ============================================================
    print("\n🧹 Limpiando datos...")

    df["awajun"] = df["awajun"].astype(str).apply(clean_text)
    df["spanish"] = df["spanish"].astype(str).apply(clean_text)

    # Filtrar pares inválidos
    initial_count = len(df)
    df = df.dropna(subset=["awajun", "spanish"])

    valid_mask = df.apply(
        lambda row: validate_pair(row["awajun"], row["spanish"]),
        axis=1
    )
    df = df[valid_mask]

    removed = initial_count - len(df)
    print(f"   ✅ {len(df)} pares válidos")
    print(f"   ❌ {removed} pares eliminados ({removed/initial_count*100:.1f}%)")

    # ============================================================
    # 3. REMOVER DUPLICADOS
    # ============================================================
    print("\n🔍 Removiendo duplicados...")

    initial_count = len(df)
    df = df.drop_duplicates(subset=["awajun", "spanish"], keep="first")
    duplicates = initial_count - len(df)

    print(f"   🗑️  {duplicates} duplicados eliminados")
    print(f"   ✅ {len(df)} pares únicos restantes")

    # ============================================================
    # 4. CREAR DATASET BIDIRECCIONAL
    # ============================================================
    print("\n🔁 Generando dataset bidireccional...")

    df_bidir = create_bidirectional_dataset(df)

    print(f"   ✅ Dataset bidireccional creado: {len(df_bidir)} ejemplos")
    print(f"      ({len(df)} originales × 2 direcciones)")

    analyze_dataset(df_bidir, "Dataset Bidireccional")

    # ============================================================
    # 5. GUARDAR DATASET COMPLETO
    # ============================================================
    CLEANED.parent.mkdir(parents=True, exist_ok=True)
    df_bidir.to_csv(CLEANED, index=False, encoding="utf-8-sig")
    print(f"\n💾 Dataset completo guardado en: {CLEANED}")

    # ============================================================
    # 6. SPLIT TRAIN/TEST ESTRATIFICADO
    # ============================================================
    print("\n✂️  Dividiendo train/test (90/10)...")

    # Split estratificado por dirección para mantener balance
    df_train, df_test = train_test_split(
        df_bidir,
        test_size=0.1,
        random_state=42,
        stratify=df_bidir['direction']  # Mantener mismo ratio de direcciones
    )

    # Guardar splits
    df_train.to_csv(TRAIN, index=False, encoding="utf-8-sig")
    df_test.to_csv(TEST, index=False, encoding="utf-8-sig")

    print(f"   🧠 Train: {len(df_train)} ejemplos")
    print(f"      - Awajún→Español: {(df_train['direction']=='agr2spa').sum()}")
    print(f"      - Español→Awajún: {(df_train['direction']=='spa2agr').sum()}")

    print(f"   🧪 Test:  {len(df_test)} ejemplos")
    print(f"      - Awajún→Español: {(df_test['direction']=='agr2spa').sum()}")
    print(f"      - Español→Awajún: {(df_test['direction']=='spa2agr').sum()}")

    # ============================================================
    # 7. ANÁLISIS FINAL
    # ============================================================
    print("\n" + "=" * 60)
    print("✅ PREPARACIÓN COMPLETADA")
    print("=" * 60)

    analyze_dataset(df_train, "Train Set")
    analyze_dataset(df_test, "Test Set")

    print("\n📝 Archivos generados:")
    print(f"   - {CLEANED}")
    print(f"   - {TRAIN}")
    print(f"   - {TEST}")

    print("\n⚠️  IMPORTANTE para el entrenamiento:")
    print(f"   - Tag Awajún: {TAG_AWAJUN}")
    print(f"   - Tag Español: {TAG_SPANISH}")
    print(f"   - Código Awajún: {LANG_AWAJUN}")
    print(f"   - Código Español: {LANG_SPANISH}")
    print("\n   Los tags se agregan SOLO en src_text, NO en tgt_text")
    print("   El modelo debe usar forced_bos_token_id para el idioma objetivo")


if __name__ == "__main__":
    main()
