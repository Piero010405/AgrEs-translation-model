from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import sys

# -------------------------------
# CONFIGURACIÓN INICIAL
# -------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
model_path = "./nllb_awajun_es_finetuned_light"

print(f"\n📦 Cargando modelo desde: {model_path}")
print(f"💻 Dispositivo activo: {device.upper()}")

# Cargar modelo y tokenizador
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

# -------------------------------
# ETIQUETAS DE IDIOMA (ajusta si tu dataset usó otras)
# -------------------------------
TAG_ES = ">>spa_Latn<<"
TAG_AW = ">>agr_Latn<<"  # o la que usaste (>>awb_Latn<<, >>awa_Latn<<, etc.)

# Validar que existan en el vocabulario
for tag in [TAG_ES, TAG_AW]:
    tok_id = tokenizer.convert_tokens_to_ids(tag)
    if tok_id is None:
        print(f"⚠️  Advertencia: la etiqueta {tag} no existe en el vocabulario del tokenizador.")
        print("   → Verifica las etiquetas usadas durante el entrenamiento.\n")
    else:
        print(f"✅ Etiqueta {tag} encontrada (id={tok_id})")

# -------------------------------
# FUNCIÓN DE TRADUCCIÓN
# -------------------------------
def traducir(texto, src_tag, tgt_tag):
    # Tokenizar texto con etiqueta de idioma origen
    entrada = f"{src_tag} {texto}"
    inputs = tokenizer(
        entrada,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=512
    ).to(device)

    forced_bos_token_id = tokenizer.convert_tokens_to_ids(tgt_tag)

    # Generar traducción
    translated = model.generate(
        **inputs,
        max_length=256,
        num_beams=4,
        no_repeat_ngram_size=2,
        forced_bos_token_id=forced_bos_token_id
    )

    output = tokenizer.decode(translated[0], skip_special_tokens=True)
    return output.strip()

# -------------------------------
# INTERFAZ INTERACTIVA
# -------------------------------
while True:
    print("\n🌐 MODOS DE TRADUCCIÓN:")
    print("1️⃣  Español → Awajún")
    print("2️⃣  Awajún → Español")
    print("💤  Escribe 'salir' para terminar.\n")

    choice = input("👉 Elige opción (1 o 2): ").strip().lower()
    if choice == "salir":
        sys.exit()
    elif choice not in ["1", "2"]:
        print("❌ Opción inválida. Intenta de nuevo.\n")
        continue

    if choice == "1":
        src_tag, tgt_tag = TAG_ES, TAG_AW
        print("\n🔹 Modo: Español → Awajún\n")
    else:
        src_tag, tgt_tag = TAG_AW, TAG_ES
        print("\n🔹 Modo: Awajún → Español\n")

    while True:
        text = input("✍️  Ingresa texto (o 'cambiar' / 'salir'): ").strip()
        if text.lower() == "salir":
            sys.exit()
        elif text.lower() == "cambiar":
            print()
            break
        elif not text:
            continue

        try:
            output = traducir(text, src_tag, tgt_tag)
            print("🗣️ Traducción:", output)
            print()
        except Exception as e:
            print("⚠️ Error al traducir:", e)
            print()
