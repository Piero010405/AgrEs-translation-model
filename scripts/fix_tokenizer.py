from transformers import AutoTokenizer

model_path = "./nllb_awajun_es_finetuned_light"

print(f"🔧 Cargando tokenizer desde {model_path}")
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokens = [">>spa_Latn<<", ">>quz_Latn<<"]
added = []

for tok in tokens:
    tok_id = tokenizer.convert_tokens_to_ids(tok)
    if tok_id == tokenizer.unk_token_id:
        tokenizer.add_tokens([tok])
        added.append(tok)
        print(f"➕ Token agregado: {tok}")
    else:
        print(f"✅ Token ya existe: {tok} (id={tok_id})")

# Guardar tokenizer actualizado
tokenizer.save_pretrained(model_path)
print(f"💾 Tokenizer actualizado y guardado en {model_path}")

# Revalidar
tokenizer = AutoTokenizer.from_pretrained(model_path)
for tok in tokens:
    print(f"✔️ {tok}: id={tokenizer.convert_tokens_to_ids(tok)}")
