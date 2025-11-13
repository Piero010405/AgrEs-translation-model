from transformers import AutoTokenizer

model_path = "./nllb_awajun_es_finetuned_light"
tokenizer = AutoTokenizer.from_pretrained(model_path)

tokens = [">>spa_Latn<<", ">>quz_Latn<<"]

for tok in tokens:
    tok_id = tokenizer.convert_tokens_to_ids(tok)
    print(f"{tok}: id={tok_id}")
