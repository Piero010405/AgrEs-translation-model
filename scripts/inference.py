from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model_path = "./nllb_awajun_es_finetuned_light"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(device)

while True:
    text = input("👉 Ingresa texto en Awajún o Español (o 'salir'): ")
    if text.lower() == "salir":
        break

    inputs = tokenizer(text, return_tensors="pt").to(device)
    translated = model.generate(**inputs, max_length=200)
    print("🗣️ Traducción:", tokenizer.decode(translated[0], skip_special_tokens=True))
    print()
