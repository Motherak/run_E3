from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification

OUT = Path("/opt/models")
OUT.mkdir(parents=True, exist_ok=True)

def save_gpt2():
    d = OUT / "gpt2"
    d.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained("gpt2")
    mdl = AutoModelForCausalLM.from_pretrained("gpt2")
    tok.save_pretrained(d)
    mdl.save_pretrained(d)
    print("[OK] saved gpt2 ->", d)

def save_detector():
    # Laut Repo/README ist der Detector: GeorgeDrayson/modernbert-ai-detection
    # (das ist das Modell, das “machine-generated text detection” macht)
    model_id = "GeorgeDrayson/modernbert-ai-detection"
    d = OUT / "modernbert_ai_detection"
    d.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    tok.save_pretrained(d)
    mdl.save_pretrained(d)
    print("[OK] saved detector ->", d)

if __name__ == "__main__":
    save_gpt2()
    save_detector()
