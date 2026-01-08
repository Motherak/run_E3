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
    print("[OK] Saved GPT2 to", d)

def save_detector():
    # ⚠️ Falls dein Detector anders heißt, musst du diese ID anpassen.
    # Das ist ein Platzhalter, der oft NICHT exakt passt.
    model_id = "GeorgeDrayson/modernbert-ai-detection"
    d = OUT / "modernbert_ai_detection"
    d.mkdir(parents=True, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    tok.save_pretrained(d)
    mdl.save_pretrained(d)
    print("[OK] Saved detector to", d)

if __name__ == "__main__":
    save_gpt2()
    save_detector()
