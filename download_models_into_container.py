from pathlib import Path
from huggingface_hub import snapshot_download

OUT = Path("/opt/models")
OUT.mkdir(parents=True, exist_ok=True)

def dl(repo_id: str, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=str(out_dir),
        local_dir_use_symlinks=False,  # wichtig im Container
        # revision="main",  # optional
    )
    print(f"[OK] downloaded {repo_id} -> {out_dir}")

if __name__ == "__main__":
    dl("gpt2", OUT / "gpt2")

    # Detector aus dem Paper/Repo – bei dir war das die Idee:
    dl("GeorgeDrayson/modernbert-ai-detection", OUT / "modernbert_ai_detection")
