from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="openai-community/gpt2",   # oder "gpt2", aber ich würde genau wie in deiner Config nehmen
    local_dir="gpt2_snapshot",
    local_dir_use_symlinks=False
)
print("done")
