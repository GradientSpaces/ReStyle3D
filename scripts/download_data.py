from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="gradient-spaces/SceneTransfer",
    filename="demo.zip",
    repo_type="dataset",
    local_dir=".",  # Downloads to current directory
)
