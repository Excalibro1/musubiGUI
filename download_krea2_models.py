from pathlib import Path

import requests
from huggingface_hub import get_hf_file_metadata, hf_hub_url
from tqdm.auto import tqdm

root = Path(__file__).resolve().parent / "models"
root.mkdir(parents=True, exist_ok=True)

files = [
    ("krea/Krea-2-Raw", "raw.safetensors", "raw.safetensors"),
    ("krea/Krea-2-Turbo", "turbo.safetensors", "turbo.safetensors"),
    ("Comfy-Org/Qwen-Image_ComfyUI", "split_files/vae/qwen_image_vae.safetensors", "qwen_image_vae.safetensors"),
    ("Comfy-Org/Qwen3-VL", "text_encoders/qwen3vl_4b_bf16.safetensors", "qwen3vl_4b_bf16.safetensors"),
]


def human_size(num: int | None) -> str:
    if num is None:
        return "unknown size"
    for unit in ["B", "KB", "MB", "GB", "TB"]:
        if num < 1024:
            return f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} PB"


def download_file(repo: str, filename: str, local_name: str) -> None:
    dst = root / local_name
    tmp = dst.with_suffix(dst.suffix + ".part")

    url = hf_hub_url(repo_id=repo, filename=filename)
    metadata = get_hf_file_metadata(url)
    size = metadata.size
    download_url = metadata.location

    if dst.exists() and size is not None and dst.stat().st_size == size:
        print(f"Already downloaded: {dst} ({human_size(size)})")
        return

    resume_pos = tmp.stat().st_size if tmp.exists() else 0
    headers = {}
    mode = "wb"
    if resume_pos and size and resume_pos < size:
        headers["Range"] = f"bytes={resume_pos}-"
        mode = "ab"
        print(f"Resuming: {dst.name} from {human_size(resume_pos)} / {human_size(size)}")
    else:
        resume_pos = 0
        print(f"Downloading: {repo}/{filename}")
        print(f"To: {dst}")
        print(f"Size: {human_size(size)}")

    with requests.get(download_url, headers=headers, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(tmp, mode) as f, tqdm(
            total=size,
            initial=resume_pos,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            desc=dst.name,
            dynamic_ncols=True,
        ) as bar:
            for chunk in r.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    f.write(chunk)
                    bar.update(len(chunk))

    tmp.replace(dst)
    print(f"Saved: {dst}\n")


for repo, filename, local_name in files:
    download_file(repo, filename, local_name)

print("Done. Models are in:", root)
