"""Workflow definitions for the Musubi GUI.

Curated workflows live in WORKFLOWS.  The helper at the bottom also discovers
new Musubi train/cache/generate scripts so newly-added upstream workflows show
up as experimental entries instead of being invisible.
"""

from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent
MUSUBI_DIR = PROJECT_DIR / "musubi-tuner"


def wf(
    label,
    prefix,
    network_module,
    model_fields,
    train_model_args,
    test_model_args=None,
    cache_latents_model_args=None,
    cache_text_model_args=None,
    architecture="image",
    generate_kind="image",
    extra_train_args=None,
    extra_test_args=None,
    supports_fp8_scaled=True,
    has_cache_latents=True,
    has_cache_text=True,
):
    scripts = {
        "train": f"{prefix}_train_network.py",
    }
    if has_cache_latents:
        scripts["cache_latents"] = f"{prefix}_cache_latents.py"
    if has_cache_text:
        scripts["cache_text"] = f"{prefix}_cache_text_encoder_outputs.py"
    if generate_kind:
        scripts["generate"] = f"{prefix}_generate_{generate_kind}.py"

    return {
        "label": label,
        "architecture": architecture,
        "enabled": True,
        "experimental": prefix != "krea2",
        "scripts": scripts,
        "network_module": network_module,
        "model_fields": model_fields,
        "train_model_args": train_model_args,
        "test_model_args": test_model_args or train_model_args,
        "cache_latents_model_args": cache_latents_model_args or {},
        "cache_text_model_args": cache_text_model_args or {},
        "default_train_args": extra_train_args or [],
        "default_test_args": extra_test_args or ["--steps", "8", "--seed", "1"],
        "sample_prompt_supported": True,
        "supports_fp8_scaled": supports_fp8_scaled,
    }


WORKFLOWS = {
    "krea2": wf(
        "Krea 2 Image LoRA",
        "krea2",
        "networks.lora_krea2",
        {
            "raw_model_path": "Krea 2 RAW model",
            "turbo_model_path": "Krea 2 Turbo model",
            "vae_model_path": "VAE model",
            "text_encoder_path": "Qwen3-VL text encoder",
        },
        {"--dit": "raw_model_path", "--vae": "vae_model_path", "--text_encoder": "text_encoder_path"},
        {"--dit": "turbo_model_path", "--vae": "vae_model_path", "--text_encoder": "text_encoder_path"},
        {"--vae": "vae_model_path"},
        {"--text_encoder": "text_encoder_path"},
        extra_train_args=[],
        extra_test_args=["--steps", "8", "--guidance_scale", "1", "--mu", "1.15", "--width", "1024", "--height", "1024", "--attn_mode", "torch", "--seed", "1"],
    ),
    "qwen_image": wf(
        "Qwen Image LoRA",
        "qwen_image",
        "networks.lora_qwen_image",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "text_encoder_path": "Text encoder"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder": "text_encoder_path"},
        cache_latents_model_args={"--vae": "vae_model_path"},
        cache_text_model_args={"--text_encoder": "text_encoder_path"},
    ),
    "zimage": wf(
        "Z-Image LoRA",
        "zimage",
        "networks.lora_zimage",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "text_encoder_path": "Text encoder / LLM"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder": "text_encoder_path"},
        cache_latents_model_args={"--vae": "vae_model_path"},
        cache_text_model_args={"--text_encoder": "text_encoder_path"},
    ),
    "wan": wf(
        "Wan Video/Image LoRA",
        "wan",
        "networks.lora_wan",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "t5_path": "T5 text encoder", "clip_path": "CLIP text encoder optional"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path", "--t5": "t5_path", "--clip": "clip_path"},
        cache_latents_model_args={"--vae": "vae_model_path"},
        cache_text_model_args={"--t5": "t5_path", "--clip": "clip_path"},
        architecture="video",
        generate_kind="video",
    ),
    "flux_2": wf(
        "Flux.2 LoRA",
        "flux_2",
        "networks.lora_flux_2",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "text_encoder_path": "Text encoder"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder": "text_encoder_path"},
        cache_latents_model_args={"--vae": "vae_model_path"},
        cache_text_model_args={"--text_encoder": "text_encoder_path"},
    ),
    "flux_kontext": wf(
        "Flux Kontext LoRA",
        "flux_kontext",
        "networks.lora_flux",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "text_encoder1_path": "Text encoder 1 / T5", "text_encoder2_path": "Text encoder 2 / CLIP"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder1": "text_encoder1_path", "--text_encoder2": "text_encoder2_path"},
        cache_latents_model_args={"--vae": "vae_model_path"},
        cache_text_model_args={"--text_encoder1": "text_encoder1_path", "--text_encoder2": "text_encoder2_path"},
    ),
    "framepack": wf(
        "FramePack LoRA",
        "fpack",
        "networks.lora_framepack",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "text_encoder1_path": "Text encoder 1", "text_encoder2_path": "Text encoder 2", "image_encoder_path": "Image encoder"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder1": "text_encoder1_path", "--text_encoder2": "text_encoder2_path", "--image_encoder": "image_encoder_path"},
        cache_latents_model_args={"--vae": "vae_model_path"},
        cache_text_model_args={"--text_encoder1": "text_encoder1_path", "--text_encoder2": "text_encoder2_path"},
        architecture="video",
        generate_kind="video",
    ),
    "hunyuan_video": wf(
        "Hunyuan Video LoRA",
        "hv",
        "networks.lora",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "text_encoder1_path": "Text encoder 1", "text_encoder2_path": "Text encoder 2"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder1": "text_encoder1_path", "--text_encoder2": "text_encoder2_path"},
        cache_latents_model_args={"--vae": "vae_model_path"},
        cache_text_model_args={"--text_encoder1": "text_encoder1_path", "--text_encoder2": "text_encoder2_path"},
        architecture="video",
        generate_kind="video",
        supports_fp8_scaled=False,
    ),
    "hunyuan_video_1_5": wf(
        "Hunyuan Video 1.5 LoRA",
        "hv_1_5",
        "networks.lora_hv_1_5",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "text_encoder_path": "Text encoder", "image_encoder_path": "Image encoder optional"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder": "text_encoder_path", "--image_encoder": "image_encoder_path"},
        cache_latents_model_args={"--vae": "vae_model_path"},
        cache_text_model_args={"--text_encoder": "text_encoder_path"},
        architecture="video",
        generate_kind="video",
    ),
    "ideogram4": wf(
        "Ideogram 4 LoRA",
        "ideogram4",
        "networks.lora_ideogram4",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "text_encoder_path": "Text encoder"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder": "text_encoder_path"},
        cache_latents_model_args={"--vae": "vae_model_path"},
        cache_text_model_args={"--text_encoder": "text_encoder_path"},
        supports_fp8_scaled=False,
    ),
    "kandinsky5": wf(
        "Kandinsky 5 LoRA",
        "kandinsky5",
        "networks.lora_kandinsky",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "text_encoder_qwen_path": "Qwen text encoder", "text_encoder_clip_path": "CLIP text encoder"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder_qwen": "text_encoder_qwen_path", "--text_encoder_clip": "text_encoder_clip_path"},
        cache_latents_model_args={"--vae": "vae_model_path"},
        cache_text_model_args={"--text_encoder_qwen": "text_encoder_qwen_path", "--text_encoder_clip": "text_encoder_clip_path"},
        architecture="video",
        generate_kind="video",
    ),
    "hidream_o1": wf(
        "HiDream-O1 LoRA",
        "hidream_o1",
        "networks.lora_hidream_o1",
        {"dit_model_path": "DiT model", "vae_model_path": "VAE model"},
        {"--dit": "dit_model_path", "--vae": "vae_model_path"},
        cache_latents_model_args={},
        cache_text_model_args={},
        has_cache_latents=False,
        has_cache_text=True,
        generate_kind="image",
    ),
}


def _human_label(prefix: str) -> str:
    return prefix.replace("_", " ").replace("hv", "hunyuan video").title() + " LoRA"


def _discover_workflows():
    """Best-effort discovery for new upstream Musubi scripts.

    This cannot know every model-specific network module or model layout, so
    discovered entries are marked experimental and use common DiT/VAE/text
    encoder fields. Curated WORKFLOWS entries always win.
    """
    discovered = {}
    if not MUSUBI_DIR.exists():
        return discovered
    train_scripts = list(MUSUBI_DIR.glob("*_train_network.py")) + list(MUSUBI_DIR.glob("*_train.py"))
    for script in train_scripts:
        name = script.name
        prefix = name.removesuffix("_train_network.py").removesuffix("_train.py")
        curated_train_scripts = {wf.get("scripts", {}).get("train") for wf in WORKFLOWS.values()}
        curated_prefixes = {s.removesuffix("_train_network.py").removesuffix("_train.py") for s in curated_train_scripts if s}
        if prefix in WORKFLOWS or prefix in discovered or prefix in curated_prefixes or name in curated_train_scripts:
            continue
        # Skip generic/non-model utilities if they ever match the pattern.
        if prefix in {"hv_train", "qwen_image_train"}:
            continue
        scripts = {"train": name}
        for cache_name in [f"{prefix}_cache_latents.py", f"{prefix}_cache_pixel.py"]:
            if (MUSUBI_DIR / cache_name).exists():
                scripts["cache_latents"] = cache_name
                break
        text_script = MUSUBI_DIR / f"{prefix}_cache_text_encoder_outputs.py"
        if text_script.exists():
            scripts["cache_text"] = text_script.name
        for gen_kind in ["image", "video"]:
            gen = MUSUBI_DIR / f"{prefix}_generate_{gen_kind}.py"
            if gen.exists():
                scripts["generate"] = gen.name
                break
        discovered[prefix] = {
            "label": _human_label(prefix),
            "architecture": "video" if "generate_video" in scripts.get("generate", "") else "image",
            "enabled": True,
            "experimental": True,
            "scripts": scripts,
            "network_module": f"networks.lora_{prefix}",
            "model_fields": {"dit_model_path": "DiT model", "vae_model_path": "VAE model", "text_encoder_path": "Text encoder"},
            "train_model_args": {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder": "text_encoder_path"},
            "test_model_args": {"--dit": "dit_model_path", "--vae": "vae_model_path", "--text_encoder": "text_encoder_path"},
            "cache_latents_model_args": {"--vae": "vae_model_path"},
            "cache_text_model_args": {"--text_encoder": "text_encoder_path"},
            "default_train_args": [],
            "default_test_args": ["--steps", "8", "--seed", "1"],
            "sample_prompt_supported": True,
            "supports_fp8_scaled": True,
        }
    return discovered


def all_workflows():
    workflows = dict(WORKFLOWS)
    for key, value in _discover_workflows().items():
        workflows.setdefault(key, value)
    return workflows


def enabled_workflows():
    return {k: v for k, v in all_workflows().items() if v.get("enabled", False)}


def get_workflow(name: str):
    workflows = enabled_workflows()
    return workflows.get(name) or workflows["krea2"]
