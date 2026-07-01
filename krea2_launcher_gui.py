import json
import os
import re
import shutil
import subprocess
import threading
import time
import zipfile
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk

from PIL import Image, ImageTk

from musubi_workflows import WORKFLOWS, enabled_workflows, get_workflow

project = Path(__file__).resolve().parent
root_dir = project / "musubi-tuner"
py = root_dir / ".venv-sage" / "Scripts" / "python.exe"
cuda126 = Path("C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/v12.6")
settings_path = project / "krea2_settings.json"
settings_gui = project / "krea2_settings_gui.py"
script_runner = project / "run_musubi_script.py"
dataset = project / "configs" / "dataset_krea2.toml"
sample_prompts = project / "configs" / "sample_prompts.txt"
models = project / "models"
logs_dir = project / "logs"

DEFAULTS = {
    "workflow": "krea2",
    "train_resolutions": ["1024x1024"],
    "lora_name": "krea2_lora", "epochs": 8, "max_train_steps": 0, "learning_rate": "1e-4",
    "network_dim": 32, "network_alpha": 32, "optimizer_type": "adamw8bit",
    "mixed_precision": "bf16", "save_precision": "float", "attention_mode": "sdpa", "blocks_to_swap": 8,
    "block_swap_ring_size": 2, "use_pinned_memory_for_block_swap": True,
    "block_swap_h2d_only": True, "fp8_base": True, "fp8_scaled": True,
    "gradient_checkpointing": True, "timestep_sampling": "krea2_shift", "discrete_flow_shift": "2.5",
    "weighting_scheme": "none", "network_dropout": "", "save_every_steps": 250, "sample_every_steps": 250,
    "test_lora_multiplier": "1.0", "test_width": 1024, "test_height": 1024,
    "test_steps": 8, "test_guidance_scale": "1", "test_seed_mode": "fixed", "test_seed": 1, "test_num_images": 1,
    "test_mu": "1.15", "test_negative_prompt": "", "test_attn_mode": "torch",
    "enable_training_samples": True, "max_data_loader_n_workers": 2,
    "gradient_accumulation_steps": 1, "max_grad_norm": "1.0", "lr_scheduler": "constant",
    "lr_warmup_steps": 0, "lr_decay_steps": 0, "lr_scheduler_num_cycles": 1,
    "lr_scheduler_power": "1.0", "lr_scheduler_min_lr_ratio": "", "optimizer_args": "",
    "lr_scheduler_args": "", "sigmoid_scale": "1.0", "logit_mean": "0.0",
    "logit_std": "1.0", "mode_scale": "1.29", "min_timestep": "", "max_timestep": "",
    "num_timestep_buckets": 0, "scale_weight_norms": "", "training_comment": "",
    "metadata_title": "", "metadata_author": "", "metadata_description": "",
    "metadata_license": "", "metadata_tags": "", "save_last_n_steps": 0,
    "save_last_n_epochs": 0, "save_last_n_steps_state": 0, "save_last_n_epochs_state": 0,
    "log_prefix": "", "log_tracker_name": "", "compile": False, "compile_backend": "inductor",
    "compile_mode": "default", "compile_dynamic": "auto", "compile_fullgraph": False,
    "guidance_scale": "", "vae_dtype": "", "wan_task": "t2v-14B", "timestep_boundary": "",
    "fp8_llm": False, "fp8_vl": False, "fp8_t5": False, "fp8_text_encoder": False,
    "full_bf16": False, "use_32bit_attention": False, "split_attn": False,
    "flash_attn": False, "flash3": False,
    "cuda_allow_tf32": False, "cuda_cudnn_benchmark": False, "gradient_checkpointing_cpu_offload": False,
    "sample_at_first": False, "preserve_distribution_shape": False, "log_config": False,
    "no_metadata": False, "dim_from_weights": False, "save_state": False,
    "save_state_on_train_end": False, "img_in_txt_in_offloading": False, "disable_numpy_memmap": False,
    "persistent_data_loader_workers": True, "seed": 42,
    "enable_tensorboard": True,
    "train_images_dir": str(project / "train_images"),
    "cache_dir": str(project / "cache"),
    "raw_model_path": str(project / "models" / "raw.safetensors"),
    "turbo_model_path": str(project / "models" / "turbo.safetensors"),
    "dit_model_path": "",
    "vae_model_path": str(project / "models" / "qwen_image_vae.safetensors"),
    "text_encoder_path": str(project / "models" / "qwen3vl_4b_bf16.safetensors"),
    "text_encoder1_path": "",
    "text_encoder2_path": "",
    "t5_path": "",
    "clip_path": "",
    "image_encoder_path": "",
    "text_encoder_qwen_path": "",
    "text_encoder_clip_path": "",
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

TOOLTIPS = {
    "lora_name": "Output filename for your LoRA checkpoints.",
    "max_train_steps": "Total optimizer steps. If above 0, this overrides epochs.",
    "epochs": "Full passes through the dataset. Used only when max steps is 0.",
    "learning_rate": "How fast the LoRA learns. Lower is safer; higher learns faster but can overfit.",
    "network_dim": "LoRA rank/capacity. Higher can learn more style detail but may overfit.",
    "network_alpha": "LoRA scaling alpha. Usually match rank/dim for predictable strength.",
    "attention_mode": "Attention backend. SDPA is most compatible; Sage/xformers can be faster but may change outputs.",
    "blocks_to_swap": "Number of model blocks moved to CPU to save VRAM. Higher uses less VRAM but is slower.",
    "save_every_steps": "How often to save a checkpoint during training.",
    "sample_every_steps": "How often to generate training samples during training.",
    "seed": "Training random seed for reproducibility.",
    "resolution_bucket": "Square max bucket size. Musubi auto-creates aspect-ratio buckets under this pixel area.",
    "optimizer_type": "Optimizer algorithm used to update LoRA weights.",
    "mixed_precision": "Training precision. bf16 is usually best on modern NVIDIA GPUs.",
    "save_precision": "Precision used when saving LoRA checkpoints.",
    "batch_size": "Images per training batch. Higher uses more VRAM.",
    "num_repeats": "Repeats each dataset item per epoch.",
    "block_swap_ring_size": "GPU ring buffers for block swapping. 2 is a balanced default.",
    "timestep_sampling": "How diffusion timesteps are sampled during training. Krea2 usually uses krea2_shift.",
    "discrete_flow_shift": "Flow shift value for shift-based timestep sampling.",
    "weighting_scheme": "Loss weighting method across timesteps.",
    "network_dropout": "Randomly drops LoRA paths during training. Blank disables it.",
    "max_data_loader_n_workers": "CPU workers for loading data. Higher may load faster but use more RAM.",
    "test_lora_multiplier": "Default LoRA strength used for sample generation.",
    "gradient_accumulation_steps": "Accumulates gradients over multiple steps before updating weights.",
    "max_grad_norm": "Gradient clipping limit. Helps prevent unstable updates.",
    "lr_scheduler": "Learning-rate schedule over training.",
    "lr_warmup_steps": "Steps spent ramping up learning rate from zero.",
    "lr_decay_steps": "Decay duration for supported LR schedulers.",
    "lr_scheduler_num_cycles": "Number of cycles for cosine/restart schedulers.",
    "lr_scheduler_power": "Power value for polynomial scheduler.",
    "lr_scheduler_min_lr_ratio": "Minimum LR ratio for schedulers that support it. Blank disables.",
    "optimizer_args": "Extra optimizer arguments separated by spaces, e.g. weight_decay=0.01.",
    "lr_scheduler_args": "Extra scheduler arguments separated by spaces.",
    "sigmoid_scale": "Scale for sigmoid timestep sampling.",
    "logit_mean": "Mean for logit-normal timestep weighting/sampling.",
    "logit_std": "Std dev for logit-normal timestep weighting/sampling.",
    "mode_scale": "Scale for mode weighting scheme.",
    "min_timestep": "Minimum training timestep. Blank uses Musubi default.",
    "max_timestep": "Maximum training timestep. Blank uses Musubi default.",
    "num_timestep_buckets": "Uniform timestep bucketing. 0 disables it.",
    "scale_weight_norms": "Scales LoRA weight norms. Blank disables.",
    "save_last_n_steps": "Keep only last N step checkpoints. 0 disables pruning.",
    "save_last_n_epochs": "Keep only last N epoch checkpoints. 0 disables pruning.",
    "save_last_n_steps_state": "Keep only last N saved training states by steps.",
    "save_last_n_epochs_state": "Keep only last N saved training states by epochs.",
    "training_comment": "Comment embedded in checkpoint metadata.",
    "metadata_title": "Title metadata embedded in checkpoint.",
    "metadata_author": "Author metadata embedded in checkpoint.",
    "metadata_description": "Description metadata embedded in checkpoint.",
    "metadata_license": "License metadata embedded in checkpoint.",
    "metadata_tags": "Comma-separated tags embedded in checkpoint metadata.",
    "log_prefix": "Prefix for log folders.",
    "log_tracker_name": "Tracker name for TensorBoard/WandB logging.",
    "compile": "Enable torch.compile. Can improve speed but may be unstable or slow to start.",
    "compile_backend": "torch.compile backend, usually inductor.",
    "compile_mode": "torch.compile optimization mode.",
    "compile_dynamic": "Dynamic shape behavior for torch.compile.",
    "compile_fullgraph": "Require torch.compile full graph mode.",
    "fp8_base": "Use FP8 base model mode to reduce VRAM.",
    "fp8_scaled": "Use scaled FP8 where supported. Usually good for Krea2.",
    "gradient_checkpointing": "Saves VRAM by recomputing activations; slower but safer.",
    "gradient_checkpointing_cpu_offload": "Offload checkpointed activations to CPU to save more VRAM.",
    "use_pinned_memory_for_block_swap": "Use pinned CPU memory for faster block swaps.",
    "block_swap_h2d_only": "Host-to-device-only swapping for frozen base weights.",
    "persistent_data_loader_workers": "Keep dataloader workers alive between epochs.",
    "enable_training_samples": "Generate sample images during training.",
    "enable_tensorboard": "Write TensorBoard logs.",
    "bucket_no_upscale": "Do not upscale images smaller than the bucket area.",
    "sample_at_first": "Generate a sample at the beginning of training.",
    "preserve_distribution_shape": "Preserve timestep distribution shape when restricting timesteps.",
    "log_config": "Log training configuration.",
    "no_metadata": "Do not write metadata into saved checkpoints.",
    "dim_from_weights": "Infer LoRA dim from loaded weights when resuming.",
    "save_state": "Save full training state for exact resume.",
    "save_state_on_train_end": "Save full training state when training finishes.",
    "cuda_allow_tf32": "Allow TF32 math on supported NVIDIA GPUs for speed.",
    "cuda_cudnn_benchmark": "Enable cuDNN benchmark for potential speedups.",
    "img_in_txt_in_offloading": "Enable additional image/text input offloading.",
    "disable_numpy_memmap": "Disable numpy memory mapping.",
    "guidance_scale": "Training guidance scale if supported by the selected workflow. Blank uses Musubi default.",
    "vae_dtype": "Optional VAE dtype override. Blank uses Musubi default.",
    "wan_task": "Wan model task variant, e.g. text-to-video, image-to-video, or text-to-image.",
    "timestep_boundary": "Wan-specific timestep boundary. Blank uses Musubi default.",
    "fp8_llm": "Use FP8 for the Z-Image text encoder/LLM.",
    "fp8_vl": "Use FP8 for the Qwen Image VL/text encoder.",
    "fp8_t5": "Use FP8 for the Wan T5 text encoder.",
    "fp8_text_encoder": "Use FP8 for the Flux text encoder.",
    "full_bf16": "Use full bf16 mode where supported.",
    "use_32bit_attention": "Use 32-bit attention for supported workflows.",
    "split_attn": "Split attention to reduce memory usage.",
    "sample_lora": "LoRA checkpoint to use for sample generation.",
    "sample_prompt": "Prompt used to generate a validation image.",
    "test_negative_prompt": "Negative prompt. Only affects generation when guidance is above 1.",
    "test_width": "Sample image width.",
    "test_height": "Sample image height.",
    "test_steps": "Sample denoising steps.",
    "test_guidance_scale": "CFG guidance. 1 disables CFG for Krea2.",
    "test_seed_mode": "Fixed uses the seed number for repeatable samples. Random omits --seed so Musubi chooses a random seed.",
    "test_seed": "Sample seed used when seed mode is Fixed.",
    "test_num_images": "Number of sample images to generate.",
    "test_mu": "Krea2 timestep-shift mu for sampling.",
    "test_attn_mode": "Attention backend used during sample generation.",
}


class ToolTip:
    def __init__(self, widget, text, delay=550):
        self.widget = widget
        self.text = text
        self.delay = delay
        self.after_id = None
        self.tip = None
        widget.bind("<Enter>", self.schedule, add="+")
        widget.bind("<Leave>", self.hide, add="+")
        widget.bind("<ButtonPress>", self.hide, add="+")

    def schedule(self, _event=None):
        self.cancel()
        self.after_id = self.widget.after(self.delay, self.show)

    def cancel(self):
        if self.after_id:
            self.widget.after_cancel(self.after_id)
            self.after_id = None

    def show(self):
        if self.tip or not self.text:
            return
        x = self.widget.winfo_pointerx() + 14
        y = self.widget.winfo_pointery() + 16
        self.tip = tw = tk.Toplevel(self.widget)
        tw.wm_overrideredirect(True)
        tw.wm_geometry(f"+{x}+{y}")
        label = tk.Label(tw, text=self.text, justify="left", wraplength=360, bg="#0f172a", fg="#e5e7eb", relief="solid", bd=1, padx=8, pady=6, font=("Segoe UI", 9))
        label.pack()

    def hide(self, _event=None):
        self.cancel()
        if self.tip:
            self.tip.destroy()
            self.tip = None


def add_tooltip(widget, key_or_text):
    text = TOOLTIPS.get(key_or_text, key_or_text)
    if text:
        ToolTip(widget, text)

LR_CHOICES = [
    "0.00001 (1e-5)",
    "0.00002 (2e-5)",
    "0.00005 (5e-5)",
    "0.0001 (1e-4)",
    "0.0002 (2e-4)",
    "0.0003 (3e-4)",
]
RESOLUTION_BUCKET_CHOICES = [
    ("512x512", "512"),
    ("768x768", "768"),
    ("1024x1024", "1024"),
    ("1280x1280", "1280"),
    ("1328x1328", "1328"),
    ("1536x1536", "1536"),
    ("2048x2048", "2048"),
]

WORKFLOW_SPECIFIC_UI = {
    "krea2": {
        "fields": [("guidance_scale", "Train guidance", "entry", None), ("vae_dtype", "VAE dtype", "combo", ["", "bf16", "fp16", "fp32"])],
        "bools": [],
    },
    "zimage": {
        "fields": [("guidance_scale", "Train guidance", "entry", None), ("vae_dtype", "VAE dtype", "combo", ["", "bf16", "fp16", "fp32"])],
        "bools": [("fp8_llm", "FP8 LLM"), ("use_32bit_attention", "32-bit attention"), ("full_bf16", "Full bf16"), ("split_attn", "Split attention")],
    },
    "qwen_image": {
        "fields": [("guidance_scale", "Train guidance", "entry", None), ("vae_dtype", "VAE dtype", "combo", ["", "bf16", "fp16", "fp32"])],
        "bools": [("fp8_vl", "FP8 VL/text encoder"), ("split_attn", "Split attention")],
    },
    "wan": {
        "fields": [("guidance_scale", "Train guidance", "entry", None), ("vae_dtype", "VAE dtype", "combo", ["", "bf16", "fp16", "fp32"]), ("wan_task", "Wan task", "combo", ["t2v-14B", "t2v-1.3B", "i2v-14B", "t2i-14B", "flf2v-14B", "t2v-1.3B-FC", "t2v-14B-FC", "i2v-14B-FC", "i2v-A14B", "t2v-A14B"]), ("timestep_boundary", "Timestep boundary", "entry", None)],
        "bools": [("fp8_t5", "FP8 T5"), ("split_attn", "Split attention")],
    },
    "flux_2": {
        "fields": [("guidance_scale", "Train guidance", "entry", None), ("vae_dtype", "VAE dtype", "combo", ["", "bf16", "fp16", "fp32"])],
        "bools": [("fp8_text_encoder", "FP8 text encoder"), ("split_attn", "Split attention")],
    },
    "flux_kontext": {
        "fields": [("guidance_scale", "Train guidance", "entry", None), ("vae_dtype", "VAE dtype", "combo", ["", "bf16", "fp16", "fp32"])],
        "bools": [("split_attn", "Split attention")],
    },
    "default": {
        "fields": [("guidance_scale", "Train guidance", "entry", None), ("vae_dtype", "VAE dtype", "combo", ["", "bf16", "fp16", "fp32"])],
        "bools": [("split_attn", "Split attention")],
    },
}


def lr_display(value):
    value = str(value).strip().lower()
    return {
        "1e-5": "0.00001 (1e-5)", "0.00001": "0.00001 (1e-5)",
        "2e-5": "0.00002 (2e-5)", "0.00002": "0.00002 (2e-5)",
        "5e-5": "0.00005 (5e-5)", "0.00005": "0.00005 (5e-5)",
        "1e-4": "0.0001 (1e-4)", "0.0001": "0.0001 (1e-4)",
        "2e-4": "0.0002 (2e-4)", "0.0002": "0.0002 (2e-4)",
        "3e-4": "0.0003 (3e-4)", "0.0003": "0.0003 (3e-4)",
    }.get(value, str(value))


def lr_value(value):
    value = str(value).strip()
    if "(" in value:
        return value.split("(", 1)[0].strip()
    return value


def selected_resolutions(s):
    """Return the single official Musubi max bucket resolution.

    Musubi's upstream dataset config supports one `resolution` plus
    `enable_bucket=true`; it then auto-generates aspect-ratio buckets under
    that target pixel area. It does not support AI-Toolkit-style explicit
    multiple bucket checkboxes without patching Musubi itself.
    """
    values = s.get("train_resolutions") or []
    if isinstance(values, str):
        values = [values]
    valid = {key for key, _label in RESOLUTION_BUCKET_CHOICES}
    for value in values:
        if value in valid:
            return [value]
    w = s.get("resolution_width", 1024)
    h = s.get("resolution_height", 1024)
    return [f"{w}x{h}"]


def write_dataset_config(s):
    dataset.parent.mkdir(parents=True, exist_ok=True)
    train_posix = Path(s["train_images_dir"]).as_posix()
    cache_dir = Path(s["cache_dir"])
    cache_dir.mkdir(parents=True, exist_ok=True)
    res = selected_resolutions(s)[0]
    w, h = [int(x) for x in res.split("x", 1)]
    # Official Musubi usage: one max resolution + enable_bucket.
    # Musubi then auto-generates aspect-ratio buckets under this pixel area.
    dataset.write_text(
        f'''[general]
resolution = [{w}, {h}]
caption_extension = ".txt"
batch_size = {int(s.get('batch_size', 1))}
enable_bucket = true
bucket_no_upscale = {str(bool(s.get('bucket_no_upscale', True))).lower()}

[[datasets]]
image_directory = "{train_posix}"
cache_directory = "{cache_dir.as_posix()}"
num_repeats = {int(s.get('num_repeats', 1))}
''',
        encoding="utf-8",
    )


def _fix_dataset_paths(s):
    write_dataset_config(s)


def load_settings():
    s = DEFAULTS.copy()
    if settings_path.exists():
        s.update(json.loads(settings_path.read_text(encoding="utf-8")))
    _fix_dataset_paths(s)
    return s


def spath(key):
    return Path(load_settings()[key])


def current_workflow():
    return get_workflow(load_settings().get("workflow", "krea2"))


def rel_script(wf, script_key):
    return root_dir / wf["scripts"][script_key]


def add_model_args(cmd, wf, arg_map, settings):
    for arg, setting_key in wf.get(arg_map, {}).items():
        value = str(settings.get(setting_key, "")).strip()
        if value:
            cmd += [arg, Path(value)]


def add_flag(cmd, enabled, flag):
    if enabled:
        cmd.append(flag)


class Launcher(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Musubi Universal Trainer GUI")
        self.geometry("1600x920")
        self.minsize(1360, 820)
        self.after(50, self.start_fullscreen)
        self.apply_theme()
        self.proc = None
        self.tb_proc = None
        self.settings = load_settings()
        self.last_benchmark_start = None
        self.log_handle = None
        self.sample_image_ref = None
        self.metric_vars = {}
        self.workflow_setting_vars = {}
        self.workflow_bool_vars = {}
        self.resolution_bucket_var = None
        self._autosave_after = None
        self._updating_inline_settings = False

        self._build_header()

        self.status = tk.StringVar()
        self.train_stats = tk.StringVar(value="Training stats: idle")

        main = ttk.PanedWindow(self, orient="horizontal")
        main.pack(fill="both", expand=True, padx=10, pady=(0, 8))

        left = ttk.Frame(main)
        right = ttk.LabelFrame(main, text="Live Log", style="Card.TLabelframe")
        main.add(left, weight=2)
        main.add(right, weight=3)
        # Start with Workflow and Live Log at an even split.
        # Do this on first real layout pass; doing it immediately is too early on Windows.
        self._initial_sash_done = False
        def set_initial_sash(_event=None):
            if self._initial_sash_done:
                return
            width = main.winfo_width()
            if width > 800:
                main.sashpos(0, int(width * 0.50))
                self._initial_sash_done = True
        main.bind("<Configure>", set_initial_sash)

        self.nb = ttk.Notebook(left)
        self.nb.pack(fill="both", expand=True)
        self.workflow_tab = ttk.Frame(self.nb)
        self.dataset_tab = ttk.Frame(self.nb)
        self.models_tab = ttk.Frame(self.nb)
        self.samples_tab = ttk.Frame(self.nb)
        self.diagnostics_tab = ttk.Frame(self.nb)
        self.nb.add(self.workflow_tab, text="Workflow")
        self.nb.add(self.dataset_tab, text="Dataset / Files")
        self.nb.add(self.models_tab, text="Models")
        self.nb.add(self.samples_tab, text="Samples")
        self.nb.add(self.diagnostics_tab, text="Diagnostics")

        self._build_workflow_tab(self.workflow_tab)
        self._build_dataset_tab(self.dataset_tab)
        self._build_models_tab(self.models_tab)
        self._build_samples_tab(self.samples_tab)
        self._build_diagnostics_tab(self.diagnostics_tab)
        self.log = scrolledtext.ScrolledText(right, wrap="word", font=("Cascadia Mono", 9), bg="#0b1020", fg="#d6deff", insertbackground="#f8fafc", relief="flat", borderwidth=0)
        self.log.pack(fill="both", expand=True, padx=8, pady=8)
        self.setup_log_colors()
        ttk.Button(right, text="Clear Log", command=lambda: self.log.delete("1.0", "end")).pack(anchor="e", padx=8, pady=(0, 8))
        self.refresh_settings()
        self.validate_dataset()
        self.refresh_sample_preview()

    def start_fullscreen(self):
        try:
            self.state("zoomed")
        except tk.TclError:
            self.attributes("-zoomed", True)

    def apply_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        bg = "#0b1220"       # app background
        surface = "#111827"  # card/panel
        surface2 = "#172033" # buttons/inputs
        border = "#243044"
        fg = "#e5e7eb"
        muted = "#9ca3af"
        accent = "#60a5fa"
        accent_dark = "#2563eb"
        danger = "#ef4444"
        success = "#22c55e"

        self.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg, fieldbackground=surface, font=("Segoe UI", 10), bordercolor=border, lightcolor=border, darkcolor=border)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("Title.TLabel", background=bg, foreground="#f8fafc", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", background=bg, foreground=muted, font=("Segoe UI", 10))
        style.configure("Status.TLabel", background=bg, foreground=muted, font=("Segoe UI", 9))
        style.configure("Stats.TLabel", background=bg, foreground=success, font=("Segoe UI", 10, "bold"))
        style.configure("MetricTitle.TLabel", background=surface, foreground=muted, font=("Segoe UI", 9, "bold"))
        style.configure("MetricValue.TLabel", background=surface, foreground="#f8fafc", font=("Segoe UI", 15, "bold"))
        style.configure("MetricHint.TLabel", background=surface, foreground=muted, font=("Segoe UI", 8))

        style.configure("TLabelframe", background=bg, foreground=fg, bordercolor=border, relief="solid")
        style.configure("Card.TLabelframe", background=bg, foreground=fg, bordercolor=border, relief="solid")
        style.configure("TLabelframe.Label", background=bg, foreground=fg, font=("Segoe UI", 10, "bold"))

        style.configure("TButton", background=surface2, foreground=fg, padding=(11, 6), borderwidth=0, focusthickness=0)
        style.map("TButton", background=[("active", "#23324a"), ("pressed", "#1e293b")], foreground=[("active", "#ffffff")])
        style.configure("Accent.TButton", background=accent_dark, foreground="#ffffff", padding=(14, 7), font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", accent), ("pressed", "#1d4ed8")])
        style.configure("Danger.TButton", background="#7f1d1d", foreground="#fecaca", padding=(11, 6))
        style.map("Danger.TButton", background=[("active", danger), ("pressed", "#991b1b")], foreground=[("active", "#ffffff")])

        style.configure("TNotebook", background=bg, borderwidth=0, tabmargins=(0, 0, 0, 0))
        style.configure(
            "TNotebook.Tab",
            background="#111827",
            foreground="#aab4c3",
            padding=(18, 7),
            borderwidth=0,
            relief="flat",
            font=("Segoe UI", 10, "bold"),
        )
        style.map(
            "TNotebook.Tab",
            background=[("selected", "#2563eb"), ("active", "#1d4ed8")],
            foreground=[("selected", "#ffffff"), ("active", "#ffffff")],
        )

        style.configure("TCombobox", fieldbackground=surface, background=surface2, foreground=fg, arrowcolor=fg, bordercolor=border, padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", surface)], foreground=[("readonly", fg)], selectbackground=[("readonly", surface)], selectforeground=[("readonly", fg)])
        style.configure("TEntry", fieldbackground=surface, foreground=fg, insertcolor=fg, bordercolor=border, padding=4)
        style.configure("Horizontal.TPanedwindow", background=bg)

    def _build_header(self):
        header = ttk.Frame(self)
        header.pack(fill="x", padx=18, pady=(10, 4))

        brand = ttk.Frame(header)
        brand.pack(side="left", fill="x", expand=True)
        ttk.Label(brand, text="Musubi-Tuner", style="Title.TLabel").pack(anchor="w")

        controls = ttk.Frame(header)
        controls.pack(side="right")
        ttk.Label(controls, text="Workflow").grid(row=0, column=0, sticky="w", padx=(0, 6))
        self.workflow_var = tk.StringVar(value=self.settings.get("workflow", "krea2"))
        self.workflow_combo = ttk.Combobox(controls, textvariable=self.workflow_var, width=22, state="readonly")
        self.workflow_combo["values"] = list(enabled_workflows().keys())
        self.workflow_combo.grid(row=1, column=0, sticky="e", padx=(0, 10))
        self.workflow_combo.bind("<<ComboboxSelected>>", lambda _e: self.change_workflow())
        ttk.Button(controls, text="START TRAINING", command=self.train, style="Accent.TButton").grid(row=1, column=1, padx=(0, 8))
        ttk.Button(controls, text="Benchmark", command=self.benchmark).grid(row=1, column=2, padx=4)
        ttk.Button(controls, text="Refresh", command=self.refresh_settings).grid(row=1, column=3, padx=4)
        ttk.Button(controls, text="Stop", command=self.stop, style="Danger.TButton").grid(row=1, column=4, padx=(4, 0))

    def _metric_card(self, parent, key, title, hint=""):
        card = ttk.LabelFrame(parent, text="", style="Card.TLabelframe")
        card.pack(side="left", fill="x", expand=True, padx=6)
        ttk.Label(card, text=title.upper(), style="MetricTitle.TLabel").pack(anchor="w", padx=12, pady=(10, 0))
        var = tk.StringVar(value="—")
        self.metric_vars[key] = var
        ttk.Label(card, textvariable=var, style="MetricValue.TLabel").pack(anchor="w", padx=12, pady=(2, 0))
        ttk.Label(card, text=hint, style="MetricHint.TLabel").pack(anchor="w", padx=12, pady=(0, 10))

    def _build_metrics_row(self):
        metrics = ttk.Frame(self)
        metrics.pack(fill="x", padx=12, pady=(8, 4))
        self._metric_card(metrics, "images", "Dataset", "top-level images")
        self._metric_card(metrics, "captions", "Captions", "matching .txt files")
        self._metric_card(metrics, "cache", "Cache", "latent / text cache")
        self._metric_card(metrics, "models", "Models", "required model paths")
        self._metric_card(metrics, "plan", "Train Plan", "steps or epochs")

    def _build_workflow_tab(self, parent):
        s = load_settings()
        wf = get_workflow(s.get("workflow", "krea2"))
        ttk.Label(parent, text=f"{wf['label']} workflow", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(8, 1))
        note = "Experimental template — verify model paths and command preview before training." if wf.get("experimental") else "Run from top to bottom: setup models, cache latents, cache text, then train."
        ttk.Label(parent, text=note, style="Status.TLabel").pack(anchor="w", padx=12, pady=(0, 6))

        resume_frame = ttk.LabelFrame(parent, text="Resume / continue from LoRA weights")
        resume_frame.pack(fill="x", padx=12, pady=(4, 8))
        self.resume_var = tk.StringVar(value="Start fresh")
        self.resume_combo = ttk.Combobox(resume_frame, textvariable=self.resume_var, width=52, state="readonly")
        self.resume_combo.pack(side="left", padx=8, pady=6)
        ttk.Button(resume_frame, text="Reload", command=self.refresh_checkpoints).pack(side="left", padx=6)
        ttk.Label(resume_frame, text="Loads LoRA weights only.", style="Status.TLabel").pack(side="left", padx=6)

        grid = ttk.Frame(parent)
        grid.pack(anchor="nw", padx=12, pady=8)
        setup_text = "1. Download Krea2 Models" if s.get("workflow", "krea2") == "krea2" else "1. Model Setup"
        setup_cmd = self.download if s.get("workflow", "krea2") == "krea2" else self.workflow_model_setup
        setup_desc = "Downloads the standard Krea2 model files." if s.get("workflow", "krea2") == "krea2" else "Open/select model paths required by this workflow."
        actions = [(setup_text, setup_cmd, setup_desc)]
        if "cache_latents" in wf.get("scripts", {}):
            actions.append(("2. Cache Latents", self.cache_latents, f"Runs {Path(wf['scripts']['cache_latents']).name}."))
        else:
            actions.append(("2. Cache Latents", lambda: messagebox.showinfo("Not available", f"{wf['label']} has no cache latents step configured."), "Not available for this workflow."))
        if "cache_text" in wf.get("scripts", {}):
            actions.append(("3. Cache Text", self.cache_text, f"Runs {Path(wf['scripts']['cache_text']).name}."))
        else:
            actions.append(("3. Cache Text", lambda: messagebox.showinfo("Not available", f"{wf['label']} has no cache text step configured."), "Not available for this workflow."))
        actions += [
            ("4. Start Training", self.train, f"Runs {Path(wf['scripts']['train']).name} with {wf['network_module']}."),
            ("Preview Launch Command", self.show_launch_command, "Shows the exact training command before running it."),
        ]
        for r, (text, cmd, desc) in enumerate(actions):
            ttk.Button(grid, text=text, command=cmd, width=24).grid(row=r, column=0, padx=6, pady=6, sticky="w")
            ttk.Label(grid, text=desc).grid(row=r, column=1, padx=6, pady=6, sticky="w")

        self._build_inline_training_settings(parent)

        self.refresh_checkpoints()

    def _build_dataset_tab(self, parent):
        ttk.Label(parent, text="Dataset and output helpers", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 6))
        grid = ttk.Frame(parent)
        grid.pack(anchor="nw", padx=12, pady=8)
        items = [
            ("Choose Training Folder", self.choose_train_folder, "Select any existing folder as your dataset."),
            ("Open Training Images", lambda: self.open_path(spath("train_images_dir")), "Put image + matching .txt caption files here."),
            ("Open Cache", lambda: self.open_path(spath("cache_dir")), "Cached latents/text outputs are stored here."),
            ("Open Output", lambda: self.open_path(project / "output"), "LoRA checkpoints and final LoRA save here."),
            ("Open Training Samples", lambda: self.open_path(project / "output" / "sample"), "Samples generated during training."),
            ("Open Test Samples", lambda: self.open_path(project / "samples"), "Images from the Test Image button."),
            ("Edit Sample Prompt", lambda: subprocess.Popen(["notepad", str(sample_prompts)]), "Prompt used for samples during training."),
            ("Open Dataset Config", lambda: subprocess.Popen(["notepad", str(dataset)]), "TOML dataset config generated by settings."),
        ]
        for r, (text, cmd, desc) in enumerate(items):
            ttk.Button(grid, text=text, command=cmd, width=24).grid(row=r, column=0, padx=6, pady=6, sticky="w")
            ttk.Label(grid, text=desc).grid(row=r, column=1, padx=6, pady=6, sticky="w")

        tools = ttk.LabelFrame(parent, text="Dataset validation and cache tools")
        tools.pack(fill="x", padx=12, pady=12)
        ttk.Button(tools, text="Validate Dataset", command=self.validate_dataset).pack(side="left", padx=6, pady=8)
        ttk.Button(tools, text="Clear Latent Cache", command=lambda: self.clear_cache("latent")).pack(side="left", padx=6)
        ttk.Button(tools, text="Clear Text Cache", command=lambda: self.clear_cache("text")).pack(side="left", padx=6)
        ttk.Button(tools, text="Clear All Cache", command=lambda: self.clear_cache("all")).pack(side="left", padx=6)
        self.dataset_status = tk.StringVar(value="Dataset not checked yet.")
        self.dataset_status_label = ttk.Label(parent, textvariable=self.dataset_status, justify="left", wraplength=720)
        self.dataset_status_label.pack(fill="x", anchor="w", padx=18, pady=8)
        self.dataset_status_label.bind("<Configure>", lambda e: self.dataset_status_label.configure(wraplength=max(240, e.width - 12)))

    def _build_models_tab(self, parent):
        ttk.Label(parent, text="Model paths for the selected workflow", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 6))
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=12, pady=8)
        ttk.Button(row, text="Refresh", command=self.refresh_model_manager).pack(side="left", padx=6)
        ttk.Button(row, text="Open Default Models Folder", command=lambda: self.open_path(project / "models")).pack(side="left", padx=6)
        self.models_frame = ttk.Frame(parent)
        self.models_frame.pack(fill="both", expand=True, padx=12, pady=8)

    def refresh_model_manager(self):
        if not hasattr(self, "models_frame"):
            return
        for child in self.models_frame.winfo_children():
            child.destroy()
        s = load_settings()
        wf = get_workflow(s.get("workflow", "krea2"))
        for row, (key, label) in enumerate(wf.get("model_fields", {}).items()):
            value = str(s.get(key, ""))
            exists = bool(value) and Path(value).exists()
            ttk.Label(self.models_frame, text=label, width=24).grid(row=row, column=0, sticky="e", padx=6, pady=5)
            ttk.Label(self.models_frame, text="OK" if exists else "MISSING", foreground="#22c55e" if exists else "#f97316", width=10).grid(row=row, column=1, sticky="w", padx=6)
            var = tk.StringVar(value=value)
            ent = ttk.Entry(self.models_frame, textvariable=var, width=58)
            ent.grid(row=row, column=2, sticky="we", padx=6)
            ttk.Button(self.models_frame, text="Browse", command=lambda k=key, v=var: self.browse_model_path(k, v)).grid(row=row, column=3, padx=4)
            ttk.Button(self.models_frame, text="Open", command=lambda v=value: self.open_model_location(v)).grid(row=row, column=4, padx=4)
        self.models_frame.columnconfigure(2, weight=1)

    def browse_model_path(self, key, var):
        current = var.get().strip()
        initial = str(Path(current).parent) if current else str(project / "models")
        chosen = filedialog.askopenfilename(initialdir=initial, filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")])
        if not chosen:
            return
        s = load_settings()
        s[key] = chosen
        settings_path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        self.refresh_settings()
        self.validate_dataset()
        self.refresh_model_manager()

    def open_model_location(self, value):
        if value and Path(value).exists():
            subprocess.Popen(["explorer", "/select,", str(Path(value))])
        else:
            self.open_path(project / "models")

    def _build_samples_tab(self, parent):
        ttk.Label(parent, text="Test image generation", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 2))
        ttk.Label(parent, text="Generate quick validation images from the latest selected LoRA checkpoint.", style="Status.TLabel").pack(anchor="w", padx=12, pady=(0, 8))

        body = ttk.PanedWindow(parent, orient="horizontal")
        body.pack(fill="both", expand=True, padx=12, pady=8)
        sidebar = ttk.LabelFrame(body, text="Sample settings", style="Card.TLabelframe")
        preview = ttk.LabelFrame(body, text="Preview", style="Card.TLabelframe")
        body.add(sidebar, weight=0)
        body.add(preview, weight=1)

        self.test_lora_multiplier_var = tk.StringVar(value=str(load_settings().get("test_lora_multiplier", "1.0")))
        s0 = load_settings()
        self.test_width_var = tk.StringVar(value=str(s0.get("test_width", 1024)))
        self.test_height_var = tk.StringVar(value=str(s0.get("test_height", 1024)))
        self.test_steps_var = tk.StringVar(value=str(s0.get("test_steps", 8)))
        self.test_guidance_var = tk.StringVar(value=str(s0.get("test_guidance_scale", "1")))
        self.test_seed_mode_var = tk.StringVar(value=str(s0.get("test_seed_mode", "fixed")))
        self.test_seed_var = tk.StringVar(value=str(s0.get("test_seed", 1)))
        self.test_num_images_var = tk.StringVar(value=str(s0.get("test_num_images", 1)))
        self.test_mu_var = tk.StringVar(value=str(s0.get("test_mu", "1.15")))
        self.test_attn_mode_var = tk.StringVar(value=str(s0.get("test_attn_mode", "torch")))
        self.sample_lora_var = tk.StringVar(value="Latest for current LoRA name")

        lora_label = ttk.Label(sidebar, text="LoRA checkpoint")
        lora_label.pack(anchor="w", padx=10, pady=(10, 2))
        add_tooltip(lora_label, "sample_lora")
        self.sample_lora_combo = ttk.Combobox(sidebar, textvariable=self.sample_lora_var, width=44, state="readonly")
        self.sample_lora_combo.pack(fill="x", padx=10, pady=(0, 10))
        add_tooltip(self.sample_lora_combo, "sample_lora")

        prompt_label = ttk.Label(sidebar, text="Prompt")
        prompt_label.pack(anchor="w", padx=10, pady=(4, 2))
        add_tooltip(prompt_label, "sample_prompt")
        prompt_frame = ttk.Frame(sidebar)
        prompt_frame.pack(fill="x", padx=10, pady=(0, 10))
        self.test_prompt_text = tk.Text(prompt_frame, height=10, width=44, wrap="word", bg="#111827", fg="#e5e7eb", insertbackground="#f8fafc", relief="flat", padx=8, pady=8, font=("Segoe UI", 10))
        prompt_scroll = ttk.Scrollbar(prompt_frame, orient="vertical", command=self.test_prompt_text.yview)
        self.test_prompt_text.configure(yscrollcommand=prompt_scroll.set)
        self.test_prompt_text.insert("1.0", "a beautiful cinematic portrait photo")
        self.test_prompt_text.pack(side="left", fill="both", expand=True)
        prompt_scroll.pack(side="right", fill="y")
        add_tooltip(self.test_prompt_text, "sample_prompt")
        neg_label = ttk.Label(sidebar, text="Negative prompt")
        neg_label.pack(anchor="w", padx=10, pady=(4, 2))
        add_tooltip(neg_label, "test_negative_prompt")
        self.test_negative_text = tk.Text(sidebar, height=3, width=44, wrap="word", bg="#111827", fg="#e5e7eb", insertbackground="#f8fafc", relief="flat", padx=8, pady=6, font=("Segoe UI", 10))
        self.test_negative_text.insert("1.0", s0.get("test_negative_prompt", ""))
        self.test_negative_text.pack(fill="x", padx=10, pady=(0, 10))
        add_tooltip(self.test_negative_text, "test_negative_prompt")

        strength_label = ttk.Label(sidebar, text="LoRA strength")
        strength_label.pack(anchor="w", padx=10, pady=(4, 2))
        add_tooltip(strength_label, "test_lora_multiplier")
        strength_combo = ttk.Combobox(sidebar, textvariable=self.test_lora_multiplier_var, values=["0.4", "0.5", "0.6", "0.8", "1.0", "1.2", "1.5"], width=16, state="normal")
        strength_combo.pack(anchor="w", padx=10, pady=(0, 10))
        add_tooltip(strength_combo, "test_lora_multiplier")
        size_row = ttk.Frame(sidebar)
        size_row.pack(fill="x", padx=10, pady=(4, 10))
        width_label = ttk.Label(size_row, text="Width"); width_label.grid(row=0, column=0, sticky="w"); add_tooltip(width_label, "test_width")
        height_label = ttk.Label(size_row, text="Height"); height_label.grid(row=0, column=1, sticky="w", padx=(10, 0)); add_tooltip(height_label, "test_height")
        width_combo = ttk.Combobox(size_row, textvariable=self.test_width_var, values=["768", "1024", "1280", "1344", "1536", "1920"], width=10, state="normal")
        width_combo.grid(row=1, column=0, sticky="w"); add_tooltip(width_combo, "test_width")
        height_combo = ttk.Combobox(size_row, textvariable=self.test_height_var, values=["768", "1024", "1280", "1344", "1536", "1920"], width=10, state="normal")
        height_combo.grid(row=1, column=1, sticky="w", padx=(10, 0)); add_tooltip(height_combo, "test_height")
        sample_grid = ttk.Frame(sidebar)
        sample_grid.pack(fill="x", padx=10, pady=(0, 10))
        sample_fields = [
            ("Steps", "test_steps", self.test_steps_var, ["4", "8", "12", "16", "20"]),
            ("Guidance", "test_guidance_scale", self.test_guidance_var, ["1", "1.5", "2", "3", "4"]),
            ("Seed mode", "test_seed_mode", self.test_seed_mode_var, ["fixed", "random"]),
            ("Seed", "test_seed", self.test_seed_var, None),
            ("Images", "test_num_images", self.test_num_images_var, ["1", "2", "4"]),
            ("Mu", "test_mu", self.test_mu_var, ["1.0", "1.15", "1.3"]),
            ("Attention", "test_attn_mode", self.test_attn_mode_var, ["torch", "sageattn", "xformers", "flash"]),
        ]
        for i, (label, key, var, values) in enumerate(sample_fields):
            lab = ttk.Label(sample_grid, text=label)
            lab.grid(row=(i//2)*2, column=i%2, sticky="w", padx=(0 if i%2==0 else 10, 0)); add_tooltip(lab, key)
            if values:
                wid = ttk.Combobox(sample_grid, textvariable=var, values=values, width=12, state="normal")
            else:
                wid = ttk.Entry(sample_grid, textvariable=var, width=14)
            wid.grid(row=(i//2)*2+1, column=i%2, sticky="w", padx=(0 if i%2==0 else 10, 0), pady=(0, 6)); add_tooltip(wid, key)
        ttk.Button(sidebar, text="Generate Sample", command=self.test_image, style="Accent.TButton").pack(fill="x", padx=10, pady=(8, 6))
        ttk.Button(sidebar, text="Refresh Preview", command=self.refresh_sample_preview).pack(fill="x", padx=10, pady=4)
        ttk.Button(sidebar, text="Open Samples Folder", command=lambda: self.open_path(project / "samples")).pack(fill="x", padx=10, pady=4)
        ttk.Label(sidebar, text="Tip: lower strength if the style looks overcooked.", style="Status.TLabel", wraplength=300).pack(anchor="w", padx=10, pady=(18, 10))

        self.sample_status = tk.StringVar(value="No sample loaded yet.")
        ttk.Label(preview, textvariable=self.sample_status, style="Status.TLabel").pack(anchor="w", padx=12, pady=(10, 8))
        self.sample_label = ttk.Label(preview, text="Generated image preview will appear here.", anchor="center")
        self.sample_label.pack(fill="both", expand=True, padx=12, pady=8)

    def latest_sample_image(self):
        sample_dir = project / "samples"
        exts = {".png", ".jpg", ".jpeg", ".webp"}
        files = [p for p in sample_dir.glob("**/*") if p.is_file() and p.suffix.lower() in exts] if sample_dir.exists() else []
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    def refresh_sample_preview(self):
        if not hasattr(self, "sample_label"):
            return
        img_path = self.latest_sample_image()
        if not img_path:
            self.sample_status.set("No generated samples found yet.")
            return
        try:
            img = Image.open(img_path)
            img.thumbnail((620, 620))
            self.sample_image_ref = ImageTk.PhotoImage(img)
            self.sample_label.configure(image=self.sample_image_ref, text="")
            self.sample_status.set(f"Showing latest sample: {img_path}")
        except Exception as e:
            self.sample_status.set(f"Could not load sample preview: {e}")

    def selected_lora_for_test(self):
        sample_selected = self.sample_lora_var.get() if hasattr(self, "sample_lora_var") else "Latest for current LoRA name"
        if sample_selected and sample_selected != "Latest for current LoRA name" and Path(sample_selected).exists():
            return Path(sample_selected)
        selected = self.resume_var.get() if hasattr(self, "resume_var") else "Start fresh"
        if selected and selected != "Start fresh" and Path(selected).exists():
            return Path(selected)
        s = load_settings()
        output = project / "output"
        final = output / f"{s['lora_name']}.safetensors"
        # Prefer the newest checkpoint for the current LoRA name.  The old code preferred
        # output/<name>.safetensors whenever it existed, which could make the Test Image
        # button use a stale final from a previous run instead of the new step checkpoint.
        files = sorted(output.glob(f"{s['lora_name']}*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True) if output.exists() else []
        return files[0] if files else final

    def _build_diagnostics_tab(self, parent):
        ttk.Label(parent, text="Readiness checks", font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(12, 6))
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=12, pady=8)
        ttk.Button(row, text="Run Diagnostics", command=self.run_diagnostics).pack(side="left", padx=6)
        ttk.Button(row, text="Preview Train Command", command=self.preview_train_command).pack(side="left", padx=6)
        ttk.Button(row, text="Open Logs Folder", command=lambda: self.open_path(logs_dir)).pack(side="left", padx=6)
        ttk.Button(row, text="Export Debug Bundle", command=self.export_debug_bundle).pack(side="left", padx=6)
        self.diagnostics_text = tk.StringVar(value="Diagnostics not run yet.")
        ttk.Label(parent, textvariable=self.diagnostics_text, justify="left", wraplength=560).pack(anchor="nw", padx=18, pady=8)

    def _build_inline_training_settings(self, parent):
        box = ttk.LabelFrame(parent, text="Training settings — autosaves as you edit")
        box.pack(fill="x", padx=12, pady=(4, 12))
        self.inline_save_status = tk.StringVar(value="Settings are saved automatically.")

        fields = [
            ("lora_name", "LoRA name", "entry", None),
            ("max_train_steps", "Max steps", "entry", None),
            ("epochs", "Epochs", "entry", None),
            ("learning_rate", "Learning rate", "combo", LR_CHOICES),
            ("network_dim", "Rank / dim", "combo", ["8", "16", "32", "64"]),
            ("network_alpha", "Alpha", "combo", ["8", "16", "32", "64"]),
            ("attention_mode", "Attention", "combo", ["sdpa", "sage_attn", "xformers", "flash_attn", "flash3"]),
            ("blocks_to_swap", "Blocks swap", "combo", ["4", "8", "12", "14", "18"]),
            ("save_every_steps", "Save every", "entry", None),
            ("sample_every_steps", "Sample every", "entry", None),
            ("seed", "Seed", "entry", None),
        ]
        s = load_settings()
        for i, (key, label, kind, values) in enumerate(fields):
            row = i // 4
            col = (i % 4) * 2
            lab = ttk.Label(box, text=label)
            lab.grid(row=row, column=col, sticky="e", padx=(10, 6), pady=6)
            add_tooltip(lab, key)
            initial = lr_display(s.get(key, DEFAULTS.get(key, ""))) if key == "learning_rate" else str(s.get(key, DEFAULTS.get(key, "")))
            var = tk.StringVar(value=initial)
            self.workflow_setting_vars[key] = var
            if kind == "combo":
                widget = ttk.Combobox(box, textvariable=var, values=values, width=14, state="normal")
                widget.bind("<<ComboboxSelected>>", lambda _e, k=key: self.schedule_inline_autosave(k))
            else:
                widget = ttk.Entry(box, textvariable=var, width=16)
            widget.grid(row=row, column=col + 1, sticky="w", padx=(0, 12), pady=6)
            add_tooltip(widget, key)
            var.trace_add("write", lambda *_args, k=key: self.schedule_inline_autosave(k))
        buckets = ttk.LabelFrame(box, text="Max bucket resolution")
        buckets.grid(row=3, column=0, columnspan=8, sticky="we", padx=10, pady=(8, 6))
        selected = selected_resolutions(s)[0]
        self.resolution_bucket_var = tk.StringVar(value=selected)
        for i, (value, label) in enumerate(RESOLUTION_BUCKET_CHOICES):
            rb = tk.Radiobutton(
                buckets,
                text=label,
                value=value,
                variable=self.resolution_bucket_var,
                command=lambda r=value: self.select_resolution_bucket(r),
                indicatoron=False,
                width=8,
                relief="flat",
                bd=0,
                padx=10,
                pady=7,
                bg="#172033",
                fg="#e5e7eb",
                selectcolor="#2563eb",
                activebackground="#2563eb",
                activeforeground="#ffffff",
                font=("Segoe UI", 9, "bold"),
                cursor="hand2",
            )
            rb.grid(row=0, column=i, sticky="we", padx=4, pady=8)
            add_tooltip(rb, "resolution_bucket")
            buckets.columnconfigure(i, weight=1)
        ttk.Label(box, text="Pick one square max resolution. Musubi then creates aspect-ratio buckets under that pixel area automatically.", style="Status.TLabel", wraplength=980).grid(row=4, column=0, columnspan=8, sticky="w", padx=10, pady=(0, 4))
        advanced = ttk.LabelFrame(box, text="Advanced training settings")
        advanced.grid(row=5, column=0, columnspan=8, sticky="we", padx=10, pady=(8, 6))
        adv_nb = ttk.Notebook(advanced)
        adv_nb.grid(row=0, column=0, columnspan=8, sticky="we", padx=8, pady=8)
        adv_core = ttk.Frame(adv_nb)
        adv_sched = ttk.Frame(adv_nb)
        adv_meta = ttk.Frame(adv_nb)
        adv_flags = ttk.Frame(adv_nb)
        adv_workflow = ttk.Frame(adv_nb)
        adv_nb.add(adv_core, text="Core")
        adv_nb.add(adv_sched, text="Scheduler")
        adv_nb.add(adv_meta, text="Metadata")
        adv_nb.add(adv_flags, text="Toggles")
        selected_model_tab = get_workflow(s.get("workflow", "krea2"))["label"].replace(" LoRA", "")
        adv_nb.add(adv_workflow, text=selected_model_tab[:18])
        advanced_fields = [
            ("optimizer_type", "Optimizer", "combo", ["adamw8bit", "AdamW", "AdamW8bit", "Adafactor", "Lion", "SGDNesterov"]),
            ("mixed_precision", "Mixed precision", "combo", ["bf16", "fp16", "no"]),
            ("save_precision", "Save precision", "combo", ["float", "bf16", "fp16"]),
            ("batch_size", "Batch size", "combo", ["1", "2", "4"]),
            ("num_repeats", "Dataset repeats", "entry", None),
            ("block_swap_ring_size", "Swap ring", "combo", ["1", "2", "3", "4"]),
            ("timestep_sampling", "Timestep sampling", "combo", ["krea2_shift", "shift", "sigmoid", "uniform"]),
            ("discrete_flow_shift", "Flow shift", "entry", None),
            ("weighting_scheme", "Weighting", "combo", ["none", "sigma_sqrt", "logit_normal", "mode", "cosmap"]),
            ("network_dropout", "Dropout", "entry", None),
            ("max_data_loader_n_workers", "Workers", "combo", ["0", "1", "2", "4", "8"]),
            ("test_lora_multiplier", "Test strength", "entry", None),
            ("gradient_accumulation_steps", "Grad accum", "combo", ["1", "2", "4", "8"]),
            ("max_grad_norm", "Max grad norm", "entry", None),
            ("lr_scheduler", "LR scheduler", "combo", ["constant", "constant_with_warmup", "linear", "cosine", "cosine_with_restarts", "polynomial"]),
            ("lr_warmup_steps", "Warmup steps", "entry", None),
            ("lr_decay_steps", "Decay steps", "entry", None),
            ("lr_scheduler_num_cycles", "LR cycles", "entry", None),
            ("lr_scheduler_power", "LR power", "entry", None),
            ("lr_scheduler_min_lr_ratio", "Min LR ratio", "entry", None),
            ("optimizer_args", "Optimizer args", "entry", None),
            ("lr_scheduler_args", "Scheduler args", "entry", None),
            ("sigmoid_scale", "Sigmoid scale", "entry", None),
            ("logit_mean", "Logit mean", "entry", None),
            ("logit_std", "Logit std", "entry", None),
            ("mode_scale", "Mode scale", "entry", None),
            ("min_timestep", "Min timestep", "entry", None),
            ("max_timestep", "Max timestep", "entry", None),
            ("num_timestep_buckets", "Timestep buckets", "entry", None),
            ("scale_weight_norms", "Scale weight norms", "entry", None),
            ("save_last_n_steps", "Keep last steps", "entry", None),
            ("save_last_n_epochs", "Keep last epochs", "entry", None),
            ("save_last_n_steps_state", "Keep step states", "entry", None),
            ("save_last_n_epochs_state", "Keep epoch states", "entry", None),
            ("training_comment", "Training comment", "entry", None),
            ("metadata_title", "Meta title", "entry", None),
            ("metadata_author", "Meta author", "entry", None),
            ("metadata_description", "Meta description", "entry", None),
            ("metadata_license", "Meta license", "entry", None),
            ("metadata_tags", "Meta tags", "entry", None),
            ("log_prefix", "Log prefix", "entry", None),
            ("log_tracker_name", "Tracker name", "entry", None),
            ("compile_backend", "Compile backend", "entry", None),
            ("compile_mode", "Compile mode", "combo", ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"]),
            ("compile_dynamic", "Compile dynamic", "combo", ["auto", "true", "false"]),
        ]
        for i, (key, label, kind, values) in enumerate(advanced_fields):
            if i < 12:
                target = adv_core; j = i
            elif i < 26:
                target = adv_sched; j = i - 12
            else:
                target = adv_meta; j = i - 26
            row = j // 4
            col = (j % 4) * 2
            lab = ttk.Label(target, text=label)
            lab.grid(row=row, column=col, sticky="e", padx=(10, 6), pady=5)
            add_tooltip(lab, key)
            var = tk.StringVar(value=str(s.get(key, DEFAULTS.get(key, ""))))
            self.workflow_setting_vars[key] = var
            if kind == "combo":
                widget = ttk.Combobox(target, textvariable=var, values=values, width=14, state="normal")
                widget.bind("<<ComboboxSelected>>", lambda _e, k=key: self.schedule_inline_autosave(k))
            else:
                widget = ttk.Entry(target, textvariable=var, width=16)
            widget.grid(row=row, column=col + 1, sticky="w", padx=(0, 12), pady=5)
            add_tooltip(widget, key)
            var.trace_add("write", lambda *_args, k=key: self.schedule_inline_autosave(k))

        bools = adv_flags
        bool_fields = [
            ("fp8_base", "FP8 base"),
            ("fp8_scaled", "FP8 scaled"),
            ("gradient_checkpointing", "Gradient checkpointing"),
            ("use_pinned_memory_for_block_swap", "Pinned memory swap"),
            ("block_swap_h2d_only", "Block swap H2D only"),
            ("persistent_data_loader_workers", "Persistent workers"),
            ("enable_training_samples", "Training samples"),
            ("enable_tensorboard", "TensorBoard"),
            ("bucket_no_upscale", "Bucket no upscale"),
            ("gradient_checkpointing_cpu_offload", "Grad CPU offload"),
            ("sample_at_first", "Sample first"),
            ("preserve_distribution_shape", "Preserve timestep dist"),
            ("log_config", "Log config"),
            ("no_metadata", "No metadata"),
            ("dim_from_weights", "Dim from weights"),
            ("save_state", "Save state"),
            ("save_state_on_train_end", "Save state at end"),
            ("compile", "Torch compile"),
            ("compile_fullgraph", "Compile fullgraph"),
            ("cuda_allow_tf32", "Allow TF32"),
            ("cuda_cudnn_benchmark", "cuDNN benchmark"),
            ("img_in_txt_in_offloading", "Img/txt offload"),
            ("disable_numpy_memmap", "Disable memmap"),
        ]
        for i, (key, label) in enumerate(bool_fields):
            var = tk.BooleanVar(value=bool(s.get(key, DEFAULTS.get(key, False))))
            self.workflow_bool_vars[key] = var
            chk = tk.Checkbutton(
                bools, text=label, variable=var, command=self.schedule_inline_autosave,
                bg="#0b1220", fg="#e5e7eb", selectcolor="#111827", activebackground="#0b1220",
                activeforeground="#ffffff", anchor="w", padx=6, pady=2
            )
            chk.grid(row=i // 3, column=i % 3, sticky="w", padx=8, pady=4)
            add_tooltip(chk, key)

        wf_key = s.get("workflow", "krea2")
        spec = WORKFLOW_SPECIFIC_UI.get(wf_key, WORKFLOW_SPECIFIC_UI["default"])
        ttk.Label(adv_workflow, text=f"Options specific to {get_workflow(wf_key)['label']}", style="Status.TLabel").grid(row=0, column=0, columnspan=8, sticky="w", padx=10, pady=(8, 4))
        for i, (key, label, kind, values) in enumerate(spec.get("fields", [])):
            row = 1 + i // 4
            col = (i % 4) * 2
            lab = ttk.Label(adv_workflow, text=label)
            lab.grid(row=row, column=col, sticky="e", padx=(10, 6), pady=5)
            add_tooltip(lab, key)
            var = tk.StringVar(value=str(s.get(key, DEFAULTS.get(key, ""))))
            self.workflow_setting_vars[key] = var
            if kind == "combo":
                widget = ttk.Combobox(adv_workflow, textvariable=var, values=values, width=16, state="normal")
                widget.bind("<<ComboboxSelected>>", lambda _e, k=key: self.schedule_inline_autosave(k))
            else:
                widget = ttk.Entry(adv_workflow, textvariable=var, width=18)
            widget.grid(row=row, column=col + 1, sticky="w", padx=(0, 12), pady=5)
            add_tooltip(widget, key)
            var.trace_add("write", lambda *_args, k=key: self.schedule_inline_autosave(k))
        start_row = 3
        for i, (key, label) in enumerate(spec.get("bools", [])):
            var = tk.BooleanVar(value=bool(s.get(key, DEFAULTS.get(key, False))))
            self.workflow_bool_vars[key] = var
            chk = tk.Checkbutton(
                adv_workflow, text=label, variable=var, command=self.schedule_inline_autosave,
                bg="#0b1220", fg="#e5e7eb", selectcolor="#111827", activebackground="#0b1220",
                activeforeground="#ffffff", anchor="w", padx=6, pady=2
            )
            chk.grid(row=start_row + i // 3, column=i % 3, columnspan=2, sticky="w", padx=8, pady=4)
            add_tooltip(chk, key)

        ttk.Label(box, textvariable=self.inline_save_status, style="Status.TLabel").grid(row=10, column=0, columnspan=8, sticky="w", padx=10, pady=(2, 8))

    def select_resolution_bucket(self, selected):
        if getattr(self, "_updating_inline_settings", False):
            return
        if self.resolution_bucket_var is not None:
            self.resolution_bucket_var.set(selected)
        self.schedule_inline_autosave("train_resolutions")

    def schedule_inline_autosave(self, key=None):
        if getattr(self, "_updating_inline_settings", False):
            return
        if self._autosave_after:
            self.after_cancel(self._autosave_after)
        if hasattr(self, "inline_save_status"):
            self.inline_save_status.set("Saving…")
        self._autosave_after = self.after(500, self.commit_inline_settings)

    def commit_inline_settings(self):
        if not getattr(self, "workflow_setting_vars", None):
            return
        s = load_settings()
        int_keys = {"max_train_steps", "epochs", "network_dim", "network_alpha", "blocks_to_swap", "save_every_steps", "sample_every_steps", "seed", "batch_size", "num_repeats", "block_swap_ring_size", "max_data_loader_n_workers", "gradient_accumulation_steps", "lr_warmup_steps", "lr_decay_steps", "lr_scheduler_num_cycles", "num_timestep_buckets", "save_last_n_steps", "save_last_n_epochs", "save_last_n_steps_state", "save_last_n_epochs_state"}
        try:
            for key, var in self.workflow_setting_vars.items():
                value = var.get().strip()
                if key == "learning_rate":
                    value = lr_value(value)
                if key in int_keys:
                    if value == "":
                        continue
                    value = int(value)
                s[key] = value
            if getattr(self, "resolution_bucket_var", None) is not None:
                s["train_resolutions"] = [self.resolution_bucket_var.get() or "1024x1024"]
            for key, var in getattr(self, "workflow_bool_vars", {}).items():
                s[key] = bool(var.get())
            settings_path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
            _fix_dataset_paths(s)
            if hasattr(self, "inline_save_status"):
                self.inline_save_status.set(f"Saved {time.strftime('%H:%M:%S')}")
            self.refresh_settings()
        except Exception as e:
            if hasattr(self, "inline_save_status"):
                self.inline_save_status.set(f"Not saved: {e}")

    def apply_preset(self, name):
        s = load_settings()
        presets = {
            "style_balanced": {
                "max_train_steps": 3000, "epochs": 8, "learning_rate": "1e-4",
                "network_dim": 32, "network_alpha": 32, "attention_mode": "sdpa",
                "save_every_steps": 250, "sample_every_steps": 250, "enable_training_samples": True,
            },
            "style_conservative": {
                "max_train_steps": 2200, "epochs": 6, "learning_rate": "5e-5",
                "network_dim": 16, "network_alpha": 16, "attention_mode": "sdpa",
                "save_every_steps": 250, "sample_every_steps": 250, "enable_training_samples": True,
            },
            "low_vram": {
                "blocks_to_swap": 14, "block_swap_ring_size": 2, "gradient_checkpointing": True,
                "max_data_loader_n_workers": 1, "persistent_data_loader_workers": False,
                "attention_mode": "sdpa", "fp8_base": True, "fp8_scaled": True,
            },
        }
        s.update(presets[name])
        settings_path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        _fix_dataset_paths(s)
        self.refresh_settings()
        self.validate_dataset()
        self.append(f"\nApplied preset: {name.replace('_', ' ').title()}\n")

    def workflow_model_setup(self):
        self.refresh_model_manager()
        if hasattr(self, "nb") and hasattr(self, "models_tab"):
            self.nb.select(self.models_tab)
        self.append("\nModel setup: use the Models tab to point each required field to the correct model file or folder.\n")

    def rebuild_workflow_tab(self):
        if not hasattr(self, "workflow_tab"):
            return
        self.workflow_setting_vars = {}
        self.workflow_bool_vars = {}
        self.resolution_bucket_var = None
        for child in self.workflow_tab.winfo_children():
            child.destroy()
        self._build_workflow_tab(self.workflow_tab)

    def change_workflow(self):
        s = load_settings()
        s["workflow"] = self.workflow_var.get()
        settings_path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        self.rebuild_workflow_tab()
        self.refresh_settings()
        self.validate_dataset()

    def refresh_settings(self):
        self.settings = load_settings()
        s = self.settings
        wf = get_workflow(s.get("workflow", "krea2"))
        if hasattr(self, "workflow_combo"):
            self.workflow_combo["values"] = list(enabled_workflows().keys())
        if hasattr(self, "workflow_var"):
            self.workflow_var.set(s.get("workflow", "krea2"))
        if getattr(self, "workflow_setting_vars", None):
            self._updating_inline_settings = True
            try:
                for key, var in self.workflow_setting_vars.items():
                    value = lr_display(s.get(key, DEFAULTS.get(key, ""))) if key == "learning_rate" else str(s.get(key, DEFAULTS.get(key, "")))
                    var.set(value)
                if getattr(self, "resolution_bucket_var", None) is not None:
                    self.resolution_bucket_var.set(selected_resolutions(s)[0])
                for key, var in getattr(self, "workflow_bool_vars", {}).items():
                    var.set(bool(s.get(key, DEFAULTS.get(key, False))))
            finally:
                self._updating_inline_settings = False
        sample_note = "off" if not s.get("enable_training_samples") else str(s.get("sample_every_steps"))
        train_limit = f"Steps: {s.get('max_train_steps')}" if int(s.get("max_train_steps") or 0) > 0 else f"Epochs: {s['epochs']}"
        speed_hint = "Fast/OOM risk" if s["blocks_to_swap"] <= 4 else "Balanced" if s["blocks_to_swap"] <= 10 else "Safer/slower"
        self.status.set(
            f"Project: {project}\n"
            f"Workflow: {wf['label']} | LoRA: {s['lora_name']} | {train_limit} | LR: {s['learning_rate']} | "
            f"Dim/Alpha: {s['network_dim']}/{s['network_alpha']} | Blocks swap: {s['blocks_to_swap']} ({speed_hint}) | "
            f"Save: {s['save_every_steps']} | Sample: {sample_note} | TensorBoard: {s.get('enable_tensorboard', True)}"
        )
        if hasattr(self, "test_lora_multiplier_var"):
            self.test_lora_multiplier_var.set(str(s.get("test_lora_multiplier", "1.0")))
            self.test_width_var.set(str(s.get("test_width", 1024)))
            self.test_height_var.set(str(s.get("test_height", 1024)))
            self.test_steps_var.set(str(s.get("test_steps", 8)))
            self.test_guidance_var.set(str(s.get("test_guidance_scale", "1")))
            self.test_seed_mode_var.set(str(s.get("test_seed_mode", "fixed")))
            self.test_seed_var.set(str(s.get("test_seed", 1)))
            self.test_num_images_var.set(str(s.get("test_num_images", 1)))
            self.test_mu_var.set(str(s.get("test_mu", "1.15")))
            self.test_attn_mode_var.set(str(s.get("test_attn_mode", "torch")))
        self.refresh_checkpoints()
        self.refresh_model_manager()

    def refresh_checkpoints(self):
        if not hasattr(self, "resume_combo"):
            return
        output = project / "output"
        files = sorted(output.glob("*.safetensors"), key=lambda p: p.stat().st_mtime, reverse=True) if output.exists() else []
        values = ["Start fresh"] + [str(p) for p in files]
        self.resume_combo["values"] = values
        if self.resume_var.get() not in values:
            self.resume_var.set(values[0])
        if hasattr(self, "sample_lora_combo"):
            sample_values = ["Latest for current LoRA name"] + [str(p) for p in files]
            self.sample_lora_combo["values"] = sample_values
            if self.sample_lora_var.get() not in sample_values:
                self.sample_lora_var.set(sample_values[0])

    def validate_dataset(self):
        s = load_settings()
        train_dir = Path(s["train_images_dir"])
        cache_dir = Path(s["cache_dir"])
        images = sorted([p for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]) if train_dir.exists() else []
        captions = [p.with_suffix(".txt") for p in images]
        missing = [p.name for p, c in zip(images, captions) if not c.exists()]
        latent_cache = list(cache_dir.rglob("*_kr2.safetensors")) if cache_dir.exists() else []
        text_cache = list(cache_dir.rglob("*_kr2_te.safetensors")) if cache_dir.exists() else []
        unsupported = [p.name for p in train_dir.iterdir() if p.is_file() and p.suffix.lower() not in IMAGE_EXTS and p.suffix.lower() != ".txt"] if train_dir.exists() else []
        wf = get_workflow(s.get("workflow", "krea2"))
        model_checks = [(label, Path(s.get(key, ""))) for key, label in wf.get("model_fields", {}).items()]
        model_ok = sum(1 for _name, path in model_checks if str(path) and path.exists())
        model_msg = " | ".join(f"{name}: {'OK' if str(path) and path.exists() else 'MISSING'}" for name, path in model_checks)

        # Professional dataset summary: resolution/aspect profile and a plain-English recommendation.
        sizes = []
        bad_images = []
        for img_path in images:
            try:
                with Image.open(img_path) as im:
                    sizes.append((im.width, im.height))
            except Exception as e:
                bad_images.append(f"{img_path.name}: {e}")
        aspect_counts = {"square": 0, "landscape": 0, "portrait": 0}
        min_side = None
        max_pixels = 0
        for w, h in sizes:
            min_side = min(w, h) if min_side is None else min(min_side, w, h)
            max_pixels = max(max_pixels, w * h)
            r = w / max(1, h)
            if 0.95 <= r <= 1.05:
                aspect_counts["square"] += 1
            elif r > 1.05:
                aspect_counts["landscape"] += 1
            else:
                aspect_counts["portrait"] += 1
        aspect_msg = f"landscape {aspect_counts['landscape']} / square {aspect_counts['square']} / portrait {aspect_counts['portrait']}"
        cache_msg = f"{len(latent_cache)} / {len(text_cache)}"
        train_limit = f"{s.get('max_train_steps')} steps" if int(s.get("max_train_steps") or 0) > 0 else f"{s.get('epochs')} epochs"
        res_msg = ", ".join(selected_resolutions(s))

        if hasattr(self, "metric_vars"):
            self.metric_vars.get("images", tk.StringVar()).set(str(len(images)))
            self.metric_vars.get("captions", tk.StringVar()).set(f"{len(images) - len(missing)}/{len(images)}")
            self.metric_vars.get("cache", tk.StringVar()).set(cache_msg)
            self.metric_vars.get("models", tk.StringVar()).set(f"{model_ok}/{len(model_checks)} OK")
            self.metric_vars.get("plan", tk.StringVar()).set(train_limit)

        recommendation = []
        if images and aspect_counts["portrait"] == 0 and aspect_counts["square"] == 0:
            recommendation.append("All images are landscape; add square/portrait images if you want the LoRA to generalize outside 16:9.")
        if len(images) >= 150:
            recommendation.append("Dataset size is good for a style LoRA; compare checkpoints around 2k–3k steps.")
        elif images:
            recommendation.append("Dataset is small; use lower LR/steps and watch for overfitting.")
        if missing:
            recommendation.append("Fix missing captions before caching text.")
        if not missing and images:
            recommendation.append("Caption coverage is complete.")

        msg = (
            f"Dataset folder: {train_dir}\n"
            f"Images: {len(images)} | Captions found: {len(images) - len(missing)} | Missing captions: {len(missing)} | Bad images: {len(bad_images)}\n"
            f"Aspect profile: {aspect_msg} | Smallest side: {min_side or 'n/a'} | Largest image: {max_pixels:,} px\n"
            f"Training resolution buckets: {res_msg}\n"
            f"Cache folder: {cache_dir}\n"
            f"Latent cache files: {len(latent_cache)} | Text cache files: {len(text_cache)}\n"
            f"Unsupported files: {len(unsupported)}\n"
            f"Models: {model_msg}\n"
            f"Recommendation: {' '.join(recommendation) if recommendation else 'Ready.'}"
        )
        if missing[:8]:
            msg += "\nMissing caption examples: " + ", ".join(missing[:8])
        if unsupported[:8]:
            msg += "\nUnsupported examples: " + ", ".join(unsupported[:8])
        if bad_images[:5]:
            msg += "\nBad image examples: " + ", ".join(bad_images[:5])
        self.dataset_status.set(msg)
        self.append("\n===== Dataset Validation =====\n" + msg + "\n")

    def diagnostics_report(self):
        s = load_settings()
        wf = get_workflow(s.get("workflow", "krea2"))
        lines = [
            f"Workflow: {wf['label']} ({s.get('workflow', 'krea2')})",
            f"Project: {project}",
            f"Musubi repo: {root_dir} [{'OK' if root_dir.exists() else 'MISSING'}]",
            f"Python: {py} [{'OK' if py.exists() else 'MISSING'}]",
            f"Dataset config: {dataset} [{'OK' if dataset.exists() else 'MISSING'}]",
            f"Training images: {s.get('train_images_dir')} [{'OK' if Path(s.get('train_images_dir', '')).exists() else 'MISSING'}]",
            f"Cache folder: {s.get('cache_dir')} [{'OK' if Path(s.get('cache_dir', '')).exists() else 'MISSING'}]",
        ]
        for key, rel in wf.get("scripts", {}).items():
            script = root_dir / rel
            lines.append(f"Script {key}: {script.name} [{'OK' if script.exists() else 'MISSING'}]")
        for key, label in wf.get("model_fields", {}).items():
            value = str(s.get(key, "")).strip()
            lines.append(f"Model {label}: {value or '(not set)'} [{'OK' if value and Path(value).exists() else 'MISSING'}]")
        try:
            out = subprocess.check_output(
                [str(py), "-c", "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no cuda')"],
                cwd=str(root_dir), text=True, stderr=subprocess.STDOUT, timeout=30, encoding="utf-8", errors="replace"
            ).strip().splitlines()
            if len(out) >= 4:
                lines += [f"Torch: {out[0]}", f"Torch CUDA: {out[1]}", f"CUDA available: {out[2]}", f"GPU: {out[3]}"]
        except Exception as e:
            lines.append(f"Torch/CUDA check failed: {e}")
        if wf.get("experimental"):
            lines.append("NOTE: This workflow is an experimental template and may need model-specific arguments adjusted.")
        return "\n".join(lines)

    def run_diagnostics(self):
        report = self.diagnostics_report()
        if hasattr(self, "diagnostics_text"):
            self.diagnostics_text.set(report)
        self.append("\n===== Diagnostics =====\n" + report + "\n")

    def preview_train_command(self):
        cmd = self.build_train_cmd(False)
        self.append("\n===== Training Command Preview =====\n" + " ".join(map(str, cmd)) + "\n")

    def show_launch_command(self):
        cmd = self.build_train_cmd(False)
        text = (
            "===== Train Launch Command =====\n"
            + " ".join(map(str, cmd))
            + "\n\n===== Dataset Config =====\n"
            + (dataset.read_text(encoding="utf-8") if dataset.exists() else "Dataset config does not exist yet.")
        )
        win = tk.Toplevel(self)
        win.title("Launch Command Preview")
        win.geometry("1100x620")
        win.configure(bg="#0b1220")
        box = scrolledtext.ScrolledText(win, wrap="word", font=("Cascadia Mono", 9), bg="#0b1020", fg="#d6deff", insertbackground="#f8fafc", relief="flat")
        box.pack(fill="both", expand=True, padx=12, pady=12)
        box.insert("1.0", text)
        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=12, pady=(0, 12))
        def copy():
            self.clipboard_clear()
            self.clipboard_append(" ".join(map(str, cmd)))
            self.append("\nCopied launch command to clipboard.\n")
        ttk.Button(btns, text="Copy Command", command=copy, style="Accent.TButton").pack(side="right", padx=6)
        ttk.Button(btns, text="Close", command=win.destroy).pack(side="right", padx=6)
        self.append("\n===== Training Command Preview =====\n" + " ".join(map(str, cmd)) + "\n")

    def export_debug_bundle(self):
        logs_dir.mkdir(parents=True, exist_ok=True)
        out = logs_dir / f"debug_bundle_{time.strftime('%Y%m%d_%H%M%S')}.zip"
        report = self.diagnostics_report()
        with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("diagnostics.txt", report)
            for file in [settings_path, dataset, sample_prompts, project / "musubi_workflows.py", project / "krea2_launcher_gui.py"]:
                if file.exists():
                    z.write(file, file.name)
            run_dir = logs_dir / "runs"
            if run_dir.exists():
                latest = sorted(run_dir.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]
                for log_file in latest:
                    z.write(log_file, f"runs/{log_file.name}")
        self.append(f"\nExported debug bundle: {out}\n")
        messagebox.showinfo("Debug bundle exported", str(out))

    def missing_required_for_training(self):
        s = load_settings()
        wf = get_workflow(s.get("workflow", "krea2"))
        missing = []
        if not Path(s.get("train_images_dir", "")).exists():
            missing.append("training images folder")
        for key in wf.get("train_model_args", {}).values():
            value = str(s.get(key, "")).strip()
            if not value or not Path(value).exists():
                missing.append(wf.get("model_fields", {}).get(key, key))
        train_script = rel_script(wf, "train")
        if not train_script.exists():
            missing.append(f"train script: {train_script}")
        return missing

    def clear_cache(self, kind):
        cache_dir = spath("cache_dir")
        if not cache_dir.exists():
            return
        patterns = ["*_kr2.safetensors"] if kind == "latent" else ["*_kr2_te.safetensors"] if kind == "text" else ["*.safetensors"]
        files = []
        for pat in patterns:
            files.extend(cache_dir.rglob(pat))
        if not files:
            messagebox.showinfo("Cache", "No matching cache files found.")
            return
        if not messagebox.askyesno("Clear cache", f"Delete {len(files)} cache files?"):
            return
        for p in files:
            p.unlink(missing_ok=True)
        self.validate_dataset()

    def choose_train_folder(self):
        s = load_settings()
        chosen = filedialog.askdirectory(initialdir=s.get("train_images_dir") or str(project))
        if not chosen:
            return
        s["train_images_dir"] = chosen
        settings_path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        _fix_dataset_paths(s)
        self.refresh_settings()
        self.validate_dataset()

    def open_settings(self):
        subprocess.Popen([str(py), str(settings_gui)], cwd=str(root_dir)).wait()
        self.refresh_settings()
        self.validate_dataset()

    def open_path(self, path: Path):
        path.mkdir(parents=True, exist_ok=True)
        subprocess.Popen(["explorer", str(path)])

    def setup_log_colors(self):
        self.log.tag_configure("header", foreground="#67e8f9", font=("Cascadia Mono", 9, "bold"))
        self.log.tag_configure("command", foreground="#c4b5fd")
        self.log.tag_configure("path", foreground="#93c5fd")
        self.log.tag_configure("info", foreground="#93c5fd")
        self.log.tag_configure("warning", foreground="#fbbf24")
        self.log.tag_configure("error", foreground="#fb7185", font=("Cascadia Mono", 9, "bold"))
        self.log.tag_configure("success", foreground="#86efac")
        self.log.tag_configure("progress", foreground="#5eead4")
        self.log.tag_configure("muted", foreground="#94a3b8")

    def log_tag_for(self, text):
        low = text.lower()
        stripped = text.strip()
        if stripped.startswith("=====") or stripped.endswith("====="):
            return "header"
        if "traceback" in low or "error" in low or "exception" in low or "failed" in low or "modulenotfounderror" in low:
            return "error"
        if "warning" in low or "missing" in low or "oom" in low:
            return "warning"
        if "avr_loss=" in low or "s/it" in low or "it/s" in low or "steps:" in low:
            return "progress"
        if low.startswith("info:") or "info:" in low:
            return "info"
        if "finished with code 0" in low or " ok" in low or "cuda true" in low:
            return "success"
        if stripped.startswith(str(py)) or " accelerate.commands.launch " in text or stripped.startswith("Run log:"):
            return "command"
        if ":\\" in text or ":/" in text:
            return "path"
        return None

    def color_line_at(self, index):
        if not hasattr(self, "log"):
            return
        start = self.log.index(f"{index} linestart")
        end = self.log.index(f"{index} lineend")
        line = self.log.get(start, end)
        tag = self.log_tag_for(line)
        for t in ["header", "command", "path", "info", "warning", "error", "success", "progress", "muted"]:
            self.log.tag_remove(t, start, end)
        if tag:
            self.log.tag_add(tag, start, end)

    def append(self, text):
        if hasattr(self, "log"):
            start = self.log.index("end-1c")
            self.log.insert("end", text)
            end = self.log.index("end-1c")
            tag = self.log_tag_for(text)
            if tag and len(text) > 1:
                self.log.tag_add(tag, start, end)
            if "\n" in text:
                self.color_line_at("end-2l")
            elif "\r" in text:
                self.color_line_at("end-1c")
            self.log.see("end")
        if self.log_handle:
            try:
                self.log_handle.write(text)
                self.log_handle.flush()
            except Exception:
                pass

    def update_training_stats(self, text):
        # tqdm may report either "10.60s/it" or "1.23it/s". At step 0 it often shows "?it/s".
        m = re.search(r"(\d+)/(\d+).*?\[([^\]]+)\].*?(?:,\s*)?([0-9.]+)s/it.*?avr_loss=([0-9.eE+-]+)", text)
        speed = None
        eta = "?"
        loss = None
        if m:
            step, total, eta_block, sec_it, loss = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
            eta = eta_block.split("<")[-1] if "<" in eta_block else eta_block
            speed = f"{sec_it}s/it"
        else:
            m = re.search(r"(\d+)/(\d+).*?([0-9.]+)s/it.*?avr_loss=([0-9.eE+-]+)", text)
            if m:
                step, total, sec_it, loss = m.group(1), m.group(2), m.group(3), m.group(4)
                speed = f"{sec_it}s/it"
            else:
                m = re.search(r"(\d+)/(\d+).*?\[([^\]]+)\].*?(?:,\s*)?([0-9.]+)it/s.*?avr_loss=([0-9.eE+-]+)", text)
                if m:
                    step, total, eta_block, it_s, loss = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
                    eta = eta_block.split("<")[-1] if "<" in eta_block else eta_block
                    speed = f"{it_s}it/s"
                else:
                    m = re.search(r"(\d+)/(\d+).*?(\?it/s|[0-9.]+it/s|[0-9.]+s/it)", text)
                    if not m:
                        return
                    step, total, speed = m.group(1), m.group(2), m.group(3)
                    if speed == "?it/s":
                        self.train_stats.set(f"Step {step}/{total} | Speed warming up... | ETA calculating... | Avg loss waiting...")
                        return
        self.train_stats.set(f"Step {step}/{total} | Speed {speed} | ETA {eta} | Avg loss {loss if loss is not None else '?'}")

    def run(self, cmd, title="Command"):
        if self.proc and self.proc.poll() is None:
            messagebox.showwarning("Already running", "A command is already running.")
            return
        run_logs = logs_dir / "runs"
        run_logs.mkdir(parents=True, exist_ok=True)
        safe_title = re.sub(r"[^A-Za-z0-9_.-]+", "_", title).strip("_") or "command"
        log_path = run_logs / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_title}.log"
        self.log_handle = open(log_path, "w", encoding="utf-8", errors="replace")
        self.append(f"\n===== {title} =====\n")
        self.append(f"Run log: {log_path}\n")
        self.append(" ".join(map(str, cmd)) + "\n\n")
        self.train_stats.set("Training stats: starting..." if "Train" in title or "Benchmark" in title else "Training stats: idle")

        def worker():
            try:
                env = os.environ.copy()
                extra_pythonpath = [str(root_dir / "src"), str(root_dir / "src" / "musubi_tuner")]
                existing_pythonpath = env.get("PYTHONPATH")
                if existing_pythonpath:
                    extra_pythonpath.append(existing_pythonpath)
                env["PYTHONPATH"] = os.pathsep.join(extra_pythonpath)
                if cuda126.exists():
                    env["CUDA_PATH"] = str(cuda126)
                    env["CUDA_HOME"] = str(cuda126)
                    extra_path = [str(cuda126 / "bin"), str(root_dir / ".venv-sage" / "Lib" / "site-packages" / "torch" / "lib")]
                    env["PATH"] = os.pathsep.join(extra_path + [env.get("PATH", "")])
                self.proc = subprocess.Popen([str(x) for x in cmd], cwd=str(root_dir), env=env, stdout=subprocess.PIPE,
                                             stderr=subprocess.STDOUT, text=True, bufsize=1,
                                             encoding="utf-8", errors="replace")
                line_buf = ""
                while True:
                    ch = self.proc.stdout.read(1)
                    if ch == "" and self.proc.poll() is not None:
                        break
                    if not ch:
                        continue
                    self.after(0, self.append, ch)
                    if ch in "\r\n":
                        if line_buf:
                            self.after(0, self.update_training_stats, line_buf)
                        line_buf = ""
                    else:
                        line_buf += ch
                        if "avr_loss=" in line_buf and "s/it" in line_buf:
                            self.after(0, self.update_training_stats, line_buf)
                code = self.proc.wait()
                self.after(0, self.append, f"\n===== Finished with code {code} =====\n")
                if title.startswith("Benchmark") and self.last_benchmark_start:
                    elapsed = time.time() - self.last_benchmark_start
                    self.after(0, self.train_stats.set, f"Benchmark finished in {elapsed/60:.1f} min for 20 steps")
            except Exception as e:
                self.after(0, self.append, f"\nERROR: {e}\n")
            finally:
                self.proc = None
                if self.log_handle:
                    try:
                        self.log_handle.close()
                    except Exception:
                        pass
                    self.log_handle = None
                self.after(0, self.refresh_checkpoints)
                self.after(0, self.refresh_sample_preview)

        if title.startswith("Benchmark"):
            self.last_benchmark_start = time.time()
        threading.Thread(target=worker, daemon=True).start()

    def stop(self):
        if self.proc and self.proc.poll() is None:
            pid = self.proc.pid
            self.append(f"\nStopping process tree PID {pid}...\n")
            try:
                # accelerate launches a child python process, so terminating only the parent
                # often leaves training running. taskkill /T kills the full tree.
                subprocess.run(["taskkill", "/PID", str(pid), "/T", "/F"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=20)
            except Exception as e:
                self.append(f"taskkill failed, trying normal terminate: {e}\n")
                try:
                    self.proc.terminate()
                except Exception:
                    pass
            self.append("Stop requested. Training process tree should be terminated.\n")

    def download(self):
        self.run([py, project / "download_krea2_models.py"], "Download Models")

    def cache_latents(self):
        s = load_settings()
        wf = get_workflow(s.get("workflow", "krea2"))
        if "cache_latents" not in wf.get("scripts", {}):
            messagebox.showinfo("Not available", f"{wf['label']} does not have a standard cache latents step configured yet.")
            return
        cmd = [py, script_runner, rel_script(wf, "cache_latents"), "--dataset_config", dataset]
        add_model_args(cmd, wf, "cache_latents_model_args", s)
        self.run(cmd, "Cache Latents")

    def cache_text(self):
        s = load_settings()
        wf = get_workflow(s.get("workflow", "krea2"))
        if "cache_text" not in wf.get("scripts", {}):
            messagebox.showinfo("Not available", f"{wf['label']} does not have a standard cache text step configured yet.")
            return
        cmd = [py, script_runner, rel_script(wf, "cache_text"), "--dataset_config", dataset]
        add_model_args(cmd, wf, "cache_text_model_args", s)
        cmd += ["--batch_size", "1"]
        self.run(cmd, "Cache Text Encoder Outputs")

    def build_train_cmd(self, benchmark=False):
        s = load_settings()
        wf = get_workflow(s.get("workflow", "krea2"))
        cmd = [py, "-m", "accelerate.commands.launch", "--num_cpu_threads_per_process", "1", "--num_processes", "1", "--mixed_precision", s["mixed_precision"],
               script_runner, rel_script(wf, "train")]
        add_model_args(cmd, wf, "train_model_args", s)
        attention_map = {
            "sdpa": "--sdpa",
            "sage_attn": "--sage_attn",
            "xformers": "--xformers",
            "flash_attn": "--flash_attn",
            "flash3": "--flash3",
        }
        attention_flag = attention_map.get(str(s.get("attention_mode", "sdpa")), "--sdpa")
        cmd += ["--dataset_config", dataset,
               attention_flag, *wf.get("default_train_args", []), "--mixed_precision", s["mixed_precision"], "--save_precision", s["save_precision"],
               "--blocks_to_swap", str(s["blocks_to_swap"]), "--block_swap_ring_size", str(s["block_swap_ring_size"]),
               "--timestep_sampling", s["timestep_sampling"], "--weighting_scheme", s["weighting_scheme"],
               "--optimizer_type", s["optimizer_type"], "--learning_rate", str(s["learning_rate"]),
               "--max_data_loader_n_workers", str(s["max_data_loader_n_workers"]),
               "--network_module", wf["network_module"], "--network_dim", str(s["network_dim"]), "--network_alpha", str(s["network_alpha"]),
               "--save_every_n_steps", str(s["save_every_steps"]), "--seed", str(s["seed"]),
               "--output_dir", project / "output", "--output_name", (s["lora_name"] + ("_benchmark" if benchmark else ""))]
        def add_opt(flag, key):
            value = str(s.get(key, "")).strip()
            if value not in {"", "0", "None", "none"}:
                cmd.extend([flag, value])

        def add_list_opt(flag, key):
            value = str(s.get(key, "")).strip()
            if value:
                cmd.append(flag)
                cmd.extend(value.split())

        add_opt("--gradient_accumulation_steps", "gradient_accumulation_steps")
        add_opt("--max_grad_norm", "max_grad_norm")
        spec_fields = {f[0] for f in WORKFLOW_SPECIFIC_UI.get(s.get("workflow", "krea2"), WORKFLOW_SPECIFIC_UI["default"]).get("fields", [])}
        for key, flag in [("guidance_scale", "--guidance_scale"), ("vae_dtype", "--vae_dtype"), ("wan_task", "--task"), ("timestep_boundary", "--timestep_boundary")]:
            if key in spec_fields:
                add_opt(flag, key)
        add_opt("--lr_scheduler", "lr_scheduler")
        add_opt("--lr_warmup_steps", "lr_warmup_steps")
        add_opt("--lr_decay_steps", "lr_decay_steps")
        add_opt("--lr_scheduler_num_cycles", "lr_scheduler_num_cycles")
        add_opt("--lr_scheduler_power", "lr_scheduler_power")
        add_opt("--lr_scheduler_min_lr_ratio", "lr_scheduler_min_lr_ratio")
        add_list_opt("--optimizer_args", "optimizer_args")
        add_list_opt("--lr_scheduler_args", "lr_scheduler_args")
        add_opt("--discrete_flow_shift", "discrete_flow_shift")
        add_opt("--sigmoid_scale", "sigmoid_scale")
        add_opt("--logit_mean", "logit_mean")
        add_opt("--logit_std", "logit_std")
        add_opt("--mode_scale", "mode_scale")
        add_opt("--min_timestep", "min_timestep")
        add_opt("--max_timestep", "max_timestep")
        add_opt("--num_timestep_buckets", "num_timestep_buckets")
        add_opt("--scale_weight_norms", "scale_weight_norms")
        add_opt("--training_comment", "training_comment")
        add_opt("--metadata_title", "metadata_title")
        add_opt("--metadata_author", "metadata_author")
        add_opt("--metadata_description", "metadata_description")
        add_opt("--metadata_license", "metadata_license")
        add_opt("--metadata_tags", "metadata_tags")
        add_opt("--save_last_n_steps", "save_last_n_steps")
        add_opt("--save_last_n_epochs", "save_last_n_epochs")
        add_opt("--save_last_n_steps_state", "save_last_n_steps_state")
        add_opt("--save_last_n_epochs_state", "save_last_n_epochs_state")
        add_opt("--log_prefix", "log_prefix")
        add_opt("--log_tracker_name", "log_tracker_name")
        if s.get("compile"):
            cmd.append("--compile")
            add_opt("--compile_backend", "compile_backend")
            add_opt("--compile_mode", "compile_mode")
            add_opt("--compile_dynamic", "compile_dynamic")
        if benchmark:
            cmd += ["--max_train_steps", "20"]
        elif int(s.get("max_train_steps") or 0) > 0:
            cmd += ["--max_train_steps", str(int(s.get("max_train_steps") or 0))]
        else:
            cmd += ["--max_train_epochs", str(s["epochs"])]
        add_flag(cmd, s.get("fp8_base"), "--fp8_base")
        add_flag(cmd, s.get("fp8_scaled") and wf.get("supports_fp8_scaled", True), "--fp8_scaled")
        add_flag(cmd, s.get("use_pinned_memory_for_block_swap"), "--use_pinned_memory_for_block_swap")
        add_flag(cmd, s.get("block_swap_h2d_only"), "--block_swap_h2d_only")
        spec_bool_keys = {k for k, _label in WORKFLOW_SPECIFIC_UI.get(s.get("workflow", "krea2"), WORKFLOW_SPECIFIC_UI["default"]).get("bools", [])}
        for key, flag in [
            ("split_attn", "--split_attn"), ("flash_attn", "--flash_attn"), ("flash3", "--flash3"),
            ("fp8_llm", "--fp8_llm"), ("fp8_vl", "--fp8_vl"), ("fp8_t5", "--fp8_t5"),
            ("fp8_text_encoder", "--fp8_text_encoder"), ("full_bf16", "--full_bf16"),
            ("use_32bit_attention", "--use_32bit_attention"),
        ]:
            if key in spec_bool_keys:
                add_flag(cmd, s.get(key), flag)
        add_flag(cmd, s.get("gradient_checkpointing"), "--gradient_checkpointing")
        add_flag(cmd, s.get("gradient_checkpointing_cpu_offload"), "--gradient_checkpointing_cpu_offload")
        add_flag(cmd, s.get("persistent_data_loader_workers"), "--persistent_data_loader_workers")
        add_flag(cmd, s.get("sample_at_first"), "--sample_at_first")
        add_flag(cmd, s.get("preserve_distribution_shape"), "--preserve_distribution_shape")
        add_flag(cmd, s.get("log_config"), "--log_config")
        add_flag(cmd, s.get("no_metadata"), "--no_metadata")
        add_flag(cmd, s.get("dim_from_weights"), "--dim_from_weights")
        add_flag(cmd, s.get("save_state"), "--save_state")
        add_flag(cmd, s.get("save_state_on_train_end"), "--save_state_on_train_end")
        add_flag(cmd, s.get("compile_fullgraph"), "--compile_fullgraph")
        add_flag(cmd, s.get("cuda_allow_tf32"), "--cuda_allow_tf32")
        add_flag(cmd, s.get("cuda_cudnn_benchmark"), "--cuda_cudnn_benchmark")
        add_flag(cmd, s.get("img_in_txt_in_offloading"), "--img_in_txt_in_offloading")
        add_flag(cmd, s.get("disable_numpy_memmap"), "--disable_numpy_memmap")
        if str(s.get("network_dropout", "")).strip():
            cmd += ["--network_dropout", str(s.get("network_dropout")).strip()]
        if s.get("enable_tensorboard", True):
            cmd += ["--logging_dir", logs_dir, "--log_with", "tensorboard"]
        selected = self.resume_var.get() if hasattr(self, "resume_var") else "Start fresh"
        if selected and selected != "Start fresh" and Path(selected).exists() and not benchmark:
            cmd += ["--network_weights", selected]
        if s.get("enable_training_samples") and not benchmark:
            cmd += ["--sample_prompts", sample_prompts, "--sample_every_n_steps", str(s["sample_every_steps"])]
        return cmd

    def train(self):
        missing = self.missing_required_for_training()
        if missing:
            messagebox.showerror("Missing required files", "Cannot start training yet. Missing:\n\n" + "\n".join(missing))
            self.run_diagnostics()
            return
        wf = current_workflow()
        self.run(self.build_train_cmd(False), f"Train {wf['label']}")

    def benchmark(self):
        missing = self.missing_required_for_training()
        if missing:
            messagebox.showerror("Missing required files", "Cannot benchmark yet. Missing:\n\n" + "\n".join(missing))
            self.run_diagnostics()
            return
        self.run(self.build_train_cmd(True), "Benchmark 20 Steps")

    def open_tensorboard(self):
        logs_dir.mkdir(parents=True, exist_ok=True)
        if self.tb_proc and self.tb_proc.poll() is None:
            subprocess.Popen(["cmd", "/c", "start", "", "http://localhost:6006"])
            return
        self.tb_proc = subprocess.Popen([str(py), "-m", "tensorboard.main", "--logdir", str(logs_dir), "--port", "6006"], cwd=str(root_dir))
        time.sleep(1)
        subprocess.Popen(["cmd", "/c", "start", "", "http://localhost:6006"])

    def save_sample_settings(self):
        s = load_settings()
        try:
            strength = float(self.test_lora_multiplier_var.get()) if hasattr(self, "test_lora_multiplier_var") else float(s.get("test_lora_multiplier", "1.0"))
            width = int(self.test_width_var.get()) if hasattr(self, "test_width_var") else int(s.get("test_width", 1024))
            height = int(self.test_height_var.get()) if hasattr(self, "test_height_var") else int(s.get("test_height", 1024))
            steps = int(self.test_steps_var.get()) if hasattr(self, "test_steps_var") else int(s.get("test_steps", 8))
            seed_mode = self.test_seed_mode_var.get().strip().lower() if hasattr(self, "test_seed_mode_var") else str(s.get("test_seed_mode", "fixed"))
            seed = int(self.test_seed_var.get()) if hasattr(self, "test_seed_var") else int(s.get("test_seed", 1))
            num_images = int(self.test_num_images_var.get()) if hasattr(self, "test_num_images_var") else int(s.get("test_num_images", 1))
            guidance = float(self.test_guidance_var.get()) if hasattr(self, "test_guidance_var") else float(s.get("test_guidance_scale", "1"))
            mu = self.test_mu_var.get().strip() if hasattr(self, "test_mu_var") else str(s.get("test_mu", "1.15"))
            attn_mode = self.test_attn_mode_var.get().strip() if hasattr(self, "test_attn_mode_var") else str(s.get("test_attn_mode", "torch"))
            negative = self.test_negative_text.get("1.0", "end").strip() if hasattr(self, "test_negative_text") else str(s.get("test_negative_prompt", ""))
            if strength <= 0:
                raise ValueError("LoRA strength must be greater than 0")
            if width < 256 or height < 256:
                raise ValueError("Image size must be at least 256")
            if steps < 1 or num_images < 1:
                raise ValueError("Steps and image count must be at least 1")
        except Exception as e:
            messagebox.showerror("Invalid sample setting", str(e))
            return None
        s["test_lora_multiplier"] = str(strength).rstrip("0").rstrip(".") if strength != int(strength) else str(int(strength))
        s["test_width"] = width
        s["test_height"] = height
        s["test_steps"] = steps
        s["test_seed_mode"] = "random" if seed_mode == "random" else "fixed"
        s["test_seed"] = seed
        s["test_num_images"] = num_images
        s["test_guidance_scale"] = str(guidance).rstrip("0").rstrip(".") if guidance != int(guidance) else str(int(guidance))
        s["test_mu"] = mu
        s["test_attn_mode"] = attn_mode
        s["test_negative_prompt"] = negative
        settings_path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
        return s

    def test_image(self):
        s = self.save_sample_settings()
        if s is None:
            return
        prompt = self.test_prompt_text.get("1.0", "end").strip() if hasattr(self, "test_prompt_text") else "a beautiful cinematic portrait photo"
        if not prompt:
            messagebox.showerror("Missing prompt", "Enter a prompt in the Samples tab first.")
            return
        lora_weight = self.selected_lora_for_test()
        if not lora_weight.exists():
            messagebox.showerror("No LoRA checkpoint found", f"Could not find a LoRA to test. Expected or latest path:\n{lora_weight}")
            return
        wf = get_workflow(s.get("workflow", "krea2"))
        if "generate" not in wf.get("scripts", {}):
            messagebox.showinfo("Not available", f"{wf['label']} does not have a sample generation script configured.")
            return
        if s.get("workflow", "krea2") == "zimage":
            cmd = [py, script_runner, rel_script(wf, "generate")]
            add_model_args(cmd, wf, "test_model_args", s)
            cmd += ["--prompt", prompt,
                    "--image_size", str(s.get("test_height", 1024)), str(s.get("test_width", 1024)),
                    "--infer_steps", str(s.get("test_steps", 8)),
                    "--guidance_scale", str(s.get("test_guidance_scale", "1")),
                    "--attn_mode", "sdpa" if str(s.get("test_attn_mode", "torch")) == "torch" else str(s.get("test_attn_mode", "torch")),
                    "--save_path", project / "samples",
                    "--blocks_to_swap", str(s["blocks_to_swap"]),
                    "--lora_weight", lora_weight, "--lora_multiplier", str(s.get("test_lora_multiplier", "1.0"))]
            if str(s.get("test_seed_mode", "fixed")).lower() != "random":
                cmd += ["--seed", str(s.get("test_seed", 1))]
            if str(s.get("test_negative_prompt", "")).strip():
                cmd += ["--negative_prompt", str(s.get("test_negative_prompt")).strip()]
            add_flag(cmd, s.get("fp8_scaled"), "--fp8_scaled")
            add_flag(cmd, s.get("use_pinned_memory_for_block_swap"), "--use_pinned_memory_for_block_swap")
        elif s.get("workflow", "krea2") == "krea2":
            cmd = [py, script_runner, rel_script(wf, "generate"), prompt]
            add_model_args(cmd, wf, "test_model_args", s)
            default_args = list(wf.get("default_test_args", []))
            for flag in ["--width", "--height", "--steps", "--guidance_scale", "--mu", "--seed", "--attn_mode"]:
                while flag in default_args:
                    i = default_args.index(flag)
                    del default_args[i:i + 2]
            cmd += [*default_args,
                   "--steps", str(s.get("test_steps", 8)), "--guidance_scale", str(s.get("test_guidance_scale", "1")),
                   "--mu", str(s.get("test_mu", "1.15")), "--width", str(s.get("test_width", 1024)), "--height", str(s.get("test_height", 1024)),
                   "--num-images", str(s.get("test_num_images", 1)),
                   "--attn_mode", str(s.get("test_attn_mode", "torch")),
                   "--save_path", project / "samples",
                   "--blocks_to_swap", str(s["blocks_to_swap"]), "--block_swap_h2d_only",
                   "--lora_weight", lora_weight, "--lora_multiplier", str(s.get("test_lora_multiplier", "1.0"))]
            if str(s.get("test_seed_mode", "fixed")).lower() != "random":
                cmd += ["--seed", str(s.get("test_seed", 1))]
            if str(s.get("test_negative_prompt", "")).strip():
                cmd += ["--negative_prompt", str(s.get("test_negative_prompt")).strip()]
            add_flag(cmd, s.get("fp8_scaled"), "--fp8_scaled")
        else:
            messagebox.showinfo("Sample generation", f"Sample generation UI is configured for Krea2 and Z-Image right now. Training/caching still works for {wf['label']} via the Workflow tab.")
            return
        seed_text = "random" if str(s.get("test_seed_mode", "fixed")).lower() == "random" else str(s.get("test_seed", 1))
        self.append(f"\nUsing LoRA for test image: {lora_weight}\nSample settings: strength={s.get('test_lora_multiplier')} size={s.get('test_width')}x{s.get('test_height')} steps={s.get('test_steps')} seed={seed_text}\n")
        self.run(cmd, "Generate Test Image")


if __name__ == "__main__":
    Launcher().mainloop()
