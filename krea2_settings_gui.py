import json
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

try:
    from musubi_workflows import enabled_workflows
except Exception:
    enabled_workflows = None

project = Path(__file__).resolve().parent
settings_path = project / "krea2_settings.json"
env_path = project / "krea2_settings.env"
dataset_path = project / "configs" / "dataset_krea2.toml"


def p(path: Path) -> str:
    return str(path)


DEFAULTS = {
    "workflow": "krea2",
    "train_resolutions": ["1024x1024"],
    "lora_name": "krea2_lora",
    "resolution_width": 1024,
    "resolution_height": 1024,
    "enable_bucket": True,
    "bucket_no_upscale": True,
    "batch_size": 1,
    "num_repeats": 1,
    "epochs": 8,
    "max_train_steps": 0,
    "learning_rate": "1e-4",
    "network_dim": 32,
    "network_alpha": 32,
    "optimizer_type": "adamw8bit",
    "mixed_precision": "bf16",
    "save_precision": "float",
    "attention_mode": "sdpa",
    "blocks_to_swap": 8,
    "block_swap_ring_size": 2,
    "use_pinned_memory_for_block_swap": True,
    "block_swap_h2d_only": True,
    "fp8_base": True,
    "fp8_scaled": True,
    "gradient_checkpointing": True,
    "timestep_sampling": "krea2_shift",
    "discrete_flow_shift": "2.5",
    "weighting_scheme": "none",
    "network_dropout": "",
    "save_every_steps": 250,
    "sample_every_steps": 250,
    "enable_training_samples": True,
    "enable_tensorboard": True,
    "test_lora_multiplier": "1.0",
    "test_width": 1024,
    "test_height": 1024,
    "max_data_loader_n_workers": 2,
    "persistent_data_loader_workers": True,
    "seed": 42,
    "train_images_dir": p(project / "train_images"),
    "cache_dir": p(project / "cache"),
    "raw_model_path": p(project / "models" / "raw.safetensors"),
    "turbo_model_path": p(project / "models" / "turbo.safetensors"),
    "vae_model_path": p(project / "models" / "qwen_image_vae.safetensors"),
    "text_encoder_path": p(project / "models" / "qwen3vl_4b_bf16.safetensors"),
    "dit_model_path": "",
    "text_encoder1_path": "",
    "text_encoder2_path": "",
    "t5_path": "",
    "clip_path": "",
    "image_encoder_path": "",
    "text_encoder_qwen_path": "",
    "text_encoder_clip_path": "",
}

BASIC_FIELDS = [
    ("workflow", "Workflow", str),
    ("lora_name", "LoRA output name", str),
    ("resolution_width", "Training resolution width", int),
    ("resolution_height", "Training resolution height", int),
    ("batch_size", "Batch size", int),
    ("num_repeats", "Dataset repeats", int),
    ("epochs", "Max train epochs, ignored if max steps > 0", int),
    ("max_train_steps", "Max train steps, 0 = use epochs", int),
    ("learning_rate", "Learning rate", str),
    ("network_dim", "LoRA rank / dim", int),
    ("network_alpha", "LoRA alpha", int),
    ("blocks_to_swap", "Blocks to swap, 0-26, lower=faster/more VRAM", int),
    ("save_every_steps", "Save checkpoint every N steps", int),
    ("sample_every_steps", "Generate sample every N steps", int),
    ("seed", "Seed", int),
]

PATH_FIELDS = [
    ("train_images_dir", "Training images folder", "dir"),
    ("cache_dir", "Cache folder", "dir"),
    ("raw_model_path", "Krea 2 RAW model", "file"),
    ("turbo_model_path", "Krea 2 Turbo model", "file"),
    ("dit_model_path", "Generic DiT / transformer model", "file"),
    ("vae_model_path", "VAE model", "file"),
    ("text_encoder_path", "Text encoder", "file"),
    ("text_encoder1_path", "Text encoder 1", "file"),
    ("text_encoder2_path", "Text encoder 2", "file"),
    ("t5_path", "T5 text encoder", "file"),
    ("clip_path", "CLIP text encoder", "file"),
    ("image_encoder_path", "Image encoder", "file"),
    ("text_encoder_qwen_path", "Qwen text encoder", "file"),
    ("text_encoder_clip_path", "CLIP text encoder alt", "file"),
]

ADV_FIELDS = [
    ("optimizer_type", "Optimizer type", str),
    ("mixed_precision", "Mixed precision", str),
    ("save_precision", "Save precision", str),
    ("attention_mode", "Attention mode", str),
    ("block_swap_ring_size", "Block swap ring size", int),
    ("timestep_sampling", "Timestep sampling", str),
    ("discrete_flow_shift", "Discrete flow shift, used by shift", str),
    ("weighting_scheme", "Weighting scheme", str),
    ("network_dropout", "Network dropout, blank = off", str),
    ("test_lora_multiplier", "Test image LoRA multiplier", str),
    ("test_width", "Test image width", int),
    ("test_height", "Test image height", int),
    ("max_data_loader_n_workers", "Data loader workers", int),
]

LR_CHOICES = [
    "0.00001 (1e-5)",
    "0.00002 (2e-5)",
    "0.00005 (5e-5)",
    "0.0001 (1e-4)",
    "0.0002 (2e-4)",
    "0.0003 (3e-4)",
    "0.0005 (5e-4)",
    "0.001 (1e-3)",
]


def lr_display(value):
    value = str(value).strip()
    mapping = {
        "1e-5": "0.00001 (1e-5)", "0.00001": "0.00001 (1e-5)",
        "2e-5": "0.00002 (2e-5)", "0.00002": "0.00002 (2e-5)",
        "5e-5": "0.00005 (5e-5)", "0.00005": "0.00005 (5e-5)",
        "1e-4": "0.0001 (1e-4)", "0.0001": "0.0001 (1e-4)",
        "2e-4": "0.0002 (2e-4)", "0.0002": "0.0002 (2e-4)",
        "3e-4": "0.0003 (3e-4)", "0.0003": "0.0003 (3e-4)",
        "5e-4": "0.0005 (5e-4)", "0.0005": "0.0005 (5e-4)",
        "1e-3": "0.001 (1e-3)", "0.001": "0.001 (1e-3)",
    }
    return mapping.get(value.lower(), value)


def lr_value(value):
    value = str(value).strip()
    if "(" in value:
        return value.split("(", 1)[0].strip()
    return value


CHOICES = {
    "workflow": list(enabled_workflows().keys()) if enabled_workflows else ["krea2"],
    "resolution_width": ["256", "512", "768", "1024", "1280", "1328", "1536", "2048"],
    "resolution_height": ["256", "512", "768", "1024", "1280", "1328", "1536", "2048"],
    "learning_rate": LR_CHOICES,
    "optimizer_type": [
        "adamw8bit",
        "AdamW",
        "Adam",
        "Adamax",
        "NAdam",
        "RAdam",
        "SGD",
        "Adagrad",
        "Adadelta",
        "RMSprop",
        "Adafactor",
    ],
    "mixed_precision": ["bf16", "fp16", "no"],
    "save_precision": ["float", "fp16", "bf16"],
    "attention_mode": ["sdpa", "sage_attn", "xformers"],
    "timestep_sampling": ["krea2_shift", "shift", "sigmoid", "uniform"],
    "weighting_scheme": ["none", "sigma_sqrt", "logit_normal", "mode", "cosmap"],
}

BOOL_FIELDS = [
    ("enable_bucket", "Enable aspect-ratio buckets"),
    ("bucket_no_upscale", "Bucket no upscale"),
    ("use_pinned_memory_for_block_swap", "Use pinned memory for block swap"),
    ("block_swap_h2d_only", "Block swap H2D only"),
    ("fp8_base", "Use fp8 base"),
    ("fp8_scaled", "Use fp8 scaled"),
    ("gradient_checkpointing", "Gradient checkpointing"),
    ("persistent_data_loader_workers", "Persistent data loader workers"),
    ("enable_training_samples", "Generate samples during training"),
    ("enable_tensorboard", "Enable TensorBoard logging"),
]


KNOWN_KEYS = (
    {key for key, _label, _typ in BASIC_FIELDS + ADV_FIELDS}
    | {key for key, _label, _kind in PATH_FIELDS}
    | {key for key, _label in BOOL_FIELDS}
)


def json_type(value):
    if isinstance(value, bool):
        return bool
    if isinstance(value, int) and not isinstance(value, bool):
        return int
    if isinstance(value, float):
        return float
    return str


def parse_json_value(value, typ):
    text = str(value).strip()
    if typ is bool:
        return text.lower() in {"1", "true", "yes", "on", "y"}
    if typ is int:
        return int(text)
    if typ is float:
        return float(text)
    return text


def load_settings():
    data = DEFAULTS.copy()
    if settings_path.exists():
        data.update(json.loads(settings_path.read_text(encoding="utf-8")))
    return data


def as_posix_string(value: str) -> str:
    return Path(value).as_posix()


def save_settings(s):
    Path(s["cache_dir"]).mkdir(parents=True, exist_ok=True)
    Path(s["train_images_dir"]).mkdir(parents=True, exist_ok=True)
    settings_path.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    env_path.write_text("".join(f"set {k.upper()}={v}\n" for k, v in s.items()), encoding="utf-8")
    resolutions = s.get("train_resolutions") or [f"{s['resolution_width']}x{s['resolution_height']}"]
    first_w, first_h = [int(x) for x in str(resolutions[0]).split("x", 1)]
    dataset_path.write_text(
        f'''[general]
resolution = [{first_w}, {first_h}]
caption_extension = ".txt"
batch_size = {s['batch_size']}
enable_bucket = true
bucket_no_upscale = {str(s['bucket_no_upscale']).lower()}

[[datasets]]
image_directory = "{as_posix_string(s['train_images_dir'])}"
cache_directory = "{as_posix_string(s['cache_dir'])}"
num_repeats = {s['num_repeats']}
''',
        encoding="utf-8",
    )


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Musubi Training Settings")
        self.set_dynamic_geometry()
        self.apply_theme()
        self.settings = load_settings()
        self.vars = {}
        self.bool_vars = {}

        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True, padx=10, pady=10)
        basic = ttk.Frame(nb)
        paths = ttk.Frame(nb)
        advanced = ttk.Frame(nb)
        json_tab = ttk.Frame(nb)
        nb.add(basic, text="Basic")
        nb.add(paths, text="Paths / Models")
        nb.add(advanced, text="Advanced")
        nb.add(json_tab, text="Other JSON")

        self.extra_fields = [(key, json_type(value)) for key, value in self.settings.items() if key not in KNOWN_KEYS]
        self._add_fields(basic, BASIC_FIELDS)
        self._add_path_fields(paths)
        self._add_fields(advanced, ADV_FIELDS)
        self._add_extra_fields(json_tab)
        row = len(ADV_FIELDS) + 1
        ttk.Label(advanced, text="Feature toggles", font=("Segoe UI", 10, "bold")).grid(row=row, column=0, columnspan=3, sticky="w", pady=(12, 4))
        for i, (key, label) in enumerate(BOOL_FIELDS, row + 1):
            var = tk.BooleanVar(value=bool(self.settings[key]))
            self.bool_vars[key] = var
            self._add_toggle(advanced, i, key, label, var)

        btns = ttk.Frame(self)
        btns.pack(fill="x", side="bottom", padx=10, pady=(0, 10))
        ttk.Button(btns, text="Save", command=self.save, style="Accent.TButton").pack(side="right", padx=5)
        ttk.Button(btns, text="Cancel", command=self.destroy).pack(side="right", padx=5)
        ttk.Label(btns, text="Ctrl+S saves", foreground="#9ca3af").pack(side="left", padx=5)
        self.bind("<Control-s>", lambda _e: self.save())
        self.bind("<Escape>", lambda _e: self.destroy())

    def set_dynamic_geometry(self):
        """Size the settings window to the current monitor so Save/Cancel stay visible."""
        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        width = max(760, min(980, screen_w - 80))
        height = max(560, min(820, screen_h - 120))
        x = max(0, (screen_w - width) // 2)
        y = max(0, (screen_h - height) // 2)
        self.geometry(f"{width}x{height}+{x}+{y}")
        self.minsize(min(760, width), min(520, height))

    def _add_toggle(self, parent, row, key, label, var):
        def refresh():
            if var.get():
                btn.configure(text=f"  ON   {label}", bg="#14532d", fg="#dcfce7", activebackground="#166534", activeforeground="#ffffff")
            else:
                btn.configure(text=f"  OFF  {label}", bg="#7f1d1d", fg="#fee2e2", activebackground="#991b1b", activeforeground="#ffffff")

        def toggle():
            var.set(not var.get())
            refresh()

        btn = tk.Button(
            parent,
            command=toggle,
            relief="flat",
            bd=0,
            highlightthickness=0,
            anchor="w",
            width=46,
            padx=10,
            pady=4,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        )
        btn.grid(row=row, column=0, columnspan=3, sticky="w", padx=10, pady=3)
        refresh()

    def apply_theme(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg = "#0b1220"
        surface = "#111827"
        surface2 = "#172033"
        border = "#243044"
        fg = "#e5e7eb"
        muted = "#9ca3af"
        accent = "#2563eb"
        self.configure(bg=bg)
        style.configure(".", background=bg, foreground=fg, fieldbackground=surface, font=("Segoe UI", 10), bordercolor=border)
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg)
        style.configure("TButton", background=surface2, foreground=fg, padding=(10, 5), borderwidth=0)
        style.map("TButton", background=[("active", "#23324a")], foreground=[("active", "#ffffff")])
        style.configure("Accent.TButton", background=accent, foreground="#ffffff", padding=(12, 6), font=("Segoe UI", 10, "bold"))
        style.map("Accent.TButton", background=[("active", "#60a5fa")])
        style.configure("TNotebook", background=bg, borderwidth=0)
        style.configure("TNotebook.Tab", background=surface2, foreground=muted, padding=(14, 8), borderwidth=0)
        style.map("TNotebook.Tab", background=[("selected", surface)], foreground=[("selected", "#60a5fa")])
        style.configure("TEntry", fieldbackground=surface, foreground=fg, insertcolor=fg, bordercolor=border, padding=4)
        style.configure("TCombobox", fieldbackground=surface, background=surface2, foreground=fg, arrowcolor=fg, padding=4)
        style.map("TCombobox", fieldbackground=[("readonly", surface)], foreground=[("readonly", fg)])
        style.configure("TCheckbutton", background=bg, foreground=fg)

    def _add_fields(self, parent, fields):
        for row, (key, label, _typ) in enumerate(fields):
            ttk.Label(parent, text=label).grid(row=row, column=0, padx=10, pady=5, sticky="e")
            initial_value = lr_display(self.settings[key]) if key == "learning_rate" else str(self.settings[key])
            var = tk.StringVar(value=initial_value)
            self.vars[key] = var
            if key in CHOICES:
                widget = ttk.Combobox(parent, textvariable=var, values=CHOICES[key], width=36, state="normal")
            else:
                widget = ttk.Entry(parent, textvariable=var, width=38)
            widget.grid(row=row, column=1, padx=10, pady=5, sticky="w")
        parent.columnconfigure(1, weight=1)

    def _add_path_fields(self, parent):
        ttk.Label(parent, text="Point these at existing folders/files if you already have models downloaded.", foreground="#555").grid(row=0, column=0, columnspan=3, padx=10, pady=(8, 12), sticky="w")
        for row, (key, label, kind) in enumerate(PATH_FIELDS, start=1):
            ttk.Label(parent, text=label).grid(row=row, column=0, padx=10, pady=5, sticky="e")
            var = tk.StringVar(value=str(self.settings[key]))
            self.vars[key] = var
            ttk.Entry(parent, textvariable=var, width=62).grid(row=row, column=1, padx=10, pady=5, sticky="we")
            ttk.Button(parent, text="Browse", command=lambda k=key, t=kind: self.browse_path(k, t)).grid(row=row, column=2, padx=8, pady=5)
        parent.columnconfigure(1, weight=1)

    def _add_extra_fields(self, parent):
        ttk.Label(
            parent,
            text="Any settings present in krea2_settings.json that are not already shown on the other tabs appear here.",
            foreground="#9ca3af",
            wraplength=820,
        ).grid(row=0, column=0, columnspan=2, padx=10, pady=(8, 12), sticky="w")
        if not self.extra_fields:
            ttk.Label(parent, text="No extra JSON-only settings found.").grid(row=1, column=0, padx=10, pady=5, sticky="w")
            return
        for row, (key, typ) in enumerate(self.extra_fields, start=1):
            ttk.Label(parent, text=f"{key} ({typ.__name__})").grid(row=row, column=0, padx=10, pady=5, sticky="e")
            var = tk.StringVar(value=str(self.settings.get(key, "")))
            self.vars[key] = var
            ttk.Entry(parent, textvariable=var, width=58).grid(row=row, column=1, padx=10, pady=5, sticky="we")
        parent.columnconfigure(1, weight=1)

    def browse_path(self, key, kind):
        current = self.vars[key].get().strip()
        initial = current if current else str(project)
        if kind == "dir":
            chosen = filedialog.askdirectory(initialdir=initial if Path(initial).exists() else str(project))
        else:
            chosen = filedialog.askopenfilename(initialdir=str(Path(initial).parent if current else project), filetypes=[("Safetensors", "*.safetensors"), ("All files", "*.*")])
        if chosen:
            self.vars[key].set(chosen)

    def save(self):
        new = load_settings()
        try:
            for key, _label, typ in BASIC_FIELDS + ADV_FIELDS:
                value = self.vars[key].get().strip()
                if key == "learning_rate":
                    value = lr_value(value)
                new[key] = typ(value)
            for key, _label, _kind in PATH_FIELDS:
                new[key] = self.vars[key].get().strip()
            for key, _label in BOOL_FIELDS:
                new[key] = bool(self.bool_vars[key].get())
            for key, typ in self.extra_fields:
                new[key] = parse_json_value(self.vars[key].get(), typ)
            if not 0 <= new["blocks_to_swap"] <= 26:
                raise ValueError("Blocks to swap must be between 0 and 26")
            if new["block_swap_ring_size"] < 1:
                raise ValueError("Block swap ring size must be 1 or higher")
            if new["batch_size"] < 1:
                raise ValueError("Batch size must be 1 or higher")
            if new["max_train_steps"] < 0:
                raise ValueError("Max train steps must be 0 or higher")
            if str(new.get("network_dropout", "")).strip():
                nd = float(new["network_dropout"])
                if not 0 <= nd < 1:
                    raise ValueError("Network dropout must be blank or between 0 and 1")
            if float(new.get("test_lora_multiplier", "1.0")) <= 0:
                raise ValueError("Test image LoRA multiplier must be higher than 0")
            if not new["train_images_dir"]:
                raise ValueError("Training images folder is required")
            if not new["cache_dir"]:
                raise ValueError("Cache folder is required")
        except Exception as e:
            messagebox.showerror("Invalid setting", str(e))
            return
        save_settings(new)
        messagebox.showinfo("Saved", f"Settings saved.\n\nUpdated:\n{settings_path}\n{dataset_path}")
        self.destroy()


if __name__ == "__main__":
    App().mainloop()
