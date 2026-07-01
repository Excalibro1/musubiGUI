# MusubiGUI

A Windows-focused GUI launcher for [kohya-ss/musubi-tuner](https://github.com/kohya-ss/musubi-tuner).

MusubiGUI provides a friendlier interface for model setup, dataset validation, caching, LoRA training, sampling, command preview, logs, and workflow-specific Musubi options.

## Features

- Multi-workflow Musubi launcher
- Auto-discovers supported Musubi workflows when possible
- Curated workflow definitions for Krea2, Z-Image, Qwen Image, Wan, Flux, FramePack, Hunyuan, Ideogram, Kandinsky, and HiDream templates
- Model path manager per workflow
- Dataset validation and cache tools
- Official Musubi bucket config behavior
- Advanced training settings with tooltips
- Workflow-specific settings tab per selected model
- Live log viewer
- Launch command preview and copy
- Sample generation with LoRA selector, seed mode, negative prompt, size, steps, guidance, and attention settings
- Krea2 model downloader helper

## Setup

### One-click Windows setup

1. Clone or download this repository.
2. Run:

```bat
setup.bat
```

The setup script will:

- clone/update `kohya-ss/musubi-tuner` into `musubi-tuner/`
- create `musubi-tuner/.venv-sage/`
- install PyTorch
- install musubi-tuner dependencies
- install TensorBoard and GUI dependencies
- create local folders such as `models/`, `train_images/`, `cache/`, `output/`, and `samples/`

Then launch:

```bat
gui.bat
```

### Manual setup

```bat
git clone https://github.com/Excalibro1/musubiGUI.git
cd musubiGUI
git clone https://github.com/kohya-ss/musubi-tuner.git
py -3.11 -m venv musubi-tuner\.venv-sage
musubi-tuner\.venv-sage\Scripts\python.exe -m pip install -U pip wheel "setuptools<81"
musubi-tuner\.venv-sage\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
cd musubi-tuner
.venv-sage\Scripts\python.exe -m pip install -e .
.venv-sage\Scripts\python.exe -m pip install tensorboard prompt-toolkit "setuptools<81"
cd ..
gui.bat
```

## Usage

1. Select a workflow from the top dropdown.
2. For Krea2, use **Download Krea2 Models** if needed.
3. For other workflows, click **Model Setup** and point every required model field to the correct file/folder.
4. Choose dataset folder in **Dataset / Files**.
5. Run:

```text
Cache Latents
Cache Text
Start Training
```

Use **Preview Launch Command** before training to inspect the exact command sent to Musubi.

## Repository layout

```text
MusubiGUI/
├─ musubi_gui.py              # main entry point
├─ krea2_launcher_gui.py      # GUI implementation
├─ musubi_workflows.py        # workflow registry + auto-discovery
├─ run_musubi_script.py       # script runner wrapper
├─ download_krea2_models.py   # Krea2 model downloader helper
├─ setup.bat                  # one-click setup
├─ gui.bat                    # Windows launcher
└─ musubi-tuner/              # local musubi-tuner checkout, not committed
```

CHATGPTMADETHISNOTME I ONLY TESTED KREA2 I JUST MADE CHATGPT MAKE THIS BECAUSE MY AITOOLKIT WASNT WORKING.
