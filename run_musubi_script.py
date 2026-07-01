"""Run a Musubi script after adding the local repo source folders to sys.path.

This makes the portable layout work:
  krea2_project/
    musubi-tuner/
    krea2_launcher_gui.py
"""
import runpy
import sys
from pathlib import Path

project = Path(__file__).resolve().parent
repo = project / "musubi-tuner"

# musubi_tuner package imports
sys.path.insert(0, str(repo / "src"))
# top-level imports used by Musubi network_module values, e.g. networks.lora_krea2
sys.path.insert(0, str(repo / "src" / "musubi_tuner"))

if len(sys.argv) < 2:
    raise SystemExit("Usage: run_musubi_script.py <script.py> [args...]")

script = Path(sys.argv[1]).resolve()
sys.argv = [str(script)] + sys.argv[2:]
runpy.run_path(str(script), run_name="__main__")
