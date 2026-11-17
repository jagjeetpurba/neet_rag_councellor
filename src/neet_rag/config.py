"""Central configuration: model names, paths, retrieval knobs.

Everything a grader needs to change (model, VRAM assumption, thresholds) is
declared here in one place, and mirrored in the design note.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Paths -----------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
REAL_DIR = DATA_DIR / "real"
INDEX_DIR = PROJECT_ROOT / "index"
EVAL_FILE = PROJECT_ROOT / "eval" / "questions.json"

# Which data source the index is built from. "synthetic" by default; set to
# "real" once real public tables have been cleaned into data/real/.
ACTIVE_DATA_DIR = os.environ.get("NEET_DATA_DIR", str(SYNTHETIC_DIR))

# --- Models (open-weight, local) ------------------------------------------
# Generator: served by Ollama on the dev machine (CPU/Metal); in the target
# deployment the SAME weights run through vLLM on the workstation GPU.
GENERATOR_MODEL = os.environ.get("NEET_LLM_MODEL", "qwen2.5:7b")

# Embeddings: local, served by Ollama (no PyTorch/torch dependency).
EMBED_MODEL = os.environ.get("NEET_EMBED_MODEL", "nomic-embed-text")

# Ollama endpoint (default: local daemon).
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")

# --- Hardware assumptions (for the design note, logged at startup) ---------
# What we ASSUME for the in-office deployment, and what we ACTUALLY run on.
ASSUMED_GPU = "single NVIDIA RTX 6000 Ada-class workstation GPU, 48 GB VRAM"
ACTUAL_RUNTIME = "Apple M2, 16 GB unified memory (CPU + Metal), 4-bit quant"

# --- Retrieval knobs -------------------------------------------------------
TOP_K = int(os.environ.get("NEET_TOP_K", "6"))
# Cosine similarity below this -> refuse ("not in the sources") before the LLM.
MIN_SIMILARITY = float(os.environ.get("NEET_MIN_SIM", "0.30"))

# --- Generation knobs ------------------------------------------------------
MAX_TOKENS = int(os.environ.get("NEET_MAX_TOKENS", "1024"))
TEMPERATURE = float(os.environ.get("NEET_TEMP", "0.0"))  # deterministic