"""Central configuration. Loads secrets from .env — never hardcode keys here."""
import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = BASE_DIR / "prompts"
DATA_DIR = BASE_DIR / "data"

DATASET_URL = "https://raw.githubusercontent.com/drmuskangarg/lonesomeness_dataset/main/Lonesomenessdataset.csv"
DATASET_PATH = DATA_DIR / "Lonesomenessdataset.csv"
DEV_RESULTS_PATH = DATA_DIR / "clinscale_dev_results.csv"
REP_DEV_PATH = DATA_DIR / "dev_representative.csv"
FULL_CHECKPOINT_PATH = DATA_DIR / "clinscale_full_checkpoint.csv"

MODEL_NAME = os.environ.get("GEMINI_MODEL", "gemini-3.5-flash-lite")
SEED = 42

# Few-shot calibration: number of train-split nearest neighbors injected into
# the prompt (0 disables retrieval and runs the zero-shot prompt).
FEWSHOT_K = int(os.environ.get("FEWSHOT_K", "6"))

# "top" = k nearest overall; "balanced" = k/2 nearest per class (contrastive).
FEWSHOT_MODE = os.environ.get("FEWSHOT_MODE", "top")

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")


def require_api_key() -> str:
    """Return the Gemini API key, or raise with setup instructions if missing."""
    if not GEMINI_API_KEY:
        raise RuntimeError(
            "GEMINI_API_KEY is not set. Copy .env.example to .env and add your key."
        )
    return GEMINI_API_KEY
