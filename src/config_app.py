# src/config_app.py
from pathlib import Path
import json

BASE_DIR = Path(__file__).resolve().parent.parent
ARTIFACTS_DIR = BASE_DIR / "artifacts"
ASSETS_DIR = BASE_DIR / "assets"

MODEL_PATH = ARTIFACTS_DIR / "model.keras"
REPORT_PATH = ARTIFACTS_DIR / "classification_report.json"
LABELS_PATH = ARTIFACTS_DIR / "labels.json"
TEMP_PATH = ARTIFACTS_DIR / "temperature.txt"

DEFAULT_CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
DEFAULT_IMG_HEIGHT = 480
DEFAULT_IMG_WIDTH = 480

def load_labels_config():
    """
    Retorna:
      class_names: list[str]
      img_height: int
      img_width: int
      model_name: str|None
      model_version: str|None
    """
    class_names = DEFAULT_CLASS_NAMES
    img_height = DEFAULT_IMG_HEIGHT
    img_width = DEFAULT_IMG_WIDTH
    model_name = None
    model_version = None

    if LABELS_PATH.exists():
        try:
            data = json.loads(LABELS_PATH.read_text(encoding="utf-8"))
            class_names = data.get("class_names", class_names)
            img_height = int(data.get("img_height", img_height))
            img_width = int(data.get("img_width", img_width))
            model_name = data.get("model_name")
            model_version = data.get("model_version")
        except Exception:
            # Si labels.json está corrupto, usamos defaults
            pass

    return class_names, img_height, img_width, model_name, model_version

def load_temperature():
    if TEMP_PATH.exists():
        try:
            return float(TEMP_PATH.read_text(encoding="utf-8").strip())
        except Exception:
            return None
    return None
