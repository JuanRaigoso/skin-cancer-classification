# =============================
# CONFIGURACIÓN GENERAL
# =============================

# Modelo base
MODEL_NAME = "EfficientNetB4"     # B0, B3, B4, B7
MODEL_VERSION = "B4_v1_advanced"

# Imagen
IMG_HEIGHT = 380
IMG_WIDTH = 380
IMG_CHANNELS = 3

# Entrenamiento
BATCH_SIZE = 16            # A100 o L4 lo soporta
EPOCHS_WARMUP = 20
EPOCHS_FINETUNE = 60

LEARNING_RATE = 1e-4
LR_FINE_TUNE = 5e-6
WEIGHT_DECAY = 1e-5

# Fine-tuning profundo
UNFROZEN_LAYERS = 300

# Regularización
DROPOUT_RATE = 0.4

# Dataset
DATA_DIR = "/content/drive/MyDrive/skin_cancer_project/data"

# MLFlow
EXPERIMENT_NAME = "skin_cancer_classification"

# Clases
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
