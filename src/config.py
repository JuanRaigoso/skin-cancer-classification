# =============================
# CONFIGURACIÓN GENERAL
# =============================

SEED = 42

# Modelo base
MODEL_NAME = "EfficientNetV2M"
MODEL_VERSION = "V2M_v2_unfreeze300"

# Imagen (EffNetV2M suele ir bien con 480, pero puedes bajar a 384 si falta VRAM)
IMG_HEIGHT = 480
IMG_WIDTH = 480
IMG_CHANNELS = 3

# Entrenamiento
BATCH_SIZE = 16            # ajusta según GPU (L4/A100 ok, T4 baja a 8)
EPOCHS_WARMUP = 20
EPOCHS_FINETUNE = 50

LEARNING_RATE = 2e-4
LR_FINE_TUNE = 2e-5
WEIGHT_DECAY = 1e-5

# Fine-tuning: cuántas capas finales desbloquear
UNFROZEN_LAYERS = 300

# Regularización
DROPOUT_RATE = 0.3
LABEL_SMOOTHING = 0.03
GRAD_CLIPNORM = 1.0

# Aug robusta
MIXUP_PROB = 0.25
CUTMIX_PROB = 0.25
AUG_PROB = 1.0  # prob de aplicar augmentations keras
RANDOM_ERASE_PROB = 0.25
BLUR_PROB = 0.15

# Dataset
DATA_DIR = "/content/drive/MyDrive/skin_cancer_project/data"

# MLFlow
EXPERIMENT_NAME = "skin_cancer_classification"

# Clases
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)

# Inferencia
USE_TTA = True
TTA_SAMPLES = 8  # 4-16
