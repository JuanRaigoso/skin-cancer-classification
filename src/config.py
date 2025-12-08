
# Modelo base
MODEL_NAME = "EfficientNetB3"     # opciones: EfficientNetB0, EfficientNetB3, EfficientNetB7

# Imagen
IMG_HEIGHT = 300
IMG_WIDTH = 300
IMG_CHANNELS = 3

# Entrenamiento
BATCH_SIZE = 24
EPOCHS_WARMUP = 30        # entrenamiento con backbone congelado
EPOCHS_FINETUNE = 30      # fine tuning con backbone descongelado
LEARNING_RATE = 1e-4
LEARNING_RATE_FINETUNE = 1e-5
OPTIMIZER = "adamw"
WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.4

# Dataset
DATA_DIR = "../data"

# MLflow
EXPERIMENT_NAME = "skin_cancer_classification"

# Clases del dataset
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
