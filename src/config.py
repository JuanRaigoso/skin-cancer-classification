MODEL_NAME = "EfficientNetB3"
MODEL_VERSION = "B3_v2"

IMG_HEIGHT = 300
IMG_WIDTH = 300
IMG_CHANNELS = 3

BATCH_SIZE = 24

EPOCHS_WARMUP = 30
EPOCHS_FINETUNE = 40       # aumentamos fine-tuning

UNFROZEN_LAYERS = 150      # deeper fine-tune
LEARNING_RATE = 1e-4
LR_FINE_TUNE = 3e-6        # LR óptimo para B3_v2

WEIGHT_DECAY = 1e-5
DROPOUT_RATE = 0.4

DATA_DIR = "/content/drive/MyDrive/skin_cancer_project/data"

EXPERIMENT_NAME = "skin_cancer_classification"

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
NUM_CLASSES = len(CLASS_NAMES)
