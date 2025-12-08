import mlflow
import mlflow.tensorflow
from config import (
    EXPERIMENT_NAME, NUM_CLASSES, CLASS_NAMES,
    EPOCHS_WARMUP, EPOCHS_FINETUNE
)
from dataloader import load_datasets, get_class_weights
from model import create_model, create_finetune_model
from utils import plot_training, plot_confusion_matrix


def main():

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog()

    with mlflow.start_run():

        # ========== DATA ==========
        train_ds, val_ds = load_datasets()
        class_weights = get_class_weights(train_ds, NUM_CLASSES)
        mlflow.log_param("class_weights", class_weights)

        # ========== TRANSFER LEARNING ==========
        model = create_model(trainable=False)
        history_warm = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_WARMUP,
            class_weight=class_weights
        )

        plot_training(history_warm, "warmup_training_B3.png")
        mlflow.log_artifact("warmup_training_B3.png")

        # ========== FINE TUNING ==========
        model = create_finetune_model()
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINETUNE,
            class_weight=class_weights
        )

        plot_training(history_ft, "finetune_training_B3.png")
        mlflow.log_artifact("finetune_training_B3.png")

        # ========== MATRIZ DE CONFUSIÓN ==========
        plot_confusion_matrix(model, val_ds, CLASS_NAMES, "confusion_matrix_B3.png")
        mlflow.log_artifact("confusion_matrix_B3.png")

        # ========== GUARDAR MODELO ==========
        model.save("model_final_B3.h5")
        mlflow.log_artifact("model_final_B3.h5")


if __name__ == "__main__":
    main()
