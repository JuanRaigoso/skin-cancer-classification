import mlflow
import mlflow.tensorflow

from config import (
    EXPERIMENT_NAME, NUM_CLASSES, CLASS_NAMES,
    EPOCHS_WARMUP, EPOCHS_FINETUNE,
    MODEL_VERSION, UNFROZEN_LAYERS, LR_FINE_TUNE
)

from dataloader import load_datasets, get_class_weights
from model import create_model, create_finetune_model
from utils import plot_training, plot_confusion_matrix, plot_roc_curve


def main():

    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog()

    with mlflow.start_run():

        # Registrar parámetros clave del experimento
        mlflow.log_param("model_version", MODEL_VERSION)
        mlflow.log_param("unfrozen_layers", UNFROZEN_LAYERS)
        mlflow.log_param("lr_fine_tune", LR_FINE_TUNE)
        mlflow.log_param("epochs_warmup", EPOCHS_WARMUP)
        mlflow.log_param("epochs_finetune", EPOCHS_FINETUNE)

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

        warmup_plot = f"warmup_training_{MODEL_VERSION}.png"
        plot_training(history_warm, warmup_plot)
        mlflow.log_artifact(warmup_plot)

        # ========== FINE TUNING ==========
        model = create_finetune_model()
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINETUNE,
            class_weight=class_weights
        )

        finetune_plot = f"finetune_training_{MODEL_VERSION}.png"
        plot_training(history_ft, finetune_plot)
        mlflow.log_artifact(finetune_plot)

        # ========== MATRIZ DE CONFUSIÓN ==========
        cm_plot = f"confusion_matrix_{MODEL_VERSION}.png"
        plot_confusion_matrix(model, val_ds, CLASS_NAMES, cm_plot)
        mlflow.log_artifact(cm_plot)

        # ========== CURVA ROC Y AUC ==========
        roc_plot = f"roc_curve_{MODEL_VERSION}.png"
        auc_scores = plot_roc_curve(model, val_ds, CLASS_NAMES, roc_plot)
        mlflow.log_artifact(roc_plot)
        mlflow.log_param("auc_scores", auc_scores)

        # ========== GUARDAR MODELO ==========
        model_path = f"model_final_{MODEL_VERSION}.h5"
        model.save(model_path)
        mlflow.log_artifact(model_path)


if __name__ == "__main__":
    main()
