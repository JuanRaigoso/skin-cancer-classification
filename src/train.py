import mlflow
import mlflow.tensorflow
from config import (
    EXPERIMENT_NAME, NUM_CLASSES, CLASS_NAMES,
    EPOCHS_WARMUP, EPOCHS_FINETUNE, MODEL_VERSION
)
from dataloader import load_datasets, get_class_weights
from model import create_model, create_finetune_model
from utils import plot_training, plot_confusion_matrix, plot_roc_curve, evaluate_model_metrics

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog()

    with mlflow.start_run():

        train_ds, val_ds = load_datasets()
        class_weights = get_class_weights(train_ds, NUM_CLASSES)
        mlflow.log_param("class_weights", class_weights)

        model = create_model(trainable=False)
        history_warm = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_WARMUP,
            class_weight=class_weights
        )

        warmup_plot = f"warmup_{MODEL_VERSION}.png"
        plot_training(history_warm, warmup_plot)
        mlflow.log_artifact(warmup_plot)

        model = create_finetune_model()
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINETUNE,
            class_weight=class_weights
        )

        finetune_plot = f"finetune_{MODEL_VERSION}.png"
        plot_training(history_ft, finetune_plot)
        mlflow.log_artifact(finetune_plot)

        cm_plot = f"cm_{MODEL_VERSION}.png"
        plot_confusion_matrix(model, val_ds, CLASS_NAMES, cm_plot)
        mlflow.log_artifact(cm_plot)

        roc_plot = f"roc_{MODEL_VERSION}.png"
        auc_scores = plot_roc_curve(model, val_ds, CLASS_NAMES, roc_plot)
        mlflow.log_param("AUC_scores", auc_scores)
        mlflow.log_artifact(roc_plot)

        report_file = f"classification_report_{MODEL_VERSION}.json"
        report = evaluate_model_metrics(model, val_ds, CLASS_NAMES, report_file)
        mlflow.log_artifact(report_file)

        model_file = f"model_final_{MODEL_VERSION}.h5"
        model.save(model_file)
        mlflow.log_artifact(model_file)


if __name__ == "__main__":
    main()
