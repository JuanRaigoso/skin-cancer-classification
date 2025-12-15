import mlflow
import mlflow.tensorflow
import tensorflow as tf

from config import (
    EXPERIMENT_NAME, NUM_CLASSES, CLASS_NAMES,
    EPOCHS_WARMUP, EPOCHS_FINETUNE, MODEL_VERSION,
    LABEL_SMOOTHING
)
from dataloader import load_datasets, get_class_weights_from_raw
from model import build_model, enable_finetuning
from utils import plot_training, plot_confusion_matrix, plot_roc_curve, evaluate_model_metrics

def label_smooth_ds(ds, smoothing):
    if smoothing <= 0:
        return ds

    def _smooth(x, y):
        k = tf.cast(tf.shape(y)[-1], tf.float32)
        y2 = (1.0 - smoothing) * y + (smoothing / k)
        return x, y2

    return ds.map(_smooth, num_parallel_calls=tf.data.AUTOTUNE)

def make_callbacks(model_version):
    ckpt_path = f"best_{model_version}.keras"
    return [
        tf.keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor="val_loss", save_best_only=True, mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=8, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=3, min_lr=1e-7, verbose=1
        ),
    ]

def main():
    mlflow.set_experiment(EXPERIMENT_NAME)
    mlflow.tensorflow.autolog()

    with mlflow.start_run():

        train_ds, val_ds, train_raw = load_datasets()

        # Label smoothing (antes de MixUp/CutMix ya hay soft labels, pero smoothing aún ayuda un poco)
        train_ds = label_smooth_ds(train_ds, LABEL_SMOOTHING)
        val_ds = label_smooth_ds(val_ds, LABEL_SMOOTHING)

        # Class weights desde RAW (sin MixUp)
        class_weights = get_class_weights_from_raw(train_raw, NUM_CLASSES)
        mlflow.log_param("class_weights", class_weights)

        callbacks = make_callbacks(MODEL_VERSION)

        # 1) Warmup
        model = build_model()
        history_warm = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_WARMUP,
            class_weight=class_weights,
            callbacks=callbacks
        )

        warmup_plot = f"warmup_{MODEL_VERSION}.png"
        plot_training(history_warm, warmup_plot)
        mlflow.log_artifact(warmup_plot)

        # 2) Fine-tuning correcto (MISMO modelo)
        model = enable_finetuning(model)
        history_ft = model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=EPOCHS_FINETUNE,
            class_weight=class_weights,
            callbacks=callbacks
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
        _ = evaluate_model_metrics(model, val_ds, CLASS_NAMES, report_file)
        mlflow.log_artifact(report_file)

        # Guardar final
        model_file = f"model_final_{MODEL_VERSION}.keras"
        model.save(model_file)
        mlflow.log_artifact(model_file)

if __name__ == "__main__":
    main()
