import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, roc_curve, auc
import seaborn as sns
import tensorflow as tf


# =========================================================
#                GRÁFICA DE ENTRENAMIENTO
# =========================================================
def plot_training(history, filename="training_curves.png"):
    plt.figure(figsize=(12, 5))

    # Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    # Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.savefig(filename)
    plt.close()


# =========================================================
#                MATRIZ DE CONFUSIÓN
# =========================================================
def plot_confusion_matrix(model, val_ds, class_names, filename="confusion_matrix.png"):
    y_true = []
    y_pred = []

    for x, y in val_ds:
        preds = model.predict(x)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=class_names, yticklabels=class_names,
                cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(filename)
    plt.close()


# =========================================================
#                CURVA ROC Y AUC MULTICLASE
# =========================================================
def plot_roc_curve(model, val_ds, class_names, filename="roc_curve.png"):
    """
    Genera curva ROC por clase y calcula:
        - AUC de cada clase
        - AUC macro
        - AUC micro

    Guarda un PNG con las curvas ROC.
    """

    y_true = []
    y_prob = []

    # Obtener predicciones
    for x, y in val_ds:
        preds = model.predict(x)
        y_prob.extend(preds)
        y_true.extend(y.numpy())

    y_true = np.array(y_true)
    y_prob = np.array(y_prob)

    # One-hot a índices
    y_true_idx = np.argmax(y_true, axis=1)

    # Crear figura
    plt.figure(figsize=(10, 8))

    n_classes = len(class_names)
    auc_scores = {}

    # ----------------------------
    # ROC y AUC por clase
    # ----------------------------
    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        roc_auc = auc(fpr, tpr)
        auc_scores[class_names[i]] = roc_auc

        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.3f})")

    # ----------------------------
    # AUC micro
    # ----------------------------
    fpr_micro, tpr_micro, _ = roc_curve(y_true.ravel(), y_prob.ravel())
    auc_micro = auc(fpr_micro, tpr_micro)
    auc_scores["AUC_micro"] = auc_micro

    # ----------------------------
    # AUC macro
    # ----------------------------
    auc_macro = np.mean(list(auc_scores.values())[:-1])
    auc_scores["AUC_macro"] = auc_macro

    # Plot settings
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Multiclass)")
    plt.legend(loc="lower right")
    plt.grid(True)

    # Guardar figura
    plt.savefig(filename)
    plt.close()

    return auc_scores
