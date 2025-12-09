import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
import json

# ======================================================
#   TRAINING CURVES
# ======================================================

def plot_training(history, filename="training.png"):
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history.history["accuracy"], label="Train")
    plt.plot(history.history["val_accuracy"], label="Val")
    plt.title("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title("Loss")
    plt.legend()

    plt.savefig(filename)
    plt.close()

# ======================================================
#   CONFUSION MATRIX
# ======================================================

def plot_confusion_matrix(model, val_ds, class_names, filename="cm.png"):
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

# ======================================================
#   ROC CURVE
# ======================================================

def plot_roc_curve(model, val_ds, class_names, filename="roc.png"):
    n_classes = len(class_names)
    y_true = []
    y_prob = []

    for x, y in val_ds:
        preds = model.predict(x)
        y_true.append(y.numpy())
        y_prob.append(preds)

    y_true = np.vstack(y_true)
    y_prob = np.vstack(y_prob)

    plt.figure(figsize=(10, 8))
    auc_scores = {}

    for i in range(n_classes):
        fpr, tpr, _ = roc_curve(y_true[:, i], y_prob[:, i])
        auc_i = auc(fpr, tpr)
        auc_scores[class_names[i]] = float(auc_i)
        plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={auc_i:.3f})")

    plt.plot([0, 1], [0, 1], "k--")
    plt.title("ROC Curve")
    plt.legend()
    plt.savefig(filename)
    plt.close()

    return auc_scores

# ======================================================
#   CLASSIFICATION REPORT (Precision, Recall, F1)
# ======================================================

def evaluate_model_metrics(model, val_ds, class_names, filename="report.json"):

    y_true = []
    y_pred = []

    for x, y in val_ds:
        preds = model.predict(x)
        y_true.extend(np.argmax(y.numpy(), axis=1))
        y_pred.extend(np.argmax(preds, axis=1))

    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        output_dict=True
    )

    with open(filename, "w") as f:
        json.dump(report, f, indent=4)

    return report
