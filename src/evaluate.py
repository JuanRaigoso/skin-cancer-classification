import tensorflow as tf
from config import CLASS_NAMES, MODEL_VERSION
from dataloader import load_datasets
from utils import evaluate_model_metrics

# IMPORTANTE: focal_loss debe existir para cargar el modelo
from model import focal_loss

def main():

    model_path = f"model_final_{MODEL_VERSION}.keras"

    print("📦 Cargando modelo entrenado...")
    model = tf.keras.models.load_model(
        model_path,
        compile=False   # NO recompilamos para evaluación
    )

    print("📂 Cargando dataset de validación...")
    _, val_ds, _ = load_datasets()

    print("📊 Evaluando métricas clínicas (precision, recall, f1)...")
    report_file = f"classification_report_{MODEL_VERSION}.json"
    report = evaluate_model_metrics(
        model,
        val_ds,
        CLASS_NAMES,
        report_file
    )

    print("\n=== RESULTADOS ===\n")
    for cls, metrics in report.items():
        if isinstance(metrics, dict):
            print(f"{cls:>10} → "
                  f"Precision: {metrics.get('precision', 0):.3f} | "
                  f"Recall: {metrics.get('recall', 0):.3f} | "
                  f"F1: {metrics.get('f1-score', 0):.3f}")

    print(f"\n✅ Archivo guardado como {report_file}")

if __name__ == "__main__":
    main()
