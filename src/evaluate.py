import tensorflow as tf
from config import CLASS_NAMES, DATA_DIR
from dataloader import load_datasets
from utils import evaluate_model_metrics

def main():

    print("Cargando modelo entrenado...")
    model = tf.keras.models.load_model("model_final_B3_v2.h5")

    print("Cargando dataset de validación...")
    _, val_ds = load_datasets()

    print("Evaluando métricas clínicas (precision, recall, f1)...")
    report = evaluate_model_metrics(model, val_ds, CLASS_NAMES, "classification_report_B3_v2.json")

    print("\n=== RESULTADOS ===\n")
    for cls, metrics in report.items():
        print(cls, metrics)

    print("\nArchivo guardado como classification_report_B3_v2.json")

if __name__ == "__main__":
    main()
