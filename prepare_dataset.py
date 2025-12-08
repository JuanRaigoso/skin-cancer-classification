import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ==== AJUSTE A TU ESTRUCTURA ====
METADATA_PATH = "HAM10000_metadata.csv"   # archivo CSV
IMAGES_DIR = "Img_gene"                   # carpeta con todas las imágenes
OUTPUT_DIR = "data"                       # donde crear train/val/test
# =================================

# 1. Cargar metadata
print("Leyendo metadata...")
df = pd.read_csv(METADATA_PATH)

# 2. Mostrar clases y cantidad
print("\nClases y cantidad de imágenes:")
print(df["dx"].value_counts())

# 3. Crear columna con ruta a la imagen
df["file_path"] = df["image_id"].apply(
    lambda x: os.path.join(IMAGES_DIR, x + ".jpg")
)

# Filtrar solo imágenes existentes
df = df[df["file_path"].apply(os.path.exists)]
print(f"\nTotal de imágenes encontradas: {len(df)}")

# 4. Separar en train (70%), val (15%), test (15%)
train_df, temp_df = train_test_split(
    df, test_size=0.30, stratify=df["dx"], random_state=42
)

val_df, test_df = train_test_split(
    temp_df, test_size=0.50, stratify=temp_df["dx"], random_state=42
)

splits = {
    "train": train_df,
    "val": val_df,
    "test": test_df
}

# 5. Crear carpetas por split y clase
for split_name, split_df in splits.items():
    split_dir = os.path.join(OUTPUT_DIR, split_name)
    for dx_class in split_df["dx"].unique():
        class_dir = os.path.join(split_dir, dx_class)
        os.makedirs(class_dir, exist_ok=True)

# 6. Copiar imágenes
for split_name, split_df in splits.items():
    print(f"\nCopiando imágenes para {split_name}...")
    for _, row in split_df.iterrows():
        src = row["file_path"]
        label = row["dx"]

        filename = os.path.basename(src)
        dst = os.path.join(OUTPUT_DIR, split_name, label, filename)

        if not os.path.exists(dst):
            shutil.copy(src, dst)

print("\n✅ Listo. Ya tienes 'data/train', 'data/val', 'data/test'")
print("Revisa: data/train/mel, data/train/nv, etc.")
