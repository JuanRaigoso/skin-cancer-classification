import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from dataloader import load_datasets
from config import MODEL_VERSION

MODEL_PATH = f"model_final_{MODEL_VERSION}.keras"

def nll_with_T(T, logits, y_true):
    # logits: [N,C] sin softmax
    scaled = logits / T
    log_probs = scaled - tf.reduce_logsumexp(scaled, axis=1, keepdims=True)
    nll = -tf.reduce_mean(tf.reduce_sum(y_true * log_probs, axis=1))
    return nll

def main():
    model = load_model(MODEL_PATH, compile=False)
    _, val_ds, _ = load_datasets()

    # recolectar logits y y_true
    y_true_list = []
    logits_list = []

    for x, y in val_ds:
        # salida del modelo es softmax, sacamos "logits aprox" con log(p)
        p = model.predict(x, verbose=0)
        logits = np.log(np.clip(p, 1e-8, 1.0))
        y_true_list.append(y.numpy())
        logits_list.append(logits)

    y_true = tf.constant(np.vstack(y_true_list), dtype=tf.float32)
    logits = tf.constant(np.vstack(logits_list), dtype=tf.float32)

    T = tf.Variable(1.0, dtype=tf.float32)
    opt = tf.keras.optimizers.Adam(learning_rate=0.05)

    for step in range(200):
        with tf.GradientTape() as tape:
            loss = nll_with_T(T, logits, y_true)
        grads = tape.gradient(loss, [T])
        opt.apply_gradients(zip(grads, [T]))
        T.assign(tf.clip_by_value(T, 0.5, 10.0))
        if step % 20 == 0:
            print(f"step={step}  T={T.numpy():.4f}  NLL={loss.numpy():.4f}")

    print("\n✅ Temperature final:", float(T.numpy()))
    with open("temperature.txt", "w") as f:
        f.write(str(float(T.numpy())))

if __name__ == "__main__":
    main()
