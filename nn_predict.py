#!/usr/bin/env python3
import tensorflow as tf
import numpy as np
import json

YOUR_MODEL_NAME = 'fashion_mnist'
TF_MODEL_PATH = f'{YOUR_MODEL_NAME}.h5'
MODEL_WEIGHTS_PATH = f'{YOUR_MODEL_NAME}.npz'
MODEL_ARCH_PATH = f'{YOUR_MODEL_NAME}.json'

# === Step 1: Train and save the model ===
print("🔧 Training and saving the model...")
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
X_train, X_test = X_train / 255.0, X_test / 255.0

model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
model.save(TF_MODEL_PATH)
print(f"✅ Model saved as {TF_MODEL_PATH}")

# === Step 2: Conversion function ===
def convert_model(model_path, out_weights_path, out_arch_path):
    """
    Converts a .h5 Keras model to .npz weights and .json architecture.
    """
    model = tf.keras.models.load_model(model_path)
    print(f"✅ Model loaded from {model_path}")

    # Save weights
    params = {}
    for layer in model.layers:
        weights = layer.get_weights()
        if weights:
            for i, w in enumerate(weights):
                param_name = f"{layer.name}_{i}"
                params[param_name] = w
    np.savez(out_weights_path, **params)
    print(f"✅ Weights saved to {out_weights_path}")

    # Save architecture
    arch = []
    for layer in model.layers:
        config = layer.get_config()
        arch.append({
            "name": layer.name,
            "type": layer.__class__.__name__,
            "config": config,
            "weights": [f"{layer.name}_{i}" for i in range(len(layer.get_weights()))]
        })
    with open(out_arch_path, "w") as f:
        json.dump(arch, f, indent=2)
    print(f"✅ Architecture saved to {out_arch_path}")

# === Step 3: Call conversion function ===
print("\n🔄 Converting model to NumPy format...")
convert_model(TF_MODEL_PATH, MODEL_WEIGHTS_PATH, MODEL_ARCH_PATH)
print("✅ Conversion complete!")

# === Step 4: Example Inference in NumPy ===
# Load weights and architecture
weights_data = np.load(MODEL_WEIGHTS_PATH)
weights = {k: weights_data[k] for k in weights_data}

with open(MODEL_ARCH_PATH) as f:
    architecture = json.load(f)

# Activation functions
def relu(x): return np.maximum(0, x)
def softmax(x): e = np.exp(x - np.max(x, axis=-1, keepdims=True)); return e / np.sum(e, axis=-1, keepdims=True)
def flatten(x): return x.reshape(x.shape[0], -1)
def dense(x, W, b): return x @ W + b

# Forward pass
def forward(x):
    for layer in architecture:
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']
        if ltype == "Flatten":
            x = flatten(x)
        elif ltype == "Dense":
            W, b = weights[wnames[0]], weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)
    return x

# Example usage
dummy_input = np.random.rand(1, 28*28).astype(np.float32)
output = forward(dummy_input)
print("🧠 Output probabilities:", output)
print("✅ Predicted class:", np.argmax(output, axis=-1))
