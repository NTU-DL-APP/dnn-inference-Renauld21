import numpy as np
import json

# === Load weights and architecture ===
weights = np.load(MODEL_WEIGHTS_PATH)
with open(MODEL_ARCH_PATH) as f:
    architecture = json.load(f)


# === Activation functions ===
def relu(x):
    return np.maximum(0, x)

def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)

# === Flatten ===
def flatten(x):
    return x.reshape(x.shape[0], -1)

# === Dense layer ===
def dense(x, W, b):
    return x @ W + b

# === Forward pass ===
def forward(x):
    for layer in architecture:
        lname = layer['name']
        ltype = layer['type']
        cfg = layer['config']
        wnames = layer['weights']


        if ltype == "Flatten":
            x = flatten(x)

        elif ltype == "Dense":
            W = weights[wnames[0]]
            b = weights[wnames[1]]
            x = dense(x, W, b)
            if cfg.get("activation") == "relu":
                x = relu(x)
            elif cfg.get("activation") == "softmax":
                x = softmax(x)

    return x

# === Example usage ===
# Load a dummy image (batch size 1)
# Make sure it's shape: (1, 28, 28, 1)
dummy_input = np.random.rand(1, 28*28).astype(np.float32)
output = forward(dummy_input)

print("🧠 Output probabilities:", output)
print("✅ Predicted class:", np.argmax(output, axis=-1))
