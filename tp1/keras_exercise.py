"""
TP1 - Exercise 1: XOR with Keras
Beginner-friendly neural network example.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers

# 1) Dataset: XOR
X = np.array([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=np.float32)

y = np.array([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
], dtype=np.float32)

# 2) Build the neural network
# 2 inputs -> hidden layer -> 1 output
model = keras.Sequential([
    layers.Input(shape=(2,)),
    layers.Dense(4, activation="tanh"),
    layers.Dense(1, activation="sigmoid"),
])

# 3) Configure training
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.05),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)

# 4) Train
model.fit(X, y, epochs=1000, verbose=0)

# 5) Predict
predictions = model.predict(X, verbose=0)

print("Keras predictions:")
for sample, prediction in zip(X, predictions):
    print(f"{sample} -> {prediction[0]:.4f}")

print("\nRounded predictions:")
print((predictions > 0.5).astype(int).ravel())
