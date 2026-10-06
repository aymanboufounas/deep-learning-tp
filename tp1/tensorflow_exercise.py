"""
TP1 - Exercise 2: XOR with TensorFlow
Manual training loop to show forward propagation and backpropagation.
"""

import tensorflow as tf

# 1) Dataset
X = tf.constant([
    [0.0, 0.0],
    [0.0, 1.0],
    [1.0, 0.0],
    [1.0, 1.0],
], dtype=tf.float32)

y = tf.constant([
    [0.0],
    [1.0],
    [1.0],
    [0.0],
], dtype=tf.float32)

# 2) Weights and biases
# 2 inputs -> 4 hidden neurons -> 1 output
W1 = tf.Variable(tf.random.normal([2, 4], stddev=0.5))
b1 = tf.Variable(tf.zeros([4]))

W2 = tf.Variable(tf.random.normal([4, 1], stddev=0.5))
b2 = tf.Variable(tf.zeros([1]))

learning_rate = 0.1
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

# 3) Forward propagation
def forward(x):
    z1 = tf.matmul(x, W1) + b1
    a1 = tf.math.tanh(z1)

    z2 = tf.matmul(a1, W2) + b2
    a2 = tf.math.sigmoid(z2)

    return a2

# 4) Training loop
for epoch in range(2000):
    with tf.GradientTape() as tape:
        predictions = forward(X)

        # Binary cross-entropy loss
        loss = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(y, predictions)
        )

    # Backpropagation: compute gradients
    variables = [W1, b1, W2, b2]
    gradients = tape.gradient(loss, variables)

    # Gradient descent / optimizer update
    optimizer.apply_gradients(zip(gradients, variables))

    if (epoch + 1) % 500 == 0:
        print(f"Epoch {epoch + 1}: loss = {loss.numpy():.6f}")

# 5) Final predictions
predictions = forward(X)

print("\nTensorFlow predictions:")
for sample, prediction in zip(X.numpy(), predictions.numpy()):
    print(f"{sample} -> {prediction[0]:.4f}")

print("\nRounded predictions:")
print(tf.cast(predictions > 0.5, tf.int32).numpy().ravel())
