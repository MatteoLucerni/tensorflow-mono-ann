import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(42)

# standardized values
X = np.random.standard_normal((200, 1))
y = 2 * X + 3

plt.plot(X, y)
plt.grid(True)
plt.show()

# Sequential model using an Input layer
model = tf.keras.Sequential(
    [
        tf.keras.Input(shape=(1,)),
        tf.keras.layers.Dense(1),
    ]
)

model.summary()

model.compile(
    loss=tf.keras.losses.mean_squared_error, optimizer=tf.keras.optimizers.Adam()
)

# training
hist = model.fit(x=X, y=y, epochs=1000)

y_pred = model.predict(X)

np.c_[y, y_pred]

tf.keras.losses.mean_squared_error(y[:, 0], y_pred[:, 0])

# loss history graph
loss_hist = hist.history["loss"]

plt.plot(range(1000), loss_hist)
plt.grid(True)
plt.show()

# weights layers
weight, bias = model.layers[0].get_weights()
print("Weights:\n", weight)
print("Biases:\n", bias)
