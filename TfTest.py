import tensorflow as tf
import numpy as np
print("TensorFlow version:", tf.__version__)

def f():
    a = int(np.random.uniform(0, 50))
    b = int(np.random.uniform(0, 50))

    x = np.zeros((2, 50), np.int32)
    x[0][a] = 1
    x[1][b] = 1
    return x

X_train = np.array([f() for i in range(10000)])
Y_train = np.array([(np.where(X_train[i][0] == 1)[0][0]+(np.where(X_train[i][1] == 1))[0][0]) for i in range(10000)]).reshape((-1,1))
X_val = np.array([f() for i in range(1000)])
Y_val = np.array([(np.where(X_val[i][0] == 1)[0][0]+(np.where(X_val[i][1] == 1))[0][0]) for i in range(1000)]).reshape((-1,1))


model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(2, 50)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(100)
])

predictions = model(X_train[:1]).numpy()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

loss_fn(Y_train[:1], predictions).numpy()

model.compile(optimizer='adam',
              loss=loss_fn,
              metrics=['accuracy'])

model.fit(X_train, Y_train, epochs=10)
model.evaluate(X_val,  Y_val, verbose=2)