import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import hy_param
import model

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, hy_param.num_input).astype('float32') / 255.0
x_test = x_test.reshape(-1, hy_param.num_input).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, hy_param.num_classes)
y_test = tf.keras.utils.to_categorical(y_test, hy_param.num_classes)

# Checkpoints directory
checkpoints_dir = os.path.abspath(os.path.join(hy_param.checkpoints_dir, "checkpoints"))
if not os.path.exists(checkpoints_dir):
    os.makedirs(checkpoints_dir)

# Update checkpoint prefix to follow the required convention
checkpoint_prefix = os.path.join(checkpoints_dir, "model.weights.h5")

# Callback for saving model
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True,
    save_freq='epoch',
    verbose=1
)

# Training the model
model.model.fit(x_train, y_train,
                epochs=hy_param.num_steps,
                batch_size=hy_param.batch_size,
                validation_data=(x_test, y_test),
                callbacks=[checkpoint_callback],
                verbose=1)

# Evaluate the model
loss, accuracy = model.model.evaluate(x_test, y_test, verbose=1)
print(f"Testing accuracy: {accuracy}")
