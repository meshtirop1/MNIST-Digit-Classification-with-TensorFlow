from __future__ import print_function
import os
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import hy_param
from train import checkpoints_dir

# Load MNIST data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(-1, hy_param.num_input).astype('float32') / 255.0
x_test = x_test.reshape(-1, hy_param.num_input).astype('float32') / 255.0
y_train = tf.keras.utils.to_categorical(y_train, hy_param.num_classes)
y_test = tf.keras.utils.to_categorical(y_test, hy_param.num_classes)

# checkpoint_dir = os.path.join(hy_param.checkpoint_dir, "checkpoints")
checkpoint_file = tf.train.latest_checkpoint(checkpoints_dir)

if checkpoint_file is None:
    print("No checkpoint file found in directory:", checkpoints_dir)
else:
    if os.path.exists(checkpoint_file) and os.path.isfile(checkpoint_file):
        if checkpoint_file.endswith('.keras') or checkpoint_file.endswith('.weights.h5') or checkpoint_file.endswith(
                '.h5'):
            # Define the model architecture (Make sure the model architecture matches with the training phase if you don't have the saved model architecture)
            # Here we'll use a placeholder function `define_model`
            # that represents the original model architecture needed for inference.
            def define_model():
                model = tf.keras.Sequential([
                    tf.keras.layers.Dense(512, activation='relu', input_shape=(hy_param.num_input,)),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(hy_param.num_classes, activation='softmax')
                ])
                return model


            # Recreate the model architecture
            model = define_model()

            # Load the weights
            model.load_weights(checkpoint_file)

            # Prepare the test data (changing reference to correctly framed test data)
            test_data = np.array(
                [x_test[6]])  # Assuming x_test[6] is the same test data referenced in the original code

            # Get the prediction for the test data
            prediction = model.predict(test_data)

            print("Predicted digit :", prediction.argmax())
            print("Actual digit :", y_test[6].argmax())
            plt.imshow(test_data.reshape(28, 28), cmap='gray')
            plt.show()
        else:
            print("Checkpoint file format not supported. Supported formats are .keras, .weights.h5, and .h5")
    else:
        print("Checkpoint file does not exist or is not a file:", checkpoint_file)
