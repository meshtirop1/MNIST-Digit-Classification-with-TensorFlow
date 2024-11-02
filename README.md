# MNIST-Digit-Classification-with-TensorFlow
This project implements a neural network model using TensorFlow to classify handwritten digits from the MNIST dataset. It demonstrates modular code organization, making it simple to adjust hyperparameters, retrain the model, or use it for predictions.

Project Structure
hy_param.py: Contains hyperparameters and configurations, such as learning rate, batch size, and checkpoint paths.
model.py: Defines the neural network architecture with two hidden layers and a softmax output layer.
train.py: Handles data loading, training, checkpoint saving, and model evaluation.
inference.py: Loads the saved model weights for predictions, displaying the predicted and actual labels along with the corresponding image.
Key Features
Modular Design: Separates configuration, model architecture, training, and inference for easy maintenance.
Checkpointing: Saves model weights at each epoch, allowing for training resumption or real-time inference.
Visualization: Displays predicted labels and the test image for quick evaluation.

git clone https://github.com/meshtirop1/mnist-digit-classification.git
cd mnist-digit-classification
