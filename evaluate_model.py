import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import mnist
from tensorflow.keras.utils import to_categorical

# Load the pre-trained model
model = load_model('mnist_v1_99.51.h5')

# Load the MNIST test dataset (you can replace this with your own test dataset)
(_, _), (X_test, y_test) = mnist.load_data()

# Preprocess the test data (you should preprocess it in the same way as the training data)
X_test = X_test.reshape(-1, 28, 28, 1)  # Reshape to match the input shape of your model
X_test = X_test.astype('float32') / 255  # Normalize pixel values to the range [0, 1]

# One-hot encode the labels
y_test = to_categorical(y_test, num_classes=10)
 
# Evaluate the model on the test data
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)

print(f'Accuracy: {accuracy * 100:.2f}%')
print(f'Loss: {loss * 100:.3f}%')