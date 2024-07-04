# MNIST-Image-Classification-Project

This project demonstrates how to build, train, and evaluate a Convolutional Neural Network (CNN) to classify images from the MNIST dataset. The MNIST dataset consists of 60,000 training images and 10,000 test images of handwritten digits (0-9).

Table of Contents
Installation
Project Structure
Dataset
Model Architecture
Training the Model
Evaluating the Model
Results
Usage
License
Installation
To set up the project environment, follow these steps:

Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/mnist-classification-project.git
cd mnist-classification-project
Install the required libraries:

bash
Copy code
pip install tensorflow keras numpy pandas matplotlib
Project Structure
The project directory contains the following files:

graphql
Copy code
mnist-classification-project/
│
├── load_data.py         # Script to load and preprocess the MNIST dataset
├── build_model.py       # Script to build the CNN model
├── train_model.py       # Script to train the CNN model
├── evaluate_model.py    # Script to evaluate the trained model
├── mnist_classification_model.h5 # Trained model file (will be created after training)
└── README.md            # Project README file
Dataset
The MNIST dataset is a well-known dataset in the machine learning community. It consists of 60,000 training images and 10,000 test images of handwritten digits (0-9). The images are grayscale and have a size of 28x28 pixels.

Model Architecture
The Convolutional Neural Network (CNN) used in this project consists of the following layers:

Conv2D Layer: 32 filters, kernel size (3, 3), ReLU activation
MaxPooling2D Layer: Pool size (2, 2)
Conv2D Layer: 64 filters, kernel size (3, 3), ReLU activation
MaxPooling2D Layer: Pool size (2, 2)
Conv2D Layer: 64 filters, kernel size (3, 3), ReLU activation
Flatten Layer
Dense Layer: 64 units, ReLU activation
Dropout Layer: 50% dropout rate
Dense Layer: 10 units, softmax activation
The model is compiled using the Adam optimizer and categorical crossentropy loss function, with accuracy as the evaluation metric.

Training the Model
The training process involves the following steps:

Load and preprocess the dataset using load_data.py.
Build the CNN model using build_model.py.
Train the model using train_model.py.
python
Copy code
# train_model.py
from load_data import load_data
from build_model import build_model
import matplotlib.pyplot as plt

def train_model():
    X_train, y_train, X_test, y_test = load_data()
    model = build_model()

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))

    # Plot training & validation accuracy values
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot training & validation loss values
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.show()

    # Save the trained model
    model.save('mnist_classification_model.h5')

if __name__ == "__main__":
    train_model()
Evaluating the Model
The evaluation process involves the following steps:

Load the test dataset using load_data.py.
Load the trained model.
Evaluate the model on the test dataset.
Generate a classification report and confusion matrix.
python
Copy code
# evaluate_model.py
from load_data import load_data
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

def evaluate_model():
    X_train, y_train, X_test, y_test = load_data()
    model = load_model('mnist_classification_model.h5')

    # Evaluate the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print(f'Test loss: {loss}')
    print(f'Test accuracy: {accuracy}')

    # Predict the labels
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    y_true = np.argmax(y_test, axis=1)

    # Classification report
    print(classification_report(y_true, y_pred_classes))

    # Confusion matrix
    conf_matrix = confusion_matrix(y_true, y_pred_classes)
    plt.figure(figsize=(10, 8))
    sns.heatmap(conf_matrix, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

if __name__ == "__main__":
    evaluate_model()
Results
After training and evaluating the model, you can expect to see the following results:

Accuracy: The model should achieve a high accuracy on the MNIST test set.
Classification Report: Detailed precision, recall, and F1-score for each class.
Confusion Matrix: Visualization of the model's performance on the test set.
Usage
To use this project, follow these steps:

Train the Model:

bash
Copy code
python train_model.py
Evaluate the Model:

bash
Copy code
python evaluate_model.py
License
This project is licensed under the MIT License. See the LICENSE file for details.
