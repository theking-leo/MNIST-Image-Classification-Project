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

    # Classification report^
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
