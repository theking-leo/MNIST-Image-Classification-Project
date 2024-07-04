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
