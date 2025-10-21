import tensorflow as tf
from tensorflow import keras
import numpy as np
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_and_preprocess_data():
    """Charge et prétraite les données MNIST."""
    # Chargement du jeu de données MNIST
    logger.info("Chargement des données MNIST...")
    (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

    # Normalisation des données
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Redimensionnement des images pour les réseaux fully-connected
    x_train = x_train.reshape(60000, 784)
    x_test = x_test.reshape(10000, 784)

    # Split train/validation
    validation_split = 0.1
    split_idx = int(x_train.shape[0] * (1 - validation_split))
    
    x_val = x_train[split_idx:]
    y_val = y_train[split_idx:]
    x_train = x_train[:split_idx]
    y_train = y_train[:split_idx]

    return (x_train, y_train), (x_val, y_val), (x_test, y_test)

def create_model():
    """Crée et retourne le modèle de base."""
    logger.info("Création du modèle...")
    model = keras.Sequential([
        # Couche d'entrée: 784 neurones (28x28 pixels)
        keras.layers.Dense(512, activation='relu', input_shape=(784,)),
        # Dropout pour réduire l'overfitting
        keras.layers.Dropout(0.2),
        # Couche de sortie: 10 neurones (10 chiffres)
        keras.layers.Dense(10, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model

def train_and_evaluate():
    """Entraîne et évalue le modèle."""
    # Chargement et prétraitement des données
    (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_and_preprocess_data()

    # Création du modèle
    model = create_model()

    # Configuration de l'entraînement
    EPOCHS = 5
    BATCH_SIZE = 128

    logger.info("Début de l'entraînement...")
    # Entraînement du modèle
    history = model.fit(
        x_train,
        y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        validation_data=(x_val, y_val)
    )

    # Évaluation du modèle
    logger.info("Évaluation du modèle...")
    test_loss, test_acc = model.evaluate(x_test, y_test)
    logger.info(f"Précision sur les données de test: {test_acc:.4f}")

    # Sauvegarde du modèle
    model.save("mnist_model.h5")
    logger.info("Modèle sauvegardé sous mnist_model.h5")

    return history, test_acc

if __name__ == "__main__":
    train_and_evaluate()