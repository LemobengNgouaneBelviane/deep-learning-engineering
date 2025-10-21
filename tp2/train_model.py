import os
import argparse
import logging
from typing import Tuple, Optional

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_model(input_shape: Tuple[int, ...], num_classes: int) -> 'Model':
    """Create and return the MNIST model."""
    try:
        import tensorflow as tf
        from tensorflow import keras
        
        model = keras.Sequential([
            keras.layers.Dense(512, activation='relu', input_shape=input_shape,
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(256, activation='relu',
                             kernel_regularizer=keras.regularizers.l2(0.001)),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        return model
    except ImportError:
        logger.warning("TensorFlow not available. Running in dry-run mode.")
        return None

def load_mnist_data() -> Tuple[Optional[tuple], Optional[tuple]]:
    """Load and preprocess MNIST data."""
    try:
        from tensorflow import keras
        import numpy as np
        
        (x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
        
        # Normalize and reshape
        x_train = x_train.astype("float32") / 255.0
        x_test = x_test.astype("float32") / 255.0
        
        # Reshape for dense layers
        x_train = x_train.reshape(60000, 784)
        x_test = x_test.reshape(10000, 784)
        
        # Split training data into train and validation
        x_val = x_train[54000:]
        y_val = y_train[54000:]
        x_train = x_train[:54000]
        y_train = y_train[:54000]
        
        return (x_train, y_train), (x_val, y_val), (x_test, y_test)
    except ImportError:
        logger.warning("TensorFlow not available. Running in dry-run mode.")
        return None, None, None

def train_model(dry_run: bool = False):
    """Main training function with MLflow tracking."""
    try:
        import mlflow
        import mlflow.tensorflow
        
        if dry_run:
            logger.info("Running in dry-run mode. Skipping actual training.")
            return
            
        EPOCHS = 5
        BATCH_SIZE = 128
        DROPOUT_RATE = 0.2
        
        with mlflow.start_run():
            # Log parameters
            mlflow.log_param("epochs", EPOCHS)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_param("dropout_rate", DROPOUT_RATE)
            
            # Load and preprocess data
            (x_train, y_train), (x_val, y_val), (x_test, y_test) = load_mnist_data()
            
            # Create and compile model
            model = create_model(input_shape=(784,), num_classes=10)
            
            if model is None:
                logger.error("Failed to create model")
                return
                
            model.compile(
                optimizer='adam',
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy']
            )
            
            # Train model
            history = model.fit(
                x_train, y_train,
                epochs=EPOCHS,
                batch_size=BATCH_SIZE,
                validation_data=(x_val, y_val)
            )
            
            # Evaluate model
            test_loss, test_acc = model.evaluate(x_test, y_test)
            logger.info(f"Précision sur les données de test: {test_acc:.4f}")
            
            # Log metrics
            mlflow.log_metric("test_accuracy", test_acc)
            
            # Save model
            model.save("mnist_model.h5")
            mlflow.keras.log_model(model, "mnist-model")
            logger.info("Modèle sauvegardé sous mnist_model.h5")
            
    except ImportError as e:
        logger.warning(f"Required libraries not available: {e}")
        logger.info("Please install required packages: pip install -r requirements.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train MNIST model with MLflow tracking')
    parser.add_argument('--dry-run', action='store_true', help='Run without training (for testing)')
    args = parser.parse_args()
    
    train_model(dry_run=args.dry_run)