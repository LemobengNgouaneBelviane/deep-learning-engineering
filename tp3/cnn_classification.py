import tensorflow as tf
from tensorflow import keras
import numpy as np

# --------------------------------------------------
# Part 1: Data preparation
# --------------------------------------------------

def prepare_cifar10():
    (x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()
    NUM_CLASSES = 10
    INPUT_SHAPE = x_train.shape[1:]

    # Normalize
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    # One-hot
    y_train = keras.utils.to_categorical(y_train, num_classes=NUM_CLASSES)
    y_test = keras.utils.to_categorical(y_test, num_classes=NUM_CLASSES)

    print(f"Input data shape: {INPUT_SHAPE}")
    print(f"y_train shape after one-hot: {y_train.shape}")
    print(f"y_test shape after one-hot: {y_test.shape}")

    return (x_train, y_train), (x_test, y_test), INPUT_SHAPE, NUM_CLASSES

# --------------------------------------------------
# Part 2: Basic CNN
# --------------------------------------------------

def build_basic_cnn(input_shape, num_classes):
    model = keras.Sequential([
        # Conv 1
        keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        # Conv 2
        keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        keras.layers.MaxPooling2D(pool_size=(2, 2)),

        keras.layers.Flatten(),
        keras.layers.Dense(512, activation='relu'),
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

# --------------------------------------------------
# Part 2.2: Residual block and small ResNet
# --------------------------------------------------

def residual_block(x, filters, kernel_size=(3,3), stride=1):
    # Main path
    y = keras.layers.Conv2D(filters, kernel_size, strides=stride, padding='same', activation='relu')(x)
    y = keras.layers.Conv2D(filters, kernel_size, padding='same')(y)

    # Skip path: adjust dimensions when stride > 1 or channels differ
    if stride > 1 or x.shape[-1] != filters:
        x = keras.layers.Conv2D(filters, (1,1), strides=stride, padding='same')(x)

    # Add and activate
    z = keras.layers.Add()([x, y])
    z = keras.layers.Activation('relu')(z)
    return z


def build_small_resnet(input_shape, num_classes):
    input_layer = keras.Input(shape=input_shape)
    x = residual_block(input_layer, 32)
    x = residual_block(x, 64, stride=2)
    x = residual_block(x, 64)

    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    output = keras.layers.Dense(num_classes, activation='softmax')(x)

    model = keras.Model(inputs=input_layer, outputs=output)
    return model

# --------------------------------------------------
# Part 3: Style transfer prep (skeleton)
# --------------------------------------------------

def create_vgg_extractor():
    # Load VGG16 without top layers
    vgg = keras.applications.VGG16(include_top=False, weights='imagenet')
    vgg.trainable = False

    content_layers = ['block5_conv2']
    style_layers = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1', 'block5_conv1']
    outputs = [vgg.get_layer(name).output for name in (style_layers + content_layers)]
    extractor = keras.Model(inputs=vgg.input, outputs=outputs)
    return extractor, style_layers, content_layers

# --------------------------------------------------
# Minimal run: train small CNN for a few epochs (smoke)
# --------------------------------------------------

def run_smoke_train(epochs=1, small_subset=True):
    (x_train, y_train), (x_test, y_test), INPUT_SHAPE, NUM_CLASSES = prepare_cifar10()

    if small_subset:
        # Use a small subset to keep runtime low
        x_train = x_train[:2000]
        y_train = y_train[:2000]
        x_test = x_test[:500]
        y_test = y_test[:500]

    model = build_basic_cnn(INPUT_SHAPE, NUM_CLASSES)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    history = model.fit(x_train, y_train, batch_size=64, epochs=epochs, validation_split=0.1, verbose=2)

    loss, acc = model.evaluate(x_test, y_test, verbose=0)
    print(f"Test accuracy: {acc:.4f}")
    return history, (loss, acc)

# --------------------------------------------------
# CLI
# --------------------------------------------------

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='TP3: CNN and Computer Vision - smoke runner')
    parser.add_argument('--smoke', action='store_true', help='Run a short smoke training')
    parser.add_argument('--resnet', action='store_true', help='Build and summarize small ResNet')
    args = parser.parse_args()

    if args.resnet:
        (_, _), (_, _), INPUT_SHAPE, NUM_CLASSES = prepare_cifar10()
        rmodel = build_small_resnet(INPUT_SHAPE, NUM_CLASSES)
        rmodel.summary()

    if args.smoke:
        run_smoke_train(epochs=2)

    if not args.smoke and not args.resnet:
        print('No action requested. Use --smoke or --resnet')
