import os
import time
os.environ["KERAS_BACKEND"] = "tensorflow"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Set memory growth to prevent GPU memory issues
physical_devices = tf.config.list_physical_devices('GPU')
if physical_devices:
    try:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
    except:
        pass

# Your data loading code
root = 'Animals'
categories = [x[0] for x in os.walk(root) if x[0]][1:]
print("Categories found:", categories)

# Create data generators
datagen = ImageDataGenerator(
    rescale=1. / 255,
    validation_split=0.3,
    horizontal_flip=True,
    rotation_range=20,
    zoom_range=0.2
)

# Create training generator
train_generator = datagen.flow_from_directory(
    root,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

# Create validation generator
validation_generator = datagen.flow_from_directory(
    root,
    target_size=(128, 128),
    batch_size=16,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Get the number of classes
num_classes = len(train_generator.class_indices)
print(f"Number of classes: {num_classes}")

# Load your pre-trained model from H5 file
start_time = time.time()
pretrained_model = keras.models.load_model('best_model.h5', compile=False)
load_time = time.time() - start_time
print(f"Model loading time: {load_time:.2f} seconds")

# Build the model by running a dummy input through it
dummy_input = tf.ones((1, 128, 128, 3))
_ = pretrained_model(dummy_input)

print("Original model summary:")
pretrained_model.summary()

# Find the index of the flatten layer
flatten_index = None
for i, layer in enumerate(pretrained_model.layers):
    if layer.name == 'flatten':
        flatten_index = i
        break

if flatten_index is None:
    raise ValueError("Flatten layer not found in the model")

# Create a new model that includes all layers up to (but not including) the flatten layer
base_model = keras.Sequential()
for layer in pretrained_model.layers[:flatten_index]:
    base_model.add(layer)

# Freeze the base model layers
base_model.trainable = False

# Build the base model with explicit input shape
base_model.build((None, 128, 128, 3))

print("Base model summary:")
base_model.summary()

# Use Functional API to create the final model
input_layer = keras.Input(shape=(128, 128, 3))
x = base_model(input_layer, training=False)  # training=False since base is frozen
x = layers.GlobalAveragePooling2D(name='transfer_gap')(x)
x = layers.Dense(128, activation='relu', name='transfer_dense_1')(x)
x = layers.Dropout(0.5, name='transfer_dropout_1')(x)
outputs = layers.Dense(num_classes, activation='softmax', name='transfer_output')(x)

model = keras.Model(inputs=input_layer, outputs=outputs)

# Compile the model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

print("Final model summary:")
model.summary()

# Add callbacks
early_stopping = keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,
    restore_best_weights=True
)

checkpoint = keras.callbacks.ModelCheckpoint(
    'best_transfer_model.h5',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Train the model and measure time
print("Starting training...")
start_time = time.time()

history = model.fit(
    train_generator,
    epochs=30,
    validation_data=validation_generator,
    callbacks=[early_stopping, checkpoint]
)

training_time = time.time() - start_time
print(f"Training time: {training_time:.2f} seconds")

# Save the final model
start_time = time.time()
model.save('final_transfer_model.h5')
save_time = time.time() - start_time
print(f"Model saving time: {save_time:.2f} seconds")

# Evaluation
print("Evaluating on validation set...")
start_time = time.time()
test_results = model.evaluate(validation_generator, verbose=1)
eval_time = time.time() - start_time
print(f"Evaluation time: {eval_time:.2f} seconds")
print(f"Test loss: {test_results[0]:.4f}, Test accuracy: {test_results[1]:.4f}")

# Print total computation time
total_time = load_time + training_time + save_time + eval_time
print(f"\nTotal computation time: {total_time:.2f} seconds")
print(f"Breakdown:")
print(f"  - Model loading: {load_time:.2f} seconds")
print(f"  - Training: {training_time:.2f} seconds")
print(f"  - Model saving: {save_time:.2f} seconds")
print(f"  - Evaluation: {eval_time:.2f} seconds")

# Plotting
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.ylim(0, 1)

plt.tight_layout()
plt.savefig('training_history.png')
plt.show()