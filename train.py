import tensorflow as tf
from tensorflow.keras import layers, models
import os
import matplotlib.pyplot as plt

# ==============================
# 1. Dataset Preparation
# ==============================

# Paths
train_dir = "Data"
val_dir   = "validate"

# Image size and batch
IMG_SIZE = (1024, 1024)
BATCH_SIZE = 8

# Data loading with augmentation
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    color_mode="grayscale"
)

num_classes = len(train_ds.class_names)
print("Classes:", train_ds.class_names)

# Normalize [-1,1]
normalization_layer = layers.Rescaling(1./127.5, offset=-1)

train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

# ============================
# 2. CNN Model Definition
# ============================

inputs = layers.Input(shape=(1024, 1024, 1))

x = layers.Conv2D(8, kernel_size=2, padding="same")(inputs)
x = layers.ReLU()(x)
x = layers.LayerNormalization()(x)

x = layers.Conv2D(16, kernel_size=2, padding="same")(x)
x = layers.ReLU()(x)
x = layers.LayerNormalization()(x)

x = layers.Conv2D(32, kernel_size=2, padding="same")(x)
x = layers.ReLU()(x)
x = layers.LayerNormalization()(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs, x)

model.summary()

# ============================
# 3. Training Setup
# ============================
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=50,
    verbose=1
)

# ============================
# 4. Save Model in .h5 and .tflite
# ============================

# Save standard model
model.save("cnn2d_model_2.h5")

# Convert to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("cnn2d_model_2.tflite", "wb") as f:
    f.write(tflite_model)

print("✅ Model saved as cnn2d_model.h5 and cnn2d_model.tflite")

plt.figure(figsize=(10, 5))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig('training_accuracy.png')
plt.close()

plt.figure(figsize=(10, 5))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('training_loss.png')

# End of script

