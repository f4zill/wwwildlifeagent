import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import json

# Dataset directory
dataset_dir = "ds"

# Hyperparameters
img_size = 224
batch_size = 32
epochs = 10

print("Loading dataset...")

train_datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2,
    rotation_range=20,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

train_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset="training"
)

val_gen = train_datagen.flow_from_directory(
    dataset_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset="validation"
)

num_classes = len(train_gen.class_indices)
print("Detected Classes: ", train_gen.class_indices)

# Save class indices
with open("class_indices.json", "w") as f:
    json.dump(train_gen.class_indices, f)
print("Saved class_indices.json")

print("Loading MobileNetV2 base model...")

base_model = tf.keras.applications.MobileNetV2(
    input_shape=(img_size, img_size, 3),
    include_top=False,
    weights="imagenet"
)

base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.3),
    layers.Dense(num_classes, activation="softmax")
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

print("Training started...")

history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=epochs
)

final_train_acc = history.history['accuracy'][-1]
final_val_acc = history.history['val_accuracy'][-1]

print("Training complete.")
print(f"Final Training Accuracy: {final_train_acc:.4f}")
print(f"Final Validation Accuracy: {final_val_acc:.4f}")

model.save("animal_classifier.h5")
print("Saved model as animal_classifier.h5")
