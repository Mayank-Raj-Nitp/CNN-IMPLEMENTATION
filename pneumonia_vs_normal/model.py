#imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib
import matplotlib.pyplot as plt

# Dataset collection 
base_dir = pathlib.Path(r"C:\Users\mayan\Desktop\OTH\ai\examples\xray\data")
train_dataset = keras.utils.image_dataset_from_directory(
    base_dir / "train",
    image_size=(180, 180),
    batch_size=32
)
validation_dataset = keras.utils.image_dataset_from_directory(
    base_dir / "val",
    image_size=(180, 180),
    batch_size=32
)
test_dataset = keras.utils.image_dataset_from_directory(
    base_dir / "test",
    image_size=(180, 180),
    batch_size=32
)

# augmentation 

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
])

#Cnn 
inputs = keras.Input(shape=(180,180,3))
x = data_augmentation(inputs)
x = layers.Rescaling(1./255)(x)
x= layers.Conv2D(filters = 32, kernel_size = 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x= layers.Conv2D(filters = 64, kernel_size = 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x= layers.Conv2D(filters = 128, kernel_size = 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)
x= layers.Conv2D(filters = 256, kernel_size = 3, activation='relu')(x)
x = layers.MaxPooling2D(2)(x)

x= layers.Conv2D(filters = 256, kernel_size = 3, activation='relu')(x)
x = layers.Flatten()(x)

x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    loss="binary_crossentropy", 
    optimizer="rmsprop",
    metrics=["accuracy"]
)

callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="bestmodel.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]
history = model.fit(
    train_dataset,
    epochs =20,
    validation_data=validation_dataset,
    callbacks=callbacks
)
test_model = keras.models.load_model("bestmodel.keras")
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") # printing the test accuracy upto 3 decimal points 


accuracy = history.history["accuracy"]
val_accuracy = history.history["val_accuracy"]
loss = history.history["loss"]
val_loss = history.history["val_loss"]
epochs = range(1, len(accuracy) + 1)
plt.plot(epochs, accuracy, "bo", label="Training accuracy")
plt.plot(epochs, val_accuracy, "b", label="Validation accuracy")
plt.title("Training and validation accuracy")
plt.legend()
plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()                                        
