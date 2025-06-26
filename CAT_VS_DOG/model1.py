# imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import pathlib

# Dataset collection 
base_dir = pathlib.Path(r"C:\Users\mayan\Desktop\OTH\ai\examples\catdog\cats_vs_dogs_aug\cats_vs_dogs_small")
train_dataset = keras.utils.image_dataset_from_directory(
    base_dir / "train",
    image_size=(180, 180),
    batch_size=32
)
validation_dataset = keras.utils.image_dataset_from_directory(
    base_dir / "validation",
    image_size=(180, 180),
    batch_size=32
)
test_dataset = keras.utils.image_dataset_from_directory(
    base_dir / "test",
    image_size=(180, 180),
    batch_size=32
)

data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.2),
])


inputs = keras.Input(shape=(180, 180, 3))
x = data_augmentation(inputs) 
x = layers.Rescaling(1./255)(x) # image pixel values are in the range of 0-255 so we are converting it to 0-1 
x = layers.Conv2D(filters = 32, kernel_size = 3, activation="relu")(x) 
#  downsampling
x = layers.MaxPooling2D(2)(x)


x = layers.Conv2D(filters = 64, kernel_size = 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(filters = 128, kernel_size = 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)
x = layers.Conv2D(filters = 256, kernel_size = 3, activation="relu")(x)
x = layers.MaxPooling2D(2)(x)

#  last filtering step before flattening the data
x = layers.Conv2D(filters = 256, kernel_size = 3, activation="relu")(x)
# Converting 3D tensor to 1D vector
x = layers.Flatten()(x)

x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)


model = keras.Model(inputs, outputs)

# making the model ready for training
model.compile(
    loss="binary_crossentropy",
    optimizer="rmsprop",
    metrics=["accuracy"]
)


callbacks = [
    keras.callbacks.ModelCheckpoint(
        filepath="convnet_from_scratch_with_augmentation.keras",
        save_best_only=True,
        monitor="val_loss"
    )
]

# actual training starts 
# histroy objects stores all the metrics like -: validation accuracy and loss,  training accuracy and loss
history = model.fit(
    train_dataset,# training data
    # epochs=100,
    epochs =20,
    validation_data=validation_dataset,
    callbacks=callbacks
)
# Load best model
test_model = keras.models.load_model("convnet_from_scratch_with_augmentation.keras")

# Evaluate
test_loss, test_acc = test_model.evaluate(test_dataset)
print(f"Test accuracy: {test_acc:.3f}") # printing the test accuracy upto 3 decimal points 


# plot of acuuravy and loss 
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
