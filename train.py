import tensorflow as tf
from tensorflow import keras

dataset = keras.utils.image_dataset_from_directory(
    "dataset", image_size = (128,128),batch_size = 32
)

print("Classes:",dataset.class_names)

model = keras.Sequential([
    keras.layers.Input(shape=(128,128,3)),

    keras.layers.Conv2D(32,(3,3),activation="relu"),

    keras.layers.MaxPooling2D(pool_size=(2,2)),

    keras.layers.Conv2D(64,(3,3),activation="relu"),

    keras.layers.MaxPooling2D(pool_size=(2,2)),

    keras.layers.Flatten(),


    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(1, activation="sigmoid")
])


model.summary()

model.compile( optimizer="adam", 
              loss="binary_crossentropy", 
              metrics=["accuracy"])

model.fit(dataset, epochs=5)

loss,accuracy = model.evaluate(dataset)

print("Loss:",loss)
print("Accuracy:",accuracy)

model.save("cat_dog_model.keras")

print("Model saved successfully!")