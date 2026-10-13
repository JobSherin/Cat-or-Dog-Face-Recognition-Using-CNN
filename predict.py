import tensorflow as tf
from tensorflow import keras
import numpy as np

model = keras.models.load_model("cat_dog_model.keras")

image = keras.utils.load_img("test_image.jpg", target_size=(128, 128))

image = keras.utils.img_to_array(image)

image = np.expand_dims(image, axis=0)

prediction = model.predict(image)

if prediction[0][0] < 0.5:
    confidence = (1-prediction[0][0]) *100
    print("Predicion: Cat")
    print("Confidence:", round(confidence, 2), "%")

else:
    confidence = prediction[0][0] *100
    print("Prediction: Dog")
    print("Confidence:", round(confidence, 2), "%") 