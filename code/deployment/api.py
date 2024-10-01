from fastapi import FastAPI, UploadFile, File
import numpy as np
import keras
import cv2
import tensorflow as tf

app = FastAPI()
model = keras.models.load_model("models/imageclassifier.h5")


def preprocess_image(image):
    image = tf.image.resize(image, (256, 256))
    image = np.expand_dims(image, 0)
    return image


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    file_bytes = np.frombuffer(file.file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    input = preprocess_image(image)
    result = model.predict([input])[0][0]
    if result > 0.5:
        return "SAD"
    else:
        return "HAPPY"
