# model.py
import tensorflow as tf

class CatDogModel:
    def __init__(self, model_path):
        self.model = tf.keras.models.load_model(model_path)  # Load the model from the specified path

    def predict(self, image):
        image = image / 255.0  # Normalize the image
        image = tf.expand_dims(image, axis=0)  # Add batch dimension
        prediction = self.model.predict(image)
        return prediction
