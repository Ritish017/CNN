from flask import Flask, request, render_template
import numpy as np
from keras.preprocessing import image
from model import CatDogModel
import os

app = Flask(__name__)

# Load your model (ensure the model file is in the project directory)
model_path = 'cat_dog_classifier.keras'  # Place the model file in the project directory
cat_dog_model = CatDogModel(model_path)

@app.route('/')
def home():
    return render_template('index.html')  # Render the home page

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return 'No file part'

    file = request.files['file']
    
    if file.filename == '':
        return 'No selected file'

    # Load and preprocess the image
    img_path = 'uploaded_image.jpg'  # Save the uploaded image
    file.save(img_path)

    # Preprocess the image
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)  # Add batch dimension
    test_image /= 255.0  # Scale pixel values to [0, 1]

    # Make the prediction
    result = cat_dog_model.predict(test_image)
    
    # Clean up by removing the uploaded image
    os.remove(img_path)

    # Return the prediction result
    prediction = 'Dog' if result[0][0] >= 0.5 else 'Cat'
    return f'The prediction is: {prediction}'

if __name__ == '__main__':
    app.run()  # Set debug=False for production
