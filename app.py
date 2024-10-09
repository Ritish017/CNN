# app.py
from flask import Flask, request, render_template
import numpy as np
from keras.preprocessing import image
from model import CatDogModel

app = Flask(__name__)

# Load your model
model_path = r"C:\Users\kurma\OneDrive\Desktop\Project\cat_dog_classifier.keras"
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
    test_image = image.load_img(img_path, target_size=(64, 64))
    test_image = image.img_to_array(test_image)

    # Make the prediction
    result = cat_dog_model.predict(test_image)
    
    # Return the prediction result
    prediction = 'Dog' if result[0][0] >= 0.5 else 'Cat'
    return f'The prediction is: {prediction}'

if __name__ == '__main__':
    app.run(debug=True)
