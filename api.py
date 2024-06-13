import os
import numpy as np
from flask import Flask, request, jsonify
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import load_model
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.applications.vgg16 import preprocess_input

app = Flask(__name__)

# Load the trained model for prediction
model = load_model('C:/skinsavermodel/model.h5')

# Load pre-trained model for feature extraction (assuming VGG16 is used for example)
def extract_feature_vector(image_path):
    img = load_img(image_path, target_size=(255, 255))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    feature_vector = model.predict(img_array)
    return feature_vector

# Reference images for validation
reference_images = {
    'bcc': 'D:/final project/Skin Cancer/Skin Cancer1/ISIC_0031585.jpg',
    'vasc': 'D:/final project/Skin Cancer/Skin Cancer1/ISIC_0028714.jpg',
    'mel': 'D:/final project/Skin Cancer/Skin Cancer1/ISIC_0032892.jpg',
    'bkl': 'D:/final project/Skin Cancer/Skin Cancer1/ISIC_0027419.jpg',
    'nv': 'D:/final project/Skin Cancer/Skin Cancer1/ISIC_0029981.jpg',
    'akiec': 'D:/final project/Skin Cancer/Skin Cancer1/ISIC_0026362.jpg',
    'df': 'D:/final project/Skin Cancer/Skin Cancer1/ISIC_0025911.jpg'
}

# Extract feature vectors for reference images
reference_vectors = {label: extract_feature_vector(path) for label, path in reference_images.items()}

def is_relevant(image_path, reference_vectors, threshold=0.5):
    query_vector = extract_feature_vector(image_path)
    similarities = [cosine_similarity(query_vector, ref_vector)[0][0] for ref_vector in reference_vectors.values()]
    max_similarity = max(similarities)
    return max_similarity >= threshold

def load_and_preprocess_image(image_path):
    img = load_img(image_path, target_size=(255, 255))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'image_file' not in request.files:
        return jsonify({'error': 'Image file is required'}), 400

    image_file = request.files['image_file']

    if image_file.filename == '':
        return jsonify({'error': 'No selected file for image'}), 400

    if image_file:
        try:
            # Ensure the upload directory exists
            os.makedirs('uploads', exist_ok=True)

            # Save the image file to the server
            image_path = os.path.join('uploads', image_file.filename)
            image_file.save(image_path)

            # Validate if the image is a skin image
            if not is_relevant(image_path, reference_vectors):
                os.remove(image_path)
                return jsonify({'error': 'The uploaded image is not a relevant skin image'}), 400

            # Preprocess the image
            img_array = load_and_preprocess_image(image_path)

            # Make prediction
            prediction = model.predict(img_array)

            # Interpret the prediction
            prediction_label = 'Malignant' if prediction[0][0] > 0.5 else 'Benign'

            # Cleanup the saved image file
            os.remove(image_path)

            return jsonify({'prediction': prediction_label})

        except Exception as e:
            return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
