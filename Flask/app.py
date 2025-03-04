import os
import numpy as np
from flask import Flask, request, render_template, send_from_directory, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet_v2 import preprocess_input

# Load the trained model
MODEL_PATH = r"c:/Users/Balir/Downloads/project/training/evgg.h5"
model = load_model(MODEL_PATH)

# Initialize Flask app
app = Flask(__name__, template_folder='templates', static_folder='static')

# Configure upload folder
UPLOAD_FOLDER = os.path.join(app.root_path, 'static/uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Define class labels
CLASSES = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload')
def upload():
    return render_template('img_input.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('error.html', message="No file uploaded. Please try again.")

    file = request.files['image']
    if file.filename == '':
        return render_template('error.html', message="No file selected. Please choose an image.")

    # Save the uploaded file
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    # Load and preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    img_data = preprocess_input(x)

    # Model prediction
    prediction = np.argmax(model.predict(img_data), axis=1)[0]
    result = CLASSES[prediction]

    # Generate URL for displaying the image
    image_url = url_for('static', filename=f'uploads/{file.filename}')

    return render_template('output.html', prediction=result, image_url=image_url)

@app.route('/static/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run(debug=True)
