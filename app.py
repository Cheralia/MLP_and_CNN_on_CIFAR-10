import os
import numpy as np
from flask import Flask, request, jsonify, render_template
from tensorflow.keras.models import load_model
from PIL import Image
import io

app = Flask(__name__, template_folder='.')

# 1. Load Models Globally (to avoid reloading on every request)
print("Loading models... this may take a moment.")
try:
    # Ensure these file names match exactly what is in your folder
    mlp_model = load_model('cifar10_mlp_classification_model.h5')
    cnn_model = load_model('cifar10_cnn_classification_model.h5')
    print("Models loaded successfully.")
except Exception as e:
    print(f"Error loading models: {e}")
    print("Please ensure .h5 files are in the same directory as app.py")

@app.route('/')
def home():
    # Serves your HTML file
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    print("model search")
    model_type = request.form.get('model_type') # 'mlp' or 'cnn'
    print("model get")
    if not file:
        return jsonify({'error': 'Invalid file'}), 400

    try:
        # --- Preprocessing Steps ---
        
        # 1. Open image and convert to RGB (removes Alpha channel if PNG)
        img = Image.open(io.BytesIO(file.read())).convert('RGB')
        
        # 2. Resize to 32x32 (CIFAR-10 native size)
        img = img.resize((32, 32))
        
        # 3. Convert to NumPy Array
        img_array = np.array(img)
        
        # 4. Normalize pixel values to [0, 1]
        img_array = img_array.astype('float32') / 255.0
        
        # 5. Handle Shape based on Model Type
        if model_type == 'mlp':
            # For MLP: Flatten images
            # Input shape becomes (1, 3072)
            input_data = img_array.reshape(1, -1)
        else:
            # For CNN: Keep 3D shape, add batch dimension
            # Input shape becomes (1, 32, 32, 3)
            input_data = img_array.reshape(1, 32, 32, 3)
            
        # --- Prediction ---
        if model_type == 'mlp':
            prediction = mlp_model.predict(input_data)
        else:
            prediction = cnn_model.predict(input_data)
            
        # Convert numpy array to list for JSON serialization
        # Result is typically [[0.1, 0.05, ...]] so we take [0]
        probabilities = prediction[0].tolist()
        
        return jsonify({'probabilities': probabilities})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)