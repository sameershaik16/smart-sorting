from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# ✅ Flask setup
app = Flask(
    __name__,
    template_folder=os.path.join(os.path.dirname(__file__), '../templates')
)

# ✅ Load your trained model
model = load_model('fruit_model.h5')
img_height, img_width = 224, 224

# ✅ Define your classes list (order must match training)
class_names = [
    'Apple__Healthy', 'Apple__Rotten',
    'Banana__Healthy', 'Banana__Rotten',
    'Bellpepper__Healthy', 'Bellpepper__Rotten',
    'Carrot__Healthy', 'Carrot__Rotten',
    'Cucumber__Healthy', 'Cucumber__Rotten',
    'Grape__Healthy', 'Grape__Rotten',
    'Guava__Healthy', 'Guava__Rotten',
    'Jujube__Healthy', 'Jujube__Rotten',
    'Mango__Healthy', 'Mango__Rotten',
    'Orange__Healthy', 'Orange__Rotten',
    'Pomegranate__Healthy', 'Pomegranate__Rotten',
    'Potato__Healthy', 'Potato__Rotten',
    'Strawberry__Healthy', 'Strawberry__Rotten',
    'Tomato__Healthy', 'Tomato__Rotten'
]


# ✅ Homepage
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/predict')
def predict():
    return render_template('predict.html')

# ✅ Prediction route
@app.route('/inspect', methods=['GET', 'POST'])
def inspect():
    # Redirect GET to /predict
    if request.method == 'GET':
        return redirect(url_for('predict_page'))

    # Handle POST image
    file = request.files.get('image')
    if not file:
        return render_template('predict.html', result="No image uploaded!")

    # Save file
    uploads_dir = os.path.join('static', 'uploads')
    os.makedirs(uploads_dir, exist_ok=True)
    file_path = os.path.join(uploads_dir, file.filename)
    file.save(file_path)

    # Preprocess image
    img = image.load_img(file_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]

    return render_template(
        'predict.html',
        result=predicted_class,
        image_url=file_path
    )

if __name__ == '__main__':
    app.run(debug=True)
