from flask import Flask, request, jsonify, render_template
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load the machine learning model
model = load_model('model.h5')

@app.route('/', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # get user inputs from the form
        title = request.form['title']
        genres = request.form['genres']
        description = request.form['description']
        anime_type = request.form['anime_type']
        producer = request.form['producer']
        studio = request.form['studio']

        # Use the model to make a prediction
        prediction = model.predict([[title, genres, description, anime_type, producer, studio]])

        # Return the predicted rating as a JSON response
        return jsonify({'prediction': prediction[0][0]})

    # If the request method is GET, render the home template
    return render_template('home.html')

if __name__ == '__main__':
    app.run(debug=True)