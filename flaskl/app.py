
import numpy as np
from flask import Flask, jsonify, request, render_template
import joblib
import pickle


model = pickle.load(open('model.pkl',"rb"))


app = Flask(__name__)

@app.route('/')
def index():
    return render_template('home.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/pred')
def pred():
    return render_template('ind.html')

@app.route('/predict', methods=['GET','POST'])
def predict():
    views = float(request.form['views'])
    likes = float(request.form['likes'])
    dislikes = float(request.form['dislikes'])
    comments = float(request.form['comments'])
    year = float(request.form['year'])
    duration = float(request.form['duration'])
    category = float(request.form['category'])
    features = np.array([[views, likes, dislikes, comments, year, duration, category]])
    prediction = model.predict(features)
    output =round(prediction[0],2)
    return render_template('result.html', prediction_text='{}'.format(output)) 
    
if __name__ == '__main__':
    app.run(debug=True, port=8001)
