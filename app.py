import pandas as pd
import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle

app = Flask(__name__)
regmodel = pickle.load(open('regmodel.pkl', 'rb'))  # Load the model
scaler = pickle.load(open('scaling.pkl', 'rb'))  # Load the scaler model

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    data = request.json['data']
    print(data)
    print(np.array(list(data.values())).reshape(1, -1))
    new_data = scaler.transform(np.array(list(data.values())).reshape(1, -1))
    output = regmodel.predict(new_data)
    print(output[0])
    return jsonify({'prediction': output[0]})


@app.route('/predict', methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_input=scaler.transform(np.array(list(data)).reshape(1, -1))
    print(final_input)
    output=regmodel.predict(final_input)
    return render_template('home.html', prediction_text='Predicted house price is : {}'.format(output[0]))
    

if __name__ == "__main__":
    app.run(debug=True) 