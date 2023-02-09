from flask import Flask, render_template, request
import pickle
import numpy as np


app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    experience = int(request.form.get('experience'))
    test_score = float(request.form.get('test_score'))
    interview_score = float(request.form.get('interview_score'))
    X = np.array([experience,test_score,interview_score])
    Y = round(model.predict(X.reshape(1,-1))[0],2)
    return render_template('index.html', prediction_text=f'The predicted salary is ${Y}')


if __name__ == '__main__':
    app.run(debug=True)