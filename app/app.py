from flask import Flask, render_template, request
import joblib

app = Flask(__name__)

# Load the trained model
model = joblib.load('../models/sentiment_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form['text']
    prediction = model.predict([text])[0]
    if prediction == 1:
        result = 'Positive'
    else:
        result = 'Negative'
    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
