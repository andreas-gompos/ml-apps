from flask import Flask, request, render_template, jsonify
from dependencies import *

app = Flask(__name__)

preprocessing_pipeline = joblib.load('./artifacts/preprocessing_pipeline.pkl') 
model = load_model("./artifacts/model")

test_processed_text = preprocessing_pipeline.transform('testing')
test_probas = model.predict_proba(test_processed_text, verbose=0) * 100

@app.route('/predict')
def my_form():
    return render_template('my-form.html')

@app.route("/predict", methods=['GET', 'POST'])
def predict():
    if  request.method == "POST":
        text =  request.form['text']
        processed_text = preprocessing_pipeline.transform(text)
        probas = model.predict_proba(processed_text, verbose=0) * 100
        

        result = {'business': '{0:.2f}%'.format(probas[0][0]),
                  'entertainment': '{0:.2f}%'.format(probas[0][1]),
                  'politics': '{0:.2f}%'.format(probas[0][2]),
                  'sport': '{0:.2f}%'.format(probas[0][3]),
                  'tech': '{0:.2f}%'.format(probas[0][4])}


        return jsonify(result)


if __name__ == "__main__":


    app.run(host='0.0.0.0', port = 5000)
