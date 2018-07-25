from flask import Flask, request, render_template, jsonify
from dependencies import *

app = Flask(__name__)

preprocessing_pipeline = joblib.load('./artifacts/preprocessing_pipeline.pkl') 
model = load_model("./artifacts/model")

test_processed_text = preprocessing_pipeline.transform('testing')
test_probas = model.predict_proba(test_processed_text, verbose=0) * 100

@app.route("/", methods=['POST'])
def predict():

    data = request.get_json()
    processed_text = preprocessing_pipeline.transform(data['text'])
    probas = model.predict_proba(processed_text, verbose=0) * 100
    
    result = {'business': str(probas[0][0]),
              'entertainment': str(probas[0][1]),
              'politics': str(probas[0][2]),
              'sport': str(probas[0][3]),
              'tech': str(probas[0][4])}

    return jsonify(result)


if __name__ == "__main__":

    app.run(host='0.0.0.0', port = 5000)
