from flask import Flask, request, render_template, jsonify	
from dependencies import *
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['POST'])
def predict():

    incoming_request = request.get_json()
    processed_text = preprocessing_pipeline.transform(incoming_request['text'])
    probas = model.predict_proba(processed_text, verbose=0) * 100
    
    model_response = {'business': str(probas[0][0]),
                      'entertainment': str(probas[0][1]),
                      'politics': str(probas[0][2]),
                      'sport': str(probas[0][3]),
                      'tech': str(probas[0][4])}

    return jsonify(model_response)

if __name__ == "__main__":

    print("loading models - waiting until the server is up")
    preprocessing_pipeline = joblib.load('./artifacts/preprocessing_pipeline.pkl') 
    model = load_model("./artifacts/model")

    test_processed_text = preprocessing_pipeline.transform('testing')
    test_probas = model.predict_proba(test_processed_text, verbose=0) * 100
    print('server is up')

    app.run(host="0.0.0.0", port=5000)