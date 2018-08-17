from sanic import Sanic, request
from sanic.response import json
from sanic_cors import CORS, cross_origin
from dependencies import *
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

app = Sanic()
CORS(app, automatic_options=True)

@app.route("/", methods=['POST'])
async def predict(request):

    incoming_request = request.json
    processed_text = preprocessing_pipeline.transform(incoming_request['text'])

    if np.count_nonzero(processed_text) == 0:
        model_response = {'business': str(0),
                          'entertainment': str(0),
                          'politics': str(0),
                          'sport': str(0),
                          'tech': str(0)}
    else:
        probas = model.predict_proba(processed_text, verbose=0) * 100
        model_response = {'business': str(probas[0][0]),
                          'entertainment': str(probas[0][1]),
                          'politics': str(probas[0][2]),
                          'sport': str(probas[0][3]),
                          'tech': str(probas[0][4])}

    return json(model_response)

if __name__ == "__main__":

    logger.info('loading models - waiting until the server is up')
    preprocessing_pipeline = joblib.load('./artifacts/preprocessing_pipeline.pkl')
    model = load_model("./artifacts/model")
    logger.info('server is up')

    app.run(host="0.0.0.0", port=5000, workers=1)
