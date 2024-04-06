
import json
import joblib
import numpy as np


# called when the deployment is created or updated
def init():
    global model
    # get the path to the registered model file and load it
    model_path = './model/model.pkl'
    model = joblib.load(model_path)

# called when a request is received
def run(raw_data):
    # get the input data as a numpy array
    data = np.array(json.loads(raw_data)['data'])
    # get a prediction from the model
    predictions = model.predict(data)
    # return the predictions as any JSON serializable format
    return predictions.tolist()
