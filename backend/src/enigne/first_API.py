from bottle import route, run, request
#from keras.models import load_model
import os
import pickle
import pandas as pd
import numpy as np

def load_model(input_model_file_name: str):
    """
    Load trained model
    :param  input_model_file_name: model file name
    :return trained model
    """
    assert os.path.exists(input_model_file_name), "Selected model does not exist"
    # Unpickle model
    with open(input_model_file_name, mode='rb') as fp:
        loaded_model = pickle.load(fp)
    return loaded_model


def predict(input_data, model):
    #predicted_result = model.predict(input_data)
    a=2
    #predicted_class = np.argmax(predicted_result)
    return a


@route('/classify')
def classify():
    input_data_file = "../dummy_data.csv"
    input_data = pd.read_csv(input_data_file)

    model = load_model("knn.pickle")
    if input_data:
        a = "Data loaded"
    else:
        a = "Data not loaded"
    if model:
        b = "Model loaded"
    else:
        b = "Model not loaded"
    answer = "{0}_{1}".format(a, b)
    return answer


run(host='localhost', port=8080, debug=True, reloader=True)
