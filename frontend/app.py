"""
Created on 4th May	
@author: Akihiro Inui	
"""

from flask import Flask, redirect, url_for, render_template, request, make_response, jsonify
import cloudpickle
import time
import wave
import urllib.request
import os
from flask import Flask, flash, request, redirect, render_template
from werkzeug.utils import secure_filename
from backend.src.feature_extraction.audio_feature_extraction import AudioFeatureExtraction
from backend.src.data_process.data_process import DataProcess
from backend.src.classifier.classifier_wrapper import Classifier
from backend.src.utils.file_utils import FileUtil
import pandas as pd
import numpy as np
from numpy import genfromtxt, newaxis
import keras


# App config
app = Flask(__name__)
keras.backend.clear_session()
app.config['UPLOAD_FOLDER'] = 'uploaded_data'
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
ALLOWED_EXTENSIONS = set(['wav', 'mp3'])

# output_wav_file_name = "data/recording.wav"
AFE = AudioFeatureExtraction("../config/master_config.ini")
CLF = Classifier("../config/master_config.ini")
model = CLF.load_model("../backend/model/2019-10-10_22_00_21.896549/logistic_regression.pkl")
# mean_list = "../backend/mean_list.csv"
# std_list = "../backend/std_list.csv"


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def prediction_process(input_audio_file_path: str):
    """
    Process one audio file, make prediction to it.
    :param: input_audio_file_path: Path to input audio file
    """
    # Feature extraction
    feature_dict = AFE.extract_file(input_audio_file_path)

    # Stats across frames
    # stats_dict = AFE.get_feature_stats(feature_dict, "mean")

    feature_vector = []
    # Process for each feature
    for feature_name, feature in feature_dict.items():
        # Take stats across frames for expert system and append to list
        if type(feature) is list:
            feature_array = np.array(feature[:])
            if feature_array.ndim == 1:
                feature_vector.append(np.mean(feature_array))
            else:
                feature_vector.extend(np.mean(feature_array, axis=0))
        # Append mel-spectrogram to 3D array
        else:
            mel_spectrogram_3d_array = np.dstack((np.empty(np.shape(feature), int), feature))
            mel_spectrogram_3d_array = mel_spectrogram_3d_array[:, :, 1].T

    # TODO: Subtract feature stats, run prediction
    mean_feature_vector = FileUtil.text2list('../backend/src/expert_feature_mean_list.txt')
    mean_feature_vector = [float(element) for element in mean_feature_vector]
    mean_subtracted_feature = np.subtract(np.array(feature_vector), np.array(mean_feature_vector))

    # Make prediction
    result = CLF.predict(model, mean_subtracted_feature)
    return list(result)


@app.route('/')
def index():
    labels = ["Unkown"]
    values = [1.0]
    colors = ["#817B7A"]
    return render_template('index.html', set=zip(values, labels, colors))


def upload_file():
    # check if the post request has the file part
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No file selected for uploading')
        return redirect(request.url)
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        flash('File successfully uploaded')
        return os.path.join(app.config['UPLOAD_FOLDER'], filename)
    else:
        flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')


@app.route("/predict", methods=['POST'])
def predict():
    # Upload file
    file_path = upload_file()

    # Prediction to recording data
    prediction_list = prediction_process(file_path)

    labels = ["Blues", "Classical"]
    colors = ["#F7464A", "#46BFBD"]
    # labels = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    # colors = ["#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA", "#ABCDEF", "#DDDDDD", "#ABCABC", "#00FF80", "#FFFF99", "#FFCCFF"]
    return render_template('index.html', set=zip(prediction_list, labels, colors))


if __name__ == '__main__':
    app.run(debug=True)
