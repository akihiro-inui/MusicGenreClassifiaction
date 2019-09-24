"""
Created on 4th May	
@author: Akihiro Inui	
"""

from flask import Flask, redirect, url_for, render_template, request, make_response, jsonify
# import pyaudio
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


ALLOWED_EXTENSIONS = set('wav')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
keras.backend.clear_session()

# RATE / number of updates per second
sampling_rate = 44100
chunk = int(sampling_rate/10)
channel_num = 1
# format = pyaudio.paInt16
record_sec = 10
output_wav_file_name = "data/recording.wav"
AFE = AudioFeatureExtraction("../config/master_config.ini")
CLF = Classifier("../config/master_config.ini")
# model = CLF.load_model("../backend/model/2019-05-07_22_14_20.801103/gru.h5")
mean_list = "../backend/mean_list.csv"
std_list = "../backend/std_list.csv"


def prediction_process(input_audio_file_path: str):
    """
    Process one audio file, make prediction to it.
    :param: input_audio_file_path: Path to input audio file
    """
    # Feature extraction
    feature_dict, file_short_feature_list = AFE.extract_file(input_audio_file_path)

    # Stats across frames
    stats_dict = AFE.get_feature_stats(feature_dict, "mean")

    # Segment dictionary data
    segmented_dict = DataProcess.segment_feature_dict(stats_dict)

    # Data shape formatting
    dataframe = pd.DataFrame.from_dict(segmented_dict, orient='index')
    #mean_data = genfromtxt(mean_list, delimiter=',')
    #std_data = genfromtxt(std_list, delimiter=',')
    #data_numpy = (np.array(dataframe[0]) - mean_data)
    #reshaped_data = data_numpy.reshape(1, -1)

    # 3D feature
    feature_2d = np.array(file_short_feature_list)
    feature_3d = feature_2d[newaxis, :, :]
    print(feature_3d.shape)
    # Make prediction
    result = CLF.predict(model, feature_3d)
    return list(result[0])


@app.route('/')
def index():
    labels = ["Unkown"]
    values = [1.0]
    colors = ["#817B7A"]
    return render_template('index.html', set=zip(values, labels, colors))


@app.route('/', methods=['POST'])
def upload_file():
    if request.method == 'POST':
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
            return redirect('/')
        else:
            flash('Allowed file types are txt, pdf, png, jpg, jpeg, gif')
            return redirect(request.url)


@app.route("/predict", methods=['POST'])
def predict():
    # Prediction to recording data
    prediction_list = prediction_process("data/recording.wav")

    labels = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    colors = ["#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA", "#ABCDEF", "#DDDDDD", "#ABCABC", "#00FF80", "#FFFF99", "#FFCCFF"]
    return render_template('index.html', set=zip(prediction_list, labels, colors))


if __name__ == '__main__':
    app.run(debug=True)
