#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 4th May
@author: Akihiro Inui
"""

from flask import Flask, redirect, url_for, render_template, request, make_response, jsonify
import pyaudio
import time
import wave
from backend.src.feature_extraction.audio_feature_extraction import AudioFeatureExtraction
from backend.src.data_process.data_process import DataProcess
from backend.src.classifier.classifier_wrapper import Classifier
from backend.src.utils.file_utils import FileUtil
import pandas as pd
import numpy as np
from numpy import genfromtxt

app = Flask(__name__)

# RATE / number of updates per second
sampling_rate = 44100
chunk = int(sampling_rate/10)
channel_num = 1
format = pyaudio.paInt16
record_sec = 10
output_wav_file_name = "data/recording.wav"
AFE = AudioFeatureExtraction("../config/master_config.ini")
CLF = Classifier("../config/master_config.ini")
model = CLF.load_model("../backend/model/2019-05-07_20_28_12.880994/kNN.pickle")
mean_list = "../backend/mean_list.csv"
std_list = "../backend/std_list.csv"


def write2wave(player, recording_list: list, output_wav_file_name: str, channel_num: int, format, sampling_rate: int):
    """
    Write recording to wav file
    :param: player: audio player
    :param: recording_list: List of recording data
    :param: output_wav_file_name: Name of wave file to write out
    :param: channel_num: Number of channels
    :param: format: data format
    """
    wf = wave.open(output_wav_file_name, 'wb')
    wf.setnchannels(channel_num)
    wf.setsampwidth(player.get_sample_size(format))
    wf.setframerate(sampling_rate)
    wf.writeframes(b''.join(recording_list))
    wf.close()


def record_process():
    # Use pyaudio for recording
    player = pyaudio.PyAudio()
    stream = player.open(format=pyaudio.paInt16, channels=channel_num, rate=sampling_rate, input=True, frames_per_buffer=chunk)

    # Record for 10 seconds
    record_list = []
    for sec in range(0, int(sampling_rate / chunk * record_sec)):
        t1 = time.time()
        record_list.append(np.fromstring(stream.read(chunk), dtype=np.int16))
        print("took {} ms".format((time.time()-t1)*1000))

    # Stop recording
    stream.stop_stream()
    stream.close()
    player.terminate()

    # Write recording to wave file
    write2wave(player, record_list, output_wav_file_name, channel_num, format, sampling_rate)


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
    mean_data = genfromtxt(mean_list, delimiter=',')
    std_data = genfromtxt(std_list, delimiter=',')
    data_numpy = (np.array(dataframe[0]) - mean_data)
    reshaped_data = data_numpy.reshape(1, -1)

    # Make prediction
    result = CLF.predict(model, reshaped_data)
    return list(result[0])


@app.route('/')
def index():
    labels = ["Unkown"]
    values = [1.0]
    colors = ["#817B7A"]
    return render_template('index.html', set=zip(values, labels, colors))


@app.route('/record', methods=['POST', "GET"])
def record():
    labels = ["Unkown"]
    values = [1.0]
    colors = ["#817B7A"]

    # Start recording and return home when it's done
    record_process()
    return render_template('predict.html', set=zip(values, labels, colors))


@app.route("/predict", methods=['POST'])
def predict():
    # Prediction to recording data
    prediction_list = prediction_process("data/recording.wav")

    labels = ["Blues", "Classical", "Country", "Disco", "Hiphop", "Jazz", "Metal", "Pop", "Reggae", "Rock"]
    colors = ["#F7464A", "#46BFBD", "#FDB45C", "#FEDCBA", "#ABCDEF", "#DDDDDD", "#ABCABC", "#00FF80", "#FFFF99", "#FFCCFF"]
    return render_template('index.html', set=zip(prediction_list, labels, colors))


if __name__ == '__main__':
    app.run(debug=True)
