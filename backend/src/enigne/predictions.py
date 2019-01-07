#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on 1/5 2019

@author: Akihiro Inui
"""

from sklearn.externals import joblib
from boto.s3.key import Key
from boto.s3.connection import S3Connection
from flask import Flask
from flask import request
from flask import json
from django import http.request
from django import http.json


BUCKET_NAME = 'your-s3-bucket-name'
MODEL_FILE_NAME = 'your-model-name.pkl'
MODEL_LOCAL_PATH = '/tmp/' + MODEL_FILE_NAME

app = Flask(__name__)

@app.route('/', methods=['POST'])
def index():
  payload = json.loads(request.get_data().decode('utf-8'))
  prediction = predict(payload['payload'])
  data = {}
  data['data'] = prediction[-1]
  return json.dumps(data)

def load_model():
  conn = S3Connection()
  bucket = conn.create_bucket(BUCKET_NAME)
  key_obj = Key(bucket)
  key_obj.key = MODEL_FILE_NAME

  contents = key_obj.get_contents_to_filename(MODEL_LOCAL_PATH)
  return joblib.load(MODEL_LOCAL_PATH)

def predict(data):
  # Process your data, create a dataframe/vector and make your predictions
  final_formatted_data = []
  return load_model().predict(final_formatted_data)