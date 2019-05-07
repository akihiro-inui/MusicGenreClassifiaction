#!/bin/bash

echo "Downloading Dataset..."
curl -LO http://opihi.cs.uvic.ca/sound/genres.tar.gz
tar -zxvf genres.tar.gz
mv genres data
echo "Dataset Downloaded"

echo "Start Converting audio file format..."
for genre in data/*; do
    for filename in $genre/*; do
        sox $filename "$filename.wav";
        rm $filename;
    done
done
echo "Converted au to wav"

echo "Installing requirements..."
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt
mkdir backend/feature
mkdir backend/model
echo "Requirements installed"

echo "Creating Custom Dataset..."
python backend/src/data_process/audio_dataset_maker.py
echo "Custom Dataset Created"

echo "Starting Main Process..."
python backend/src/music_genre_classification
echo "All Process Completed"
