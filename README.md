# Content-based Music Genre Classification



## Abstract

The automatic classification system to the most widely used music dataset; the GTZAN Music Genre Dataset was implemented. The system was implemented with a set of low-level features and several supervised classification methods.

## Introduction

Music Genre Classification to the most widely used dataset; GTZAN was implemented with a set of several low-level feature extraction and machine learning methods. 10 features are extracted from audio files and classified with 3 classifiers, k-NN, multilayer perceptron and convolutional neural network.

For the further information about the feature extraction methods, please have a look the papers in the reference.

## The Dataset
The GTZAN Music Genre Dataset, which is a collection of 1000 songs in 10 genres, is the most widely used dataset. 

Although many studies have been conducted using GTZAN, several faults have been pointed out. Sturm[1], for example, identified and analysed the contents of GTZAN and provided a catalogue of its faults. As it is the most used dataset, however, the system performance of MGC in this project should first be evaluated with GTZAN in order to compare against other systems used in other studies.

Details on the GTZAN Music Genre Dataset are presented in the table below. In GTZAN, each song is recorded at a sampling rate of 22.05 kHz and mono 16-bit audio files in .wav format.

<p align="center">
<img src="assets/GTZAN.png?raw=true" alt="GTZAN" width="500">
</p>

## Pre-processing

The input audio signal is segmented into analysis windows of 46ms length with an overlap of half size of an analysis window. The number of samples in an analysis window is usually the equal power of two to facilitate the use of FFT. For the system, 2048 samples are framed for an analysis window. Also, hamm widow is applied to each analysis window.

## Feature Extraction
<p align="center">
<img src="assets/Ex.png?raw=true" alt="Feature Extraction" width="600">
</p>

Ten types of low-level feature—six short-term and four long-term features—noted are extracted from the analysis windows and texture windows, respectively. In order to characterise the temporal evaluation of the audio signal, long-term features are computed by aggregating the short-term features.

<p align="center">
<img src="assets/Features.png?raw=true" alt="Feature Extraction" width="300">
</p>

Over a texture window which consists of 64 analysis windows, short-term features are integrated with mean values.


## Classifier
Fuzzy k-NN

This method uses Euclidean distance between training and testing data. Each test sample is classified depending on the k number of training samples that surround the test sample. In the case of dissimilarity ties, the class appearing most often among the k nearest training observations will be the predicted answer. However, only a class label is assigned the test sample, and it does not have information on the strength of membership in that class

Fuzzy k-NN, a combination of fuzzy logic and k-NN, was proposed to address the problem [5]. It has two steps: 1) fuzzy labelling, which computes fuzzy vectors of training data, and 2) fuzzy classification, which computes the fuzzy vectors of test data.

Multilayer Perceptron

The multilayer Perceptron to this project consists of 3 layers with relu function. The training data is splitted into 10% for the validation and 90% for the training. It loads the data from "Data.csv".

Logistic Regression




## Results
k-NN: 62.0%


MLP: 83.6%


### Dependency
Please see requirements.txt


## Complete Installation

1. Install sox and/or ffmpeg (see below)
2. Clone this project
3. Go to the folder(MusicGenreClassification)
4. bash run_me_first.sh
5. Run backend/src/data_process/audio_dataset_maker.py
6. Run backend/src/music_genre_classification.py
* You may need to install sox and/or ffmpeg if you do not have one. (Windows complains sometimes..) 

For installing sox (Windows)
https://sourceforge.net/projects/sox/

For installing sox (Mac)
http://macappstore.org/sox/

For installing ffmpeg
https://www.ffmpeg.org/download.html
or 
brew install ffmpeg

* Make sure you are using Python3


## References
[1] B. L.Sturm, "The GTZAN dataset: Its contents, its faults, their effects on evaluation, and its
future use," arXiv preprint arXiv:1306.1461, 2013.

[2] E.Loweimi, S. M. Ahadi, T. Drugman and S. Loveymi, "On the importance of pre-emphasis
and window shape in phase-based speech recognition," in International Conference on
Nonlinear Speech Processing(pp. 160-167). Springer, Berlin, Heidelberg, June, 2013.

[3] K. K. Chang, J. S. R. Jang and C. S. Iliopoulos, "Music Genre Classification via Compressive
Sampling," in ISMIR, pp. 387-392, August, 2010.

[4] B. L. Sturm, "On music genre classification via compressive sampling," in Multimedia and
Expo (ICME), 2013 IEEE International Conference on (pp. 1-6). IEEE, July, 2013.

[5] Keller, M. R. Gray and J. A. Givens, A fuzzy k-nearest neighbor algorithm. IEEE
transactions on systems, man, and cybernetics, (4), pp.580-585, 1985.

[6] http://benanne.github.io/2014/08/05/spotify-cnns.html

## Author

Akihiro Inui
http://www.portfolio.akihiroinui.com


## License

MusicGenreClassification is available under the MIT license. See the LICENSE file for more info.

