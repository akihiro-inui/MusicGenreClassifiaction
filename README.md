
<p align="center">
<img src="assets/AAG_Logo.jpg?raw=true" alt="MusicGenreClassification" width="1000">
</p>


# Content-based Music Genre Classification

Academic research in the field of Music Information Retrieval, The University of Edinburgh

## Abstract

The automatic classification system to the most widely used music dataset; the GTZAN Music Genre Dataset was implemented. The system was implemented with a set of low-level features and several supervised classification methods.

## Introduction

Music Genre Classification to the most widely used dataset; GTZAN was implemented with a set of several low-level feature extraction and machine learning methods.

Music Genre Collection must be saved in "Target" folder. After feature extraction computed in readgenre, randomise should be run in order to save data matrices for iteretion. Following these steps, knn or mlp give you predicted genre of songs.

## The Dataset
The GTZAN Music Genre Dataset, which is a collection of 1000 songs in 10 genres, is the most widely used dataset. 

Although many studies have been conducted using GTZAN, several faults have been pointed out. Sturm[1], for example, identified and analysed the contents of GTZAN and provided a catalogue of its faults. As it is the most used dataset, however, the system performance of MGC in this project should first be evaluated with GTZAN in order to compare against other systems used in other studies.

Details on the GTZAN Music Genre Dataset are presented in the table below. In GTZAN, each song is recorded at a sampling rate of 22.05 kHz and mono 16-bit audio files in .wav format.

<p align="center">
<img src="assets/GTZAN.png?raw=true" alt="GTZAN" width="500">
</p>

## Pre-processing
<p align="center">
<img src="assets/PreP.png?raw=true" alt="Pre-Processing" width="700">
</p>

First, the input signal is sent to a high pass filter. The pre-emphasis increases the relative energy of the high-frequency spectrum [2].
The importance of pre-emphasis is often noted in speech processing, especially if the extracted features are MFCC, and 0.97 is usually chosen as the pre-emphasis coefficient [2]. The pre-emphasis compensates for the high-frequency formants which were suppressed during the sound production by instruments or the human voice. In this system, pre-emphasis was optionally selected and applied to the input signal.

Second, the emphasised input audio signal is segmented into analysis windows of 46ms length with an overlap of half size of an analysis window. The number of samples in an analysis window is usually the equal power of two to facilitate the use of FFT. For the system, 2048 samples are framed for an analysis window.

Finally, the framed signals are inputted into a matrix and the silence removal is applied to each analysis frame. As the silence in the audio signal can affect the FFT computation in the reproduction of the system produced by Chang et al. [3] and Sturm [4], an appropriate silence removal method was considered. For each analysis frame, the number of non-zero samples are counted and compared to the number of zeros. The silence removal only validates frames which have non-zero samples that are more than the half of an analysis window (1024). Hence, frames containing zeros more than 1024 samples are invalidated.

## Feature Extraction
<p align="center">
<img src="assets/Ex.png?raw=true" alt="Feature Extraction" width="600">
</p>

Ten types of low-level feature—six short-term and four long-term features—noted are extracted from the analysis windows and texture windows, respectively. In order to characterise the temporal evaluation of the audio signal, long-term features are computed by aggregating the short-term features.

<p align="center">
<img src="assets/Features.png?raw=true" alt="Feature Extraction" width="300">
</p>

Over a texture window which consists of 64 analysis windows, short-term features are integrated with mean values.
Also, extracted features from audio files were plotted in a figure like below. These featuremaps are saved in a drive (See installation).


<p align="center">
<img src="assets/FMap.png?raw=true" alt="Feature Extraction" width="400">
</p>


## Classifier
Fuzzy k-NN

This method uses Euclidean distance between training and testing data. Each test sample is classified depending on the k number of training samples that surround the test sample. In the case of dissimilarity ties, the class appearing most often among the k nearest training observations will be the predicted answer. However, only a class label is assigned the test sample, and it does not have information on the strength of membership in that class

Fuzzy k-NN, a combination of fuzzy logic and k-NN, was proposed to address the problem [5]. It has two steps: 1) fuzzy labelling, which computes fuzzy vectors of training data, and 2) fuzzy classification, which computes the fuzzy vectors of test data.

Multilayer Perceptron

Convolutional Neural Network

## Results
k-NN: 62.0%
MLP: 78.0% 

CNN: 46.3% (Need to find more suitable parameters, seems overfitting)
<p>
<img src="assets/CNN accuracy.png?raw=true" alt="Feature Extraction" width="400", align="left">
<img src="assets/CNN loss.png?raw=true" alt="Feature Extraction" width="400",align="right">
</p>


## Code
mgc.py




### Dependency
Python 3.6.5

numpy  version 1.14.2

pandas version 0.22.0

scipy  version 1.0.1

keras  version 2.1.3

PIL    version 5.0.0


## Complete Installation

1. Download Data from the link below

   GTZAN Dataset
   https://drive.google.com/open?id=16YkvirrRcdhqxix_0DBBjZeri4bjcfff

   FeatureMap
   https://drive.google.com/open?id=1yBGJUJL2UO9uatPtEDtRuAACqRTbdeiY

2. Locate the data in your working directory

3. Download and install libraries (See Dependancy)

4. run "mgc.py" (Comment out unnecessary classifiers)

5. If you use CNN, run "traintest.py" before you run "mgc.py"

6. Give me any advice


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


## Author

Akihiro Inui
http://www.portfolio.akihiroinui.com

### attribution


## License

MusicGenreClassification is available under the MIT license. See the LICENSE file for more info.

