# E2E_AudioVisualSpeechRecognition
![b](https://img.shields.io/github/license/lightning830/E2E_AudioVisualSpeechRecognition)

Conformer based early fusion model for audio-visual speech recognition

![a](https://github.com/lightning830/E2E_AudioVisualSpeechRecognition/blob/master/av_model2.png)

## Requirement
torch 1.10.0 + gpu\
matplotlib\
editdistance\
scipy\
opencv\
librosa\
conformer\
LRS2 dataset

## Usage
### 1. Edit config.py
Please change "project structure" in particular
### 2. Pretrain
Edit "PRETRAIN_NUM_WORDS" in config.py and run "pretrain.py".\
"Curriculum Learning" is used to pretrain in LRS2/pretrain.txt. The number of words used in each iterarion as follows:1,2,3,5,7,9,13,17,21,29,37. For each iterations, please terminate the training when the validation set WER flattens. 
### 3.Train
Run "train.py". \
Learning with the full text in LRS2/train.txt.
### 4.Test
Run "test.py". \
Test WER or CER in LRS2/test.txt.



## Reference
https://ken.ieice.org/ken/paper/20221022hCNR/\
N.Aoki (Department of Information Science, Tokyo Univ. of Sci.)
