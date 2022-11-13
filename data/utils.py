"""
Author: Smeet Shah
Copyright (c) 2020 Smeet Shah
File part of 'deep_avsr' GitHub repository available at -
https://github.com/lordmartian/deep_avsr
"""

import torch
from torch.nn.utils.rnn import pad_sequence
import numpy as np
from scipy import signal
from scipy.io import wavfile
import cv2 as cv
from scipy.special import softmax
import librosa, random
import sys,os
# sys.path.append("../")# 実行しているファイルがpretrain.pyなのでそこからのspecaugment相対パス
from config import args
sys.path.append(args["CODE_DIRECTORY"])
from specaugment import freq_mask, time_mask



def prepare_main_input(audioFile, visualFeaturesFile, targetFile, noise, reqInpLen, charToIx, noiseSNR, audioParams, videoParams, noise_file=None):

    """
    Function to convert the data sample in the main dataset into appropriate tensors.
    """

    if targetFile is not None:

        #reading the target from the target file and converting each character to its corresponding index
        with open(targetFile, "r") as f:
            trgt = f.readline().strip()[7:]

        # tmp: write nosie_file
        if noise_file is not None:
            output = "/home/python/deep_avsr_conformer/audio_visual/noise_output.csv"
            with open(output ,'a') as file:
                # file.write(trgt)
                print(trgt+','+str(noise_file), file=file)            


        #converting each character in target to its corresponding index
        trgt = [charToIx[char] for char in trgt]

        trgt_input = np.array(trgt)
        trgt_input = np.insert(trgt_input, 0, charToIx["<BOS>"]) #先頭にBOSを配置、input用

        trgt.append(charToIx["<EOS>"])
        trgt = np.array(trgt)
        trgtLen = len(trgt)

        #the target length must be less than or equal to 100 characters (restricted space where our model will work)
        if trgtLen > 100:
            print("Target length more than 100 characters. Exiting")
            exit()

    #STFT feature extraction
    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]
    sampFreq, inputAudio = wavfile.read(audioFile)

    #pad the audio to get atleast 4 STFT vectors
    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    inputAudio = inputAudio/np.max(np.abs(inputAudio))

    #adding noise to the audio
    if noise is not None:
        noiseSNR_ = random.choice(noiseSNR)
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        max_noise = np.max(np.abs(noise))
        if max_noise > 0: #when select clean section, pass this module
            noise = noise/max_noise
            gain = 10**(noiseSNR_/10)
            noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
            inputAudio = inputAudio + noise

    #音声をtmpに抽出
    p = '/home/python/deep_avsr_conformer/audio_visual/tmp2/'+audioFile.split('/')[-2]
    if os.path.isdir(p) == False:
        os.mkdir(p)
    p += '/'+audioFile.split('/')[-1]
    data = inputAudio*np.max(np.abs(inputAudio))
    wavfile.write(p, rate=sampFreq, data=data)
    
    #normalising the audio to unit power
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))
    #computing STFT and taking only the magnitude of it
    stftVals = librosa.feature.melspectrogram(inputAudio, sr=sampFreq, window=stftWindow, n_mels=80,\
        n_fft=int(sampFreq*stftWinLen), hop_length=int(sampFreq*(stftWinLen-stftOverlap)))
    audInp = librosa.power_to_db(stftVals)
    audInp = audInp.T


    #loading the visual features
    vidInp = np.load(visualFeaturesFile)
    vidInp = np.squeeze(vidInp)


    #padding zero vectors to extend the audio and video length to a least possible integer length such that
    #video length = 4 * audio length
    if len(audInp)/4 >= len(vidInp):
        inpLen = int(np.ceil(len(audInp)/4))
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")
        leftPadding = int(np.floor((inpLen - len(vidInp))/2))
        rightPadding = int(np.ceil((inpLen - len(vidInp))/2))
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")
    else:
        inpLen = len(vidInp)
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")


    #checking whether the input length is greater than or equal to the required length
    #if not, extending the input by padding zero vectors
    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        audInp = np.pad(audInp, ((4*leftPadding,4*rightPadding),(0,0)), "constant")
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")

    inpLen = len(vidInp)

    # specaugment
    audInp = torch.from_numpy(audInp) #tensor cast because specaugment needs tensor
    audInp = audInp.transpose(0,1).unsqueeze(0) #(1, Dim=80, T)
    t_func = lambda x: int(x*args["T_RATIO"]) if int(x*args["T_RATIO"]) > 1 else 1 # calcurate size of time mask
    audInp = time_mask(freq_mask(audInp, F=args["F_SIZE"], num_masks=args["M_F"]),T=t_func(audInp.shape[2]), num_masks=args["M_T"])
    audInp = audInp.squeeze(0).transpose(0,1)


    vidInp = torch.from_numpy(vidInp)
    inp = (audInp,vidInp)
    inpLen = torch.tensor(inpLen)
    if targetFile is not None:
        trgt = torch.from_numpy(trgt)
        trgtLen = torch.tensor(trgtLen)
        trgt_input = torch.tensor(trgt_input) #先頭がBOSになってEOSがないtrgt

    else:
        trgt, trgtLen, trgt_input = None, None, None

    return inp, trgt, inpLen, trgtLen, trgt_input


def prepare_main_input_row(audioFile, visualFeaturesFile, targetFile, noise, reqInpLen, charToIx, noiseSNR, audioParams, videoParams):

    """
    Function to convert the data sample in the main dataset into appropriate tensors.
    """

    if targetFile is not None:

        #reading the target from the target file and converting each character to its corresponding index
        with open(targetFile, "r") as f:
            trgt = f.readline().strip()[7:]

        #converting each character in target to its corresponding index
        trgt = [charToIx[char] for char in trgt]

        trgt_input = np.array(trgt)
        trgt_input = np.insert(trgt_input, 0, charToIx["<BOS>"]) #先頭にBOSを配置、input用

        trgt.append(charToIx["<EOS>"])
        trgt = np.array(trgt)
        trgtLen = len(trgt)

        #the target length must be less than or equal to 100 characters (restricted space where our model will work)
        if trgtLen > 100:
            print("Target length more than 100 characters. Exiting")
            exit()

    #STFT feature extraction
    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]
    sampFreq, inputAudio = wavfile.read(audioFile)

    #pad the audio to get atleast 4 STFT vectors
    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    inputAudio = inputAudio/np.max(np.abs(inputAudio))

    #adding noise to the audio
    if noise is not None:
        noiseSNR_ = random.choice(noiseSNR)
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        max_noise = np.max(np.abs(noise))
        if max_noise > 0: #when select clean section, pass this module
            noise = noise/max_noise
            gain = 10**(noiseSNR_/10)
            noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
            inputAudio = inputAudio + noise

    #normalising the audio to unit power
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))

    #computing STFT and taking only the magnitude of it
    stftVals = librosa.feature.melspectrogram(inputAudio, sr=sampFreq, window=stftWindow, n_mels=80,\
        n_fft=int(sampFreq*stftWinLen), hop_length=int(sampFreq*(stftWinLen-stftOverlap)))
    audInp = librosa.power_to_db(stftVals)
    audInp = audInp.T


    #loading the visual features
    vidInp = np.load(visualFeaturesFile)
    vidInp = np.squeeze(vidInp)


    #padding zero vectors to extend the audio and video length to a least possible integer length such that
    #video length = 4 * audio length
    if len(audInp)/4 >= len(vidInp):
        inpLen = int(np.ceil(len(audInp)/4))
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")
        leftPadding = int(np.floor((inpLen - len(vidInp))/2))
        rightPadding = int(np.ceil((inpLen - len(vidInp))/2))
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")
    else:
        inpLen = len(vidInp)
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")


    #checking whether the input length is greater than or equal to the required length
    #if not, extending the input by padding zero vectors
    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        audInp = np.pad(audInp, ((4*leftPadding,4*rightPadding),(0,0)), "constant")
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")

    inpLen = len(vidInp)

    # specaugment
    audInp = torch.from_numpy(audInp) #tensor cast because specaugment needs tensor
    audInp = audInp.transpose(0,1).unsqueeze(0) #(1, Dim=80, T)
    t_func = lambda x: int(x*args["T_RATIO"]) if int(x*args["T_RATIO"]) > 1 else 1 # calcurate size of time mask
    audInp = time_mask(freq_mask(audInp, F=args["F_SIZE"], num_masks=args["M_F"]),T=t_func(audInp.shape[2]), num_masks=args["M_T"])
    audInp = audInp.squeeze(0).transpose(0,1)


    vidInp = torch.from_numpy(vidInp)
    inp = (audInp,vidInp)
    inpLen = torch.tensor(inpLen)
    if targetFile is not None:
        trgt = torch.from_numpy(trgt)
        trgtLen = torch.tensor(trgtLen)
        trgt_input = torch.tensor(trgt_input) #先頭がBOSになってEOSがないtrgt

    else:
        trgt, trgtLen, trgt_input = None, None, None

    return inp, trgt, inpLen, trgtLen, trgt_input


def prepare_pretrain_input(audioFile, visualFeaturesFile, targetFile, noise, numWords, charToIx, noiseSNR, audioParams, videoParams):

    """
    Function to convert the data sample in the pretrain dataset into appropriate tensors.
    """

    #reading the whole target file and the target
    with open(targetFile, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    trgt = lines[0][7:]
    words = trgt.split(" ")

    #if number of words in target is less than the required number of words, consider the whole target
    if len(words) <= numWords:
        trgtNWord = trgt
        sampFreq, inputAudio = wavfile.read(audioFile)
        vidInp = np.load(visualFeaturesFile)
        vidInp = np.squeeze(vidInp)

    else:
        #make a list of all possible sub-sequences with required number of words in the target
        nWords = [" ".join(words[i:i+numWords]) for i in range(len(words)-numWords+1)]
        nWordLens = np.array([len(nWord)+1 for nWord in nWords]).astype(np.float)

        #choose the sub-sequence for target according to a softmax distribution of the lengths
        #this way longer sub-sequences (which are more diverse) are selected more often while
        #the shorter sub-sequences (which appear more frequently) are not entirely missed out
        ix = np.random.choice(np.arange(len(nWordLens)), p=softmax(nWordLens))
        trgtNWord = nWords[ix]

        #reading the start and end times in the video corresponding to the selected sub-sequence
        startTime = float(lines[4+ix].split(" ")[1])
        endTime = float(lines[4+ix+numWords-1].split(" ")[2])
        #loading the audio
        sampFreq, audio = wavfile.read(audioFile)
        inputAudio = audio[int(sampFreq*startTime):int(sampFreq*endTime)]
        #loading visual features
        videoFPS = videoParams["videoFPS"]
        vidInp = np.load(visualFeaturesFile)
        vidInp = np.squeeze(vidInp)
        vidInp = vidInp[int(np.floor(videoFPS*startTime)):int(np.ceil(videoFPS*endTime))]


    #converting each character in target to its corresponding index
    trgt = [charToIx[char] for char in trgtNWord]

    trgt_input = np.array(trgt)
    trgt_input = np.insert(trgt_input, 0, charToIx["<BOS>"]) #先頭にBOSを配置、input用

    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)


    #STFT feature extraction
    stftWindow = audioParams["stftWindow"]
    stftWinLen = audioParams["stftWinLen"]
    stftOverlap = audioParams["stftOverlap"]

    #pad the audio to get atleast 4 STFT vectors
    if len(inputAudio) < sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)):
        padding = int(np.ceil((sampFreq*(stftWinLen + 3*(stftWinLen - stftOverlap)) - len(inputAudio))/2))
        inputAudio = np.pad(inputAudio, padding, "constant")
    inputAudio = inputAudio/np.max(np.abs(inputAudio))

    #adding noise to the audio
    if noise is not None:
        noiseSNR_ = random.choice(noiseSNR)
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        max_noise = np.max(np.abs(noise))
        if max_noise > 0: #when select clean section, pass this module
            noise = noise/max_noise
            gain = 10**(noiseSNR_/10)
            noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
            inputAudio = inputAudio + noise

    #normalising the audio to unit power
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))

    #computing the STFT and taking only the magnitude of it
    stftVals = librosa.feature.melspectrogram(inputAudio, sr=sampFreq, window=stftWindow, n_mels=80,\
         n_fft=int(sampFreq*stftWinLen), hop_length=int(sampFreq*(stftWinLen-stftOverlap)))
    audInp = librosa.power_to_db(stftVals)
    # import matplotlib.pyplot as plt
    # plt.imsave('/home/python/deep_avsr_conformer/audio_visual/tmp/'+str(noiseSNR_)+'.jpg', audInp)
    audInp = audInp.T


    #padding zero vectors to extend the audio and video length to a least possible integer length such that
    #video length = 4 * audio length
    if len(audInp)/4 >= len(vidInp):
        inpLen = int(np.ceil(len(audInp)/4))
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")
        leftPadding = int(np.floor((inpLen - len(vidInp))/2))
        rightPadding = int(np.ceil((inpLen - len(vidInp))/2))
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")
    else:
        inpLen = len(vidInp)
        leftPadding = int(np.floor((4*inpLen - len(audInp))/2))
        rightPadding = int(np.ceil((4*inpLen - len(audInp))/2))
        audInp = np.pad(audInp, ((leftPadding,rightPadding),(0,0)), "constant")


    #checking whether the input length is greater than or equal to the required length
    #if not, extending the input by padding zero vectors (FOR CTC LOSS INF AVOIDING!!!!)
    reqInpLen = req_input_length(trgt)
    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        audInp = np.pad(audInp, ((4*leftPadding,4*rightPadding),(0,0)), "constant")
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")

    inpLen = len(vidInp)

    # specaugment
    audInp = torch.from_numpy(audInp) #tensor cast because specaugment needs tensor
    audInp = audInp.transpose(0,1).unsqueeze(0) #(1, Dim=80, T)
    t_func = lambda x: int(x*args["T_RATIO"]) if int(x*args["T_RATIO"]) > 1 else 1 # calcurate size of time mask
    audInp = time_mask(freq_mask(audInp, F=args["F_SIZE"], num_masks=args["M_F"]),T=t_func(audInp.shape[2]), num_masks=args["M_T"])
    audInp = audInp.squeeze(0).transpose(0,1)

    # audInp = (T, Dim=80)
    # vidInp = (T/4, Dim=512)
    vidInp = torch.from_numpy(vidInp)
    inp = (audInp,vidInp)
    inpLen = torch.tensor(inpLen)
    trgt = torch.from_numpy(trgt)
    trgtLen = torch.tensor(trgtLen)
    trgt_input = torch.tensor(trgt_input) #先頭がBOSになってEOSがないtrgt


    return inp, trgt, inpLen, trgtLen, trgt_input


def prepare_pretrain_input_row(audioFile, visualFeaturesFile, targetFile, noise, numWords, charToIx, noiseSNR, audioParams, videoParams):

    """
    Function to convert the data sample in the pretrain dataset into appropriate tensors.
    """

    #reading the whole target file and the target
    with open(targetFile, "r") as f:
        lines = f.readlines()
    lines = [line.strip() for line in lines]

    trgt = lines[0][7:]
    words = trgt.split(" ")

    #if number of words in target is less than the required number of words, consider the whole target
    if len(words) <= numWords:
        trgtNWord = trgt
        sampFreq, inputAudio = wavfile.read(audioFile)
        vidInp = np.load(visualFeaturesFile)
        vidInp = np.squeeze(vidInp)

    else:
        #make a list of all possible sub-sequences with required number of words in the target
        nWords = [" ".join(words[i:i+numWords]) for i in range(len(words)-numWords+1)]
        nWordLens = np.array([len(nWord)+1 for nWord in nWords]).astype(np.float)

        #choose the sub-sequence for target according to a softmax distribution of the lengths
        #this way longer sub-sequences (which are more diverse) are selected more often while
        #the shorter sub-sequences (which appear more frequently) are not entirely missed out
        ix = np.random.choice(np.arange(len(nWordLens)), p=softmax(nWordLens))
        trgtNWord = nWords[ix]

        #reading the start and end times in the video corresponding to the selected sub-sequence
        startTime = float(lines[4+ix].split(" ")[1])
        endTime = float(lines[4+ix+numWords-1].split(" ")[2])
        #loading the audio
        sampFreq, audio = wavfile.read(audioFile)
        inputAudio = audio[int(sampFreq*startTime):int(sampFreq*endTime)]
        #loading visual features
        videoFPS = videoParams["videoFPS"]
        vidInp = np.load(visualFeaturesFile)
        vidInp = np.squeeze(vidInp)
        vidInp = vidInp[int(np.floor(videoFPS*startTime)):int(np.ceil(videoFPS*endTime))]


    #converting each character in target to its corresponding index
    trgt = [charToIx[char] for char in trgtNWord]

    trgt_input = np.array(trgt)
    trgt_input = np.insert(trgt_input, 0, charToIx["<BOS>"]) #先頭にBOSを配置、input用

    trgt.append(charToIx["<EOS>"])
    trgt = np.array(trgt)
    trgtLen = len(trgt)


    #adding noise to the audio
    if noise is not None:
        noiseSNR_ = random.choice(noiseSNR)
        pos = np.random.randint(0, len(noise)-len(inputAudio)+1)
        noise = noise[pos:pos+len(inputAudio)]
        max_noise = np.max(np.abs(noise))
        if max_noise > 0: #when select clean section, pass this module
            noise = noise/max_noise
            gain = 10**(noiseSNR_/10)
            noise = noise*np.sqrt(np.sum(inputAudio**2)/(gain*np.sum(noise**2)))
            inputAudio = inputAudio + noise

    #normalising the audio to unit power
    inputAudio = inputAudio/np.sqrt(np.sum(inputAudio**2)/len(inputAudio))
    inpLen = len(vidInp)

    if len(inputAudio) < inpLen*640:
        d = abs(len(inputAudio)-inpLen*640)
        inputAudio = np.pad(inputAudio, (d//2, d-d//2))
    elif len(inputAudio) > inpLen*640:
        d = abs(len(inputAudio)-inpLen*640)
        inputAudio = inputAudio[d//2:-(d-d//2)]

    #checking whether the input length is greater than or equal to the required length
    #if not, extending the input by padding zero vectors (FOR CTC LOSS INF AVOIDING!!!!)
    reqInpLen = req_input_length(trgt)
    if inpLen < reqInpLen:
        leftPadding = int(np.floor((reqInpLen - inpLen)/2))
        rightPadding = int(np.ceil((reqInpLen - inpLen)/2))
        inputAudio = np.pad(inputAudio, ((640*leftPadding,640*rightPadding)), "constant")
        vidInp = np.pad(vidInp, ((leftPadding,rightPadding),(0,0)), "constant")

    inputAudio = np.nan_to_num(inputAudio)
    audInp = torch.from_numpy(inputAudio) #tensor cast because specaugment needs tensor
    inpLen = len(vidInp)

    # audInp = (T)
    # vidInp = (T/4, Dim=512)

    vidInp = torch.from_numpy(vidInp)
    inp = (audInp,vidInp)
    inpLen = torch.tensor(inpLen)
    trgt = torch.from_numpy(trgt)
    trgtLen = torch.tensor(trgtLen)
    trgt_input = torch.tensor(trgt_input) #先頭がBOSになってEOSがないtrgt


    return inp, trgt, inpLen, trgtLen, trgt_input


def collate_fn(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    inputBatch = (pad_sequence([data[0][0] for data in dataBatch]),
                  pad_sequence([data[0][1] for data in dataBatch]))
    targetBatch_2d = pad_sequence([data[1] for data in dataBatch])

    if not any(data[1] is None for data in dataBatch):
        targetBatch_1d = torch.cat([data[1] for data in dataBatch])
    else:
        targetBatch_1d = None

    inputLenBatch = torch.stack([data[2] for data in dataBatch])
    if not any(data[3] is None for data in dataBatch):
        targetLenBatch = torch.stack([data[3] for data in dataBatch])
    else:
        targetLenBatch = None

    targetBatch_2d_input = pad_sequence([data[4] for data in dataBatch])
    

    return inputBatch, (targetBatch_1d, targetBatch_2d, targetBatch_2d_input), inputLenBatch, targetLenBatch


def collate_fn_row(dataBatch):
    """
    Collate function definition used in Dataloaders.
    """
    inputBatch = (pad_sequence([data[0][0] for data in dataBatch]),
                  pad_sequence([data[0][1] for data in dataBatch]))
    targetBatch_2d = pad_sequence([data[1] for data in dataBatch])

    if not any(data[1] is None for data in dataBatch):
        targetBatch_1d = torch.cat([data[1] for data in dataBatch])
    else:
        targetBatch_1d = None

    inputLenBatch = torch.stack([data[2] for data in dataBatch])
    if not any(data[3] is None for data in dataBatch):
        targetLenBatch = torch.stack([data[3] for data in dataBatch])
    else:
        targetLenBatch = None

    targetBatch_2d_input = pad_sequence([data[4] for data in dataBatch])
    

    return inputBatch, (targetBatch_1d, targetBatch_2d, targetBatch_2d_input), inputLenBatch, targetLenBatch


def req_input_length(trgt):
    """
    Function to calculate the minimum required input length from the target.
    Req. Input Length = No. of unique chars in target + No. of repeats in repeated chars (excluding the first one)
    """
    reqLen = len(trgt)
    lastChar = trgt[0]
    for i in range(1, len(trgt)):
        if trgt[i] != lastChar:
            lastChar = trgt[i]
        else:
            reqLen = reqLen + 1
    return reqLen
