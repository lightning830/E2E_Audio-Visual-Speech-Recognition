

from torch.utils.data import Dataset
from scipy.io import wavfile
import numpy as np
import random, glob, os
from config import args

from .utils import prepare_pretrain_input
from .utils import prepare_main_input



class LRS2Pretrain(Dataset):

    """
    A custom dataset class for the LRS2 pretrain (includes pretain, preval) dataset.
    """

    def __init__(self, dataset, datadir, numWords, charToIx, stepSize, audioParams, videoParams, noiseParams):
        super(LRS2Pretrain, self).__init__()
        with open(datadir + "/" + dataset + "U600.txt", "r") as f:
            lines = f.readlines()
        self.datalist = [datadir + "/pretrain/" + line.strip() for line in lines]
        self.numWords = numWords
        self.charToIx = charToIx
        self.dataset = dataset
        self.stepSize = stepSize
        self.audioParams = audioParams
        self.videoParams = videoParams
        self.noisefile = noiseParams["noiseFile"] # niose directory
        self.noises = []
        for p in glob.glob(self.noisefile+'/*'):
            _, noise = wavfile.read(p)
            self.noises.append(noise)
        self.noiseProb = noiseParams["noiseProb"]
        self.noiseSNR = noiseParams["noiseSNR"]
        return


    def __getitem__(self, index):
        if self.dataset == "pretrain":
            #index goes from 0 to stepSize-1
            #dividing the dataset into partitions of size equal to stepSize and selecting a random partition
            #fetch the sample at position 'index' in this randomly selected partition
            base = self.stepSize * np.arange(int(len(self.datalist)/self.stepSize)+1)
            ixs = base + index
            ixs = ixs[ixs < len(self.datalist)]
            index = np.random.choice(ixs)

        #passing the sample files and the target file paths to the prepare function to obtain the input tensors
        audioFile = self.datalist[index] + ".wav"
        visualFeaturesFile = self.datalist[index] + ".npy"
        targetFile = self.datalist[index] + ".txt"
        if np.random.choice([True, False], p=[self.noiseProb, 1-self.noiseProb]):# select add noise or not
            noise = random.choice(self.noises) # random selected noise file path
        else:
            noise = None
        inp, trgt, inpLen, trgtLen, trgt_input = prepare_pretrain_input(audioFile, visualFeaturesFile, targetFile, noise, self.numWords,
                                                            self.charToIx, self.noiseSNR, self.audioParams, self.videoParams)
        return inp, trgt, inpLen, trgtLen, trgt_input


    def __len__(self):
        #each iteration covers only a random subset of all the training samples whose size is given by the step size
        #this is done only for the pretrain set, while the whole preval set is considered
        if self.dataset == "pretrain":
            return self.stepSize
        else:
            return len(self.datalist)



class LRS2Main(Dataset):

    """
    A custom dataset class for the LRS2 main (includes train, val, test) dataset
    """

    def __init__(self, dataset, datadir, reqInpLen, charToIx, stepSize, audioParams, videoParams, noiseParams):
        super(LRS2Main, self).__init__()
        self.random_seed = args["SEED"]
        with open(datadir + "/" + dataset + "U600.txt", "r") as f:
            lines = f.readlines()
        self.datalist = [datadir + "/main/" + line.strip().split(" ")[0] for line in lines]
        self.reqInpLen = reqInpLen
        self.charToIx = charToIx
        self.dataset = dataset
        self.stepSize = stepSize
        self.audioParams = audioParams
        self.videoParams = videoParams
        self.noisefile = noiseParams["noiseFile"] # niose directory
        self.noises = []
        self.noise_number = [] # main only for memorizeing what kind of noise
        for p in glob.glob(self.noisefile+'/*'):
            _, noise = wavfile.read(p)
            self.noises.append(noise)
            self.noise_number.append(os.path.basename(p))
        random.seed(self.random_seed)
        self.noise_select = [random.randint(0, len(self.noise_number)-1) for i in range(1500)]
        self.noiseSNR = noiseParams["noiseSNR"]
        self.noiseProb = noiseParams["noiseProb"]
        return


    def __getitem__(self, index):
        #using the same procedure as in pretrain dataset class only for the train dataset
        if self.dataset == "train":
            base = self.stepSize * np.arange(int(len(self.datalist)/self.stepSize)+1)
            ixs = base + index
            ixs = ixs[ixs < len(self.datalist)]
            index = np.random.choice(ixs)

        #passing the sample files and the target file paths to the prepare function to obtain the input tensors
        audioFile = self.datalist[index] + ".wav"
        visualFeaturesFile = self.datalist[index] + ".npy"
        targetFile = self.datalist[index] + ".txt"
        if np.random.choice([True, False], p=[self.noiseProb, 1-self.noiseProb]):
            num = self.noise_select[index]
            num = 29
            noise = self.noises[num]
            noise_file = self.noise_number[num]
            
            # noise = random.choice(self.noises) # random selected noise file path

            # random.seed(self.random_seed)
            # num = random.randint(0, len(self.noise_number)-1) # random selected noise number
            # self.random_seed += 1
            # noise = self.noises[num]
            # noise_file = self.noise_number[num]
        else:
            noise = None
            noise_file = None
        inp, trgt, inpLen, trgtLen, trgt_input = prepare_main_input(audioFile, visualFeaturesFile, targetFile, noise, self.reqInpLen, self.charToIx,
                                                        self.noiseSNR, self.audioParams, self.videoParams, noise_file)
        return inp, trgt, inpLen, trgtLen, trgt_input


    def __len__(self):
        #using step size only for train dataset and not for val and test datasets because
        #the size of val and test datasets is smaller than step size and we generally want to validate and test
        #on the complete dataset
        if self.dataset == "train":
            return self.stepSize
        else:
            return len(self.datalist)
