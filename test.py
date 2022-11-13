

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import csv
import pandas as pd

from config import args
from models.av_net2 import AVNet
from models.loss import CustomLoss
from models.lrs2_char_lm import LRS2CharLM
from data.lrs2_dataset import LRS2Main
from data.utils import collate_fn
from utils.general import evaluate

import warnings
warnings.filterwarnings('ignore')


def main():
    output = 'output_noise_tmp.csv'
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda:0" if gpuAvailable else "cpu")
    kwargs = {"num_workers":args["NUM_WORKERS"], "pin_memory":True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    #declaring the test dataset and test dataloader
    audioParams = {"stftWindow":args["STFT_WINDOW"], "stftWinLen":args["STFT_WIN_LENGTH"], "stftOverlap":args["STFT_OVERLAP"]}
    videoParams = {"videoFPS":args["VIDEO_FPS"]}
    if args["TEST_DEMO_NOISY"]:
        noiseParams = {"noiseFile":args["NOISE_DIRECTORY"], "noiseProb":1, "noiseSNR":args["NOISE_SNR_DB"]}
    else:
        noiseParams = {"noiseFile":args["NOISE_DIRECTORY"], "noiseProb":0, "noiseSNR":args["NOISE_SNR_DB"]}
    testData = LRS2Main("test", args["DATA_DIRECTORY"], args["MAIN_REQ_INPUT_LENGTH"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                        audioParams, videoParams, noiseParams)
    testLoader = DataLoader(testData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn, shuffle=False, **kwargs)


    if args["TRAINED_MODEL_FILE"] is not None:

        print("\nTrained Model File: %s" %(args["TRAINED_MODEL_FILE"]))

        #declaring the model,loss function and loading the trained model weights
        model = AVNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
                      args["AUDIO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"], device=device)
        model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["TRAINED_MODEL_FILE"], map_location=device), strict=False)
        model.to(device)
        loss_function = CustomLoss(ramda = args["lambda"])


        #declaring the language model
        lm = LRS2CharLM()
        lm.load_state_dict(torch.load(args["TRAINED_LM_FILE"], map_location=device))
        lm.to(device)
        if not args["USE_LM"]:
            lm = None


        print("\nTesting the trained model .... \n")

        beamSearchParams = {"beamWidth":args["BEAM_WIDTH"], "alpha":args["LM_WEIGHT_ALPHA"], "beta":args["LENGTH_PENALTY_BETA"],
                            "threshProb":args["THRESH_PROBABILITY"]}
        # if args["TEST_DEMO_MODE"] == "AO":
        #     testParams = {"decodeScheme":args["TEST_DEMO_DECODING"], "beamSearchParams":beamSearchParams, "spaceIx":args["CHAR_TO_INDEX"][" "],
        #                   "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "lm":lm, "aoProb":1, "voProb":0}
        # elif args["TEST_DEMO_MODE"] == "VO":
        #     testParams = {"decodeScheme":args["TEST_DEMO_DECODING"], "beamSearchParams":beamSearchParams, "spaceIx":args["CHAR_TO_INDEX"][" "],
        #                   "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "lm":lm, "aoProb":0, "voProb":1}
        # elif args["TEST_DEMO_MODE"] == "AV":
        #     testParams = {"decodeScheme":args["TEST_DEMO_DECODING"], "beamSearchParams":beamSearchParams, "spaceIx":args["CHAR_TO_INDEX"][" "],
        #                   "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "lm":lm, "aoProb":0, "voProb":0}
        # else:
        #     print("Invalid Operation Mode.")
        #     exit()

        testParams = {"decodeScheme": args["TEST_DEMO_DECODING"], "spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "aoProb":0, "voProb":0, 
                            "beamSearchParams":beamSearchParams, "lm":lm}

        #evaluating the model over the test set
        testloss, testCER, testWER, testCER_att, testWER_att, predictionBatchs, predictionBatchs_att, targetBatchs =\
             evaluate(model, testLoader, loss_function, device, testParams)

        predict_list = []
        for predict in predictionBatchs_att:
            predict = predict.numpy()
            predict_str = ''
            for c in predict:
                _c = args["INDEX_TO_CHAR"][c]
                predict_str += _c
                if _c == '<EOS>':
                    predict_list.append(predict_str)
                    predict_str = ''

        target_list = []
        for target in targetBatchs:
            target = target.numpy()
            target_str = ''
            for c in target:
                _c = args["INDEX_TO_CHAR"][c]
                target_str += _c
                if _c == '<EOS>':
                    target_list.append(target_str)
                    target_str = ''

        with open(output, "w", newline="") as f:
            writer = csv.writer(f)
            # print(target_list)
            for i,j in zip(target_list, predict_list):
                writer.writerow([i,j])

        #printing the test set loss, CER and WER
        # print("Test Loss: %.6f || Test CER: %.3f || Test WER: %.3f" %(testLoss, testCER, testWER))
        print("Loss: %.6f ||CER: %.3f  attCER: %.3f WER: %.3f attWER: %.3f"
              %(testloss, testCER, testCER_att, testWER, testWER_att))
        print("\nTesting Done.\n")


    else:
        print("Path to the trained model file not specified.\n")

    return



if __name__ == "__main__":
    main()
