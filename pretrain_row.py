
import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import os, shutil

from config import args
from models.av_net3 import AVNet
from models.loss import CustomLoss
from data.lrs2_dataset_row import LRS2Pretrain
from data.utils import collate_fn_row
from utils.general import num_params, train, evaluate

import warnings
warnings.filterwarnings('ignore')



def main():

    matplotlib.use("Agg")
    np.random.seed(args["SEED"])
    torch.manual_seed(args["SEED"])
    gpuAvailable = torch.cuda.is_available()
    device = torch.device("cuda" if gpuAvailable else "cpu")
    kwargs = {"num_workers": args["NUM_WORKERS"], "pin_memory": True} if gpuAvailable else {}
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    #declaring the pretrain and the preval datasets and the corresponding dataloaders
    audioParams = {"stftWindow":args["STFT_WINDOW"], "stftWinLen":args["STFT_WIN_LENGTH"], "stftOverlap":args["STFT_OVERLAP"]}
    videoParams = {"videoFPS":args["VIDEO_FPS"]}
    noiseParams = {"noiseFile":args["NOISE_DIRECTORY"], "noiseProb":args["NOISE_PROBABILITY"], "noiseSNR":args["NOISE_SNR_DB"]}
    pretrainData = LRS2Pretrain("pretrain", args["DATA_DIRECTORY"], args["PRETRAIN_NUM_WORDS"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                                audioParams, videoParams, noiseParams)
    pretrainLoader = DataLoader(pretrainData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn_row, shuffle=True, **kwargs)
    noiseParams = {"noiseFile":args["NOISE_DIRECTORY"], "noiseProb":0, "noiseSNR":args["NOISE_SNR_DB"]}
    prevalData = LRS2Pretrain("preval", args["DATA_DIRECTORY"], args["PRETRAIN_NUM_WORDS"], args["CHAR_TO_INDEX"], args["STEP_SIZE"],
                              audioParams, videoParams, noiseParams)
    prevalLoader = DataLoader(prevalData, batch_size=args["BATCH_SIZE"], collate_fn=collate_fn_row, shuffle=True, **kwargs)


    #declaring the model, optimizer, scheduler and the loss function
    model = AVNet(args["TX_NUM_FEATURES"], args["TX_ATTENTION_HEADS"], args["TX_NUM_LAYERS"], args["PE_MAX_LENGTH"],
                  args["AUDIO_FEATURE_SIZE"], args["TX_FEEDFORWARD_DIM"], args["TX_DROPOUT"], args["NUM_CLASSES"])
    model.to(device)
    # model = torch.nn.DataParallel(model)
    optimizer = optim.Adam(model.parameters(), lr=args["INIT_LR"], betas=(args["MOMENTUM1"], args["MOMENTUM2"]))
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=args["LR_SCHEDULER_FACTOR"],
                                                     patience=args["LR_SCHEDULER_WAIT"], threshold=args["LR_SCHEDULER_THRESH"],
                                                     threshold_mode="abs", min_lr=args["FINAL_LR"], verbose=True)
    loss_function = CustomLoss(ramda = args["lambda"])



    #removing the checkpoints directory if it exists and remaking it
    if os.path.exists(args["CODE_DIRECTORY"] + "/audio_visual/checkpoints"):
        while True:
            ch = input("Continue and remove the 'checkpoints' directory? y/n: ")
            if ch == "y":
                break
            elif ch == "n":
                exit()
            else:
                print("Invalid input")
        shutil.rmtree(args["CODE_DIRECTORY"] + "/audio_visual/checkpoints")

    os.mkdir(args["CODE_DIRECTORY"] + "/audio_visual/checkpoints")
    os.mkdir(args["CODE_DIRECTORY"] + "/audio_visual/checkpoints/models")
    os.mkdir(args["CODE_DIRECTORY"] + "/audio_visual/checkpoints/plots")


    #loading the pretrained weights
    if args["PRETRAINED_MODEL_FILE"] is not None:
        print("\n\nPre-trained Model File: %s" %(args["PRETRAINED_MODEL_FILE"]))
        print("\nLoading the pre-trained model .... \n")
        model.load_state_dict(torch.load(args["CODE_DIRECTORY"] + args["PRETRAINED_MODEL_FILE"], map_location=device))
        model.to(device)
        print("Loading Done.\n")



    trainingLossCurve = list()
    validationLossCurve = list()
    trainingWERCurve = list()
    validationWERCurve = list()
    validationWERattCurve = list()


    #printing the total and trainable parameters in the model
    numTotalParams, numTrainableParams = num_params(model)
    print("\nNumber of total parameters in the model = %d" %(numTotalParams))
    print("Number of trainable parameters in the model = %d\n" %(numTrainableParams))

    print("Number of Words = %d" %(args["PRETRAIN_NUM_WORDS"]))
    print("\nPretraining the model .... \n")

    trainParams = {"spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "aoProb":args["AUDIO_ONLY_PROBABILITY"],
                   "voProb":args["VIDEO_ONLY_PROBABILITY"]}
    valParams = {"decodeScheme":"greedy", "spaceIx":args["CHAR_TO_INDEX"][" "], "eosIx":args["CHAR_TO_INDEX"]["<EOS>"], "aoProb":0, "voProb":0}

    for step in range(args["NUM_STEPS"]):

        #train the model for one step
        trainingLoss, trainingCER, trainingWER = train(model, pretrainLoader, optimizer, loss_function, device, trainParams)
        trainingLossCurve.append(trainingLoss)
        trainingWERCurve.append(trainingWER)

        #evaluate the model on validation set
        validationLoss, validationCER, validationWER, validationCER_att, validationWER_att, predictionBatchs, predictionBatchs_att, targetBatchs =\
             evaluate(model, prevalLoader, loss_function, device, valParams)
        validationLossCurve.append(validationLoss)
        validationWERCurve.append(validationWER)
        validationWERattCurve.append(validationWER_att)

        #printing the stats after each step
        print("Step: %03d || Tr.Loss: %.6f Val.Loss: %.6f || Tr.CER: %.3f  Val.CER: %.3f attCER: %.3f || Tr.WER: %.3f  Val.WER: %.3f attWER: %.3f"
              %(step, trainingLoss, validationLoss, trainingCER, validationCER, validationCER_att, trainingWER, validationWER, validationWER_att))

        #make a scheduler step
        scheduler.step(validationWER)


        #saving the model weights and loss/metric curves in the checkpoints directory after every few steps
        if ((step%args["SAVE_FREQUENCY"] == 0) or (step == args["NUM_STEPS"]-1)) and (step != 0):

            savePath = args["CODE_DIRECTORY"] + "/audio_visual/checkpoints/models/pretrain_{:03d}w-step_{:04d}-wer_{:.3f}.pt".format(args["PRETRAIN_NUM_WORDS"],
                                                                                                                        step, validationWER)
            torch.save(model.state_dict(), savePath)

            plt.figure()
            plt.title("Loss Curves")
            plt.xlabel("Step No.")
            plt.ylabel("Loss value")
            plt.plot(list(range(1, len(trainingLossCurve)+1)), trainingLossCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationLossCurve)+1)), validationLossCurve, "red", label="Validation")
            plt.legend()
            plt.savefig(args["CODE_DIRECTORY"] + "/audio_visual/checkpoints/plots/pretrain_{:03d}w-step_{:04d}-loss.png".format(args["PRETRAIN_NUM_WORDS"], step))
            plt.close()

            plt.figure()
            plt.title("WER Curves")
            plt.xlabel("Step No.")
            plt.ylabel("WER")
            plt.plot(list(range(1, len(trainingWERCurve)+1)), trainingWERCurve, "blue", label="Train")
            plt.plot(list(range(1, len(validationWERCurve)+1)), validationWERCurve, "red", label="Validation")
            plt.plot(list(range(1, len(validationWERattCurve)+1)), validationWERattCurve, "green", label="Validation_att")
            plt.legend()
            plt.savefig(args["CODE_DIRECTORY"] + "/audio_visual/checkpoints/plots/pretrain_{:03d}w-step_{:04d}-wer.png".format(args["PRETRAIN_NUM_WORDS"], step))
            plt.close()


    print("\nPretraining Done.\n")

    return



if __name__ == "__main__":
    main()