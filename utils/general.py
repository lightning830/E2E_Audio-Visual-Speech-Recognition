
import torch
import numpy as np
from tqdm import tqdm

from .metrics import compute_cer, compute_wer, compute_cer_att, compute_wer_att
from .decoders import ctc_greedy_decode, ctc_search_decode, att_greedy_decode, att_search_decode



def num_params(model):
    """
    Function that outputs the number of total and trainable paramters in the model.
    """
    numTotalParams = sum([params.numel() for params in model.parameters()])
    numTrainableParams = sum([params.numel() for params in model.parameters() if params.requires_grad])
    return numTotalParams, numTrainableParams



def train(model, trainLoader, optimizer, loss_function, device, trainParams):

    """
    Function to train the model for one iteration. (Generally, one iteration = one epoch, but here it is one step).
    It also computes the training loss, CER and WER. The CTC decode scheme is always 'greedy' here.
    """

    trainingLoss = 0
    ctcLoss = 0
    attLoss = 0
    trainingCER = 0
    trainingWER = 0
    # trainingCER_att = 0
    # trainingWER_att = 0

    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(tqdm(trainLoader, leave=False, desc="Train",
                                                                                          ncols=75)):
        # (T, B, D=80), (sumL,), (L, B), (L+bos, B)
        inputBatch, targetBatch, targetBatch_2d, targetBatch_2d_input = \
            ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device)), (targetBatch[0].int()).to(device), (targetBatch[1].int()).to(device), (targetBatch[2].int()).to(device)
        # (B,), (B,)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)

        opmode = np.random.choice(["AO", "VO", "AV"],
                                  p=[trainParams["aoProb"], trainParams["voProb"], 1-(trainParams["aoProb"]+trainParams["voProb"])])
        if opmode == "AO":
            inputBatch = (inputBatch[0], None)
        elif opmode == "VO":
            inputBatch = (None, inputBatch[1])
        else:
            pass

        optimizer.zero_grad()
        model.train()

        # (T, B, C) (L, B, C)
        outputBatch_ctc, outputBatch_att = model(inputBatch, targetBatch_2d_input)
        with torch.backends.cudnn.flags(enabled=False):
            loss, ctc, att = loss_function(outputBatch_ctc, outputBatch_att, targetBatch_2d.transpose(0,1), inputLenBatch, targetLenBatch)
        loss.backward()
        optimizer.step()

        trainingLoss = trainingLoss + loss.detach().item()
        ctcLoss = ctcLoss + ctc.item()
        attLoss = attLoss + att.item()
        predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch_ctc.detach(), inputLenBatch, trainParams["eosIx"])
        # predictionBatch_att = att_greedy_decode(model, inputBatch, max_len=100, batch=inputLenBatch.size()[0], eosIx = trainParams["eosIx"])
        trainingCER = trainingCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        trainingWER = trainingWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, trainParams["spaceIx"])
        # trainingCER_att = trainingCER_att + compute_cer_att(predictionBatch_att, targetBatch, targetLenBatch, trainParams["eosIx"])
        # trainingWER_att = trainingWER_att + compute_wer_att(predictionBatch_att, targetBatch, targetLenBatch, trainParams["eosIx"], trainParams["spaceIx"])

    trainingLoss = trainingLoss/len(trainLoader)
    ctcLoss = ctcLoss/len(trainLoader) # no use
    attLoss = attLoss/len(trainLoader) # no use
    trainingCER = trainingCER/len(trainLoader)
    trainingWER = trainingWER/len(trainLoader)
    # trainingCER_att = trainingCER_att/len(trainLoader)
    # trainingWER_att = trainingWER_att/len(trainLoader)
    return trainingLoss, trainingCER, trainingWER



def evaluate(model, evalLoader, loss_function, device, evalParams):

    """
    Function to evaluate the model over validation/test set. It computes the loss, CER and WER over the evaluation set.
    The CTC decode scheme can be set to either 'greedy' or 'search'.
    """

    evalLoss = 0
    ctcLoss = 0
    attLoss = 0
    evalCER = 0
    evalWER = 0
    evalCER_att = 0
    evalWER_att = 0
    predictionBatchs = []
    predictionBatchs_att = []
    targetBatchs = []

    for batch, (inputBatch, targetBatch, inputLenBatch, targetLenBatch) in enumerate(tqdm(evalLoader, leave=False, desc="Eval",
                                                                                          ncols=75)):

        inputBatch, targetBatch, targetBatch_2d, targetBatch_2d_input = \
            ((inputBatch[0].float()).to(device), (inputBatch[1].float()).to(device)), (targetBatch[0].int()).to(device), (targetBatch[1].int()).to(device), (targetBatch[2].int()).to(device)
        inputLenBatch, targetLenBatch = (inputLenBatch.int()).to(device), (targetLenBatch.int()).to(device)

        opmode = np.random.choice(["AO", "VO", "AV"],
                                  p=[evalParams["aoProb"], evalParams["voProb"], 1-(evalParams["aoProb"]+evalParams["voProb"])])
        if opmode == "AO":
            inputBatch = (inputBatch[0], None)
        elif opmode == "VO":
            inputBatch = (None, inputBatch[1])
        else:
            pass

        model.eval()
        with torch.no_grad():
            outputBatch_ctc, outputBatch_att = model(inputBatch, targetBatch_2d_input)
            with torch.backends.cudnn.flags(enabled=False):
                loss, ctc, att = loss_function(outputBatch_ctc, outputBatch_att, targetBatch_2d.transpose(0,1), inputLenBatch, targetLenBatch)


        evalLoss = evalLoss + loss.item()
        ctcLoss = ctcLoss + ctc.item()
        attLoss = attLoss + att.item()
        if evalParams["decodeScheme"] == "greedy":
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch_ctc, inputLenBatch, evalParams["eosIx"])#(L'xB,)
            predictionBatch_att, predictionLenBatch_att = att_greedy_decode(model, inputBatch, max_len=100, eosIx = evalParams["eosIx"])#(L,B)
            
        elif evalParams["decodeScheme"] == "search":
            predictionBatch, predictionLenBatch = ctc_greedy_decode(outputBatch_ctc, inputLenBatch, evalParams["eosIx"])

            # predictionBatch, predictionLenBatch = ctc_search_decode(outputBatch_ctc, inputLenBatch, evalParams["beamSearchParams"],\
            #                                                         evalParams["spaceIx"], evalParams["eosIx"], evalParams["lm"],device=device)
            predictionBatch_att, predictionLenBatch_att = att_search_decode(model, inputBatch, max_len=100, eosIx = evalParams["eosIx"],\
                                                    beamSearchParams=evalParams["beamSearchParams"], lm=evalParams["lm"])



        else:
            print("Invalid Decode Scheme")
            exit()

        evalCER = evalCER + compute_cer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch)
        evalWER = evalWER + compute_wer(predictionBatch, targetBatch, predictionLenBatch, targetLenBatch, evalParams["spaceIx"])
        evalCER_att = evalCER_att + compute_cer(predictionBatch_att, targetBatch, predictionLenBatch_att, targetLenBatch)
        evalWER_att = evalWER_att + compute_wer(predictionBatch_att, targetBatch, predictionLenBatch_att, targetLenBatch, evalParams["spaceIx"])

        predictionBatchs.append(predictionBatch.to('cpu'))
        predictionBatchs_att.append(predictionBatch_att.to('cpu'))
        targetBatchs.append(targetBatch.to('cpu'))

    evalLoss = evalLoss/len(evalLoader)
    ctcLoss = ctcLoss/len(evalLoader) # no use
    attLoss = attLoss/len(evalLoader) # no use
    evalCER = evalCER/len(evalLoader)
    evalWER = evalWER/len(evalLoader)
    evalCER_att = evalCER_att/len(evalLoader)
    evalWER_att = evalWER_att/len(evalLoader)
    return evalLoss, evalCER, evalWER, evalCER_att, evalWER_att, predictionBatchs, predictionBatchs_att, targetBatchs