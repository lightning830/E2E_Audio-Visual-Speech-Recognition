

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from itertools import groupby


np.seterr(divide="ignore")


def get_subsequent_mask(sz, device):
    mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

class Translator(nn.Module):
    ''' Load a trained model and translate in beam search fashion. '''
    # decodeでbeamwidthをbatchとして推論
    def __init__(
            self, model, beam_size, max_seq_len,
            trg_pad_idx, trg_bos_idx, trg_eos_idx, lm=None):
        

        super(Translator, self).__init__()

        self.alpha = 0.7
        self.beam_size = beam_size
        self.max_seq_len = max_seq_len
        self.trg_bos_idx = trg_bos_idx
        self.trg_eos_idx = trg_eos_idx
        self.lm = lm
        self.statebatch_h = None
        self.statebatch_c = None

        self.model = model
        self.model.eval()

        self.register_buffer('init_seq', torch.LongTensor([[trg_bos_idx]]).to(model.device))
        self.register_buffer(
            'blank_seqs', 
            torch.full((beam_size, max_seq_len), trg_pad_idx, dtype=torch.long).to(model.device))
        self.blank_seqs[:, 0] = self.trg_bos_idx
        self.register_buffer(
            'len_map', 
            torch.arange(1, max_seq_len + 1, dtype=torch.long).unsqueeze(0).to(model.device))


    def _model_decode(self, trg_seq, enc_output):
        trg_mask = get_subsequent_mask(trg_seq.size(0),trg_seq.device).type(torch.bool)
        _, dec_output = self.model.decode(targetBatch=trg_seq, memory=enc_output, tgt_mask=trg_mask)
        return F.softmax(dec_output, dim=-1)


    def _get_init_state(self, src_seq):
        beam_size = self.beam_size

        enc_output = self.model.encode(src_seq)
        dec_output = self._model_decode(self.init_seq, enc_output)
        
        best_k_probs, best_k_idx = dec_output[-1, :, :].topk(beam_size)

        scores = torch.log(best_k_probs).view(beam_size)
        gen_seq = self.blank_seqs.clone().detach()
        gen_seq[:, 1] = best_k_idx[0]#[[40,20,0,,,],[40,13,0,,,],,,]
        enc_output = enc_output.repeat(1, beam_size, 1)
        return enc_output, gen_seq, scores


    def _get_the_best_score_and_idx(self, gen_seq, dec_output, scores, step):
        assert len(scores.size()) == 1
        
        beam_size = self.beam_size
        # if step == 6:
        #     print(gen_seq[:,step-1])
        if self.lm is not None:
            i_list = []#0に置き換えた番号
            input = gen_seq[:, step-1]-1
            for i in range(input.shape[0]):
                if input[i] > 37 or input[i] < 0:
                    input[i] = 0 #39のものは0に置き換え
                    i_list.append(i)
            if step == 2:
                lm_batch, self.statebatch = self.lm(input.unsqueeze(0), None)
            else:
                initstate_h, initstate_c = self.statebatch_h, self.statebatch_c
                lm_batch, self.statebatch = self.lm(input.unsqueeze(0), (initstate_h, initstate_c))#全部-1ずれているのに注意
            lm_batch = lm_batch.squeeze()
            lm_batch[i_list] = 0 #39の次の確率は全てlogxに合わせて小さくする
            minis = (torch.ones(beam_size,1)*(0)).to(self.model.device)#0と39の確率も小さくする
            lm_batch = torch.cat([minis, lm_batch, minis], dim=1)#lmにないから，paddinの0と，eosの39を追加
            
        # Get k candidates for each beam, k^2 candidates in total.
        best_k2_probs, best_k2_idx = dec_output[-1, :, :].topk(beam_size)

        # Include the previous scores.1つ前の確率＋次の確率（beam_size個）を足す
        if self.lm is not None:# best_k2_idxの文字の確率だけ抽出（38->beam_size）
            c=None
            for i in range(lm_batch.shape[0]):
                if c==None:
                    c=lm_batch[i,best_k2_idx[i]]
                else:
                    x = lm_batch[i,best_k2_idx[i]]
                    c=torch.cat([c,x], dim=0)
            c = c.view(beam_size,-1)
            scores = (torch.log(best_k2_probs)+0.6*c).view(beam_size, -1) + scores.view(beam_size, 1)

        else:
            scores = torch.log(best_k2_probs).view(beam_size, -1) + scores.view(beam_size, 1)

        # Get the best k candidates from k^2 candidates. その全部で上位beam_size個を選別
        scores, best_k_idx_in_k2 = scores.view(-1).topk(beam_size)
 
        # Get the corresponding positions of the best k candidiates.　
        best_k_r_idxs, best_k_c_idxs = best_k_idx_in_k2 // beam_size, best_k_idx_in_k2 % beam_size
        #best_k2_idx形状(beam,beam)の(best_k_r_idxs,bestk_c_idxs)（それぞれ形状(beam,)）の要素をかき集める
        best_k_idx = best_k2_idx[best_k_r_idxs, best_k_c_idxs]#(beam,)

        # Copy the corresponding previous tokens.
        gen_seq[:, :step] = gen_seq[best_k_r_idxs, :step]
        # Set the best tokens in this beam search step
        gen_seq[:, step] = best_k_idx

        if self.lm is not None:
            self.statebatch_h = self.statebatch[0][:,best_k_r_idxs,:]
            self.statebatch_c = self.statebatch[1][:,best_k_r_idxs,:]

        return gen_seq, scores


    def translate_sentence(self, src_seq):
        # Only accept batch size equals to 1 in this function.
        # TODO: expand to batch operation.
        # assert src_seq.size(0) == 1

        trg_eos_idx = self.trg_eos_idx 
        max_seq_len, beam_size, alpha = self.max_seq_len, self.beam_size, self.alpha 

        with torch.no_grad():
            enc_output, gen_seq, scores = self._get_init_state(src_seq)

            ans_idx = 0   # default
            for step in range(2, max_seq_len):    # decode up to max length
                dec_output = self._model_decode(gen_seq[:, :step].T, enc_output)#0埋めしているからstepまでにしてるだけ
                gen_seq, scores = self._get_the_best_score_and_idx(gen_seq, dec_output, scores, step)

                # Check if all path finished
                # -- locate the eos in the generated sequences
                eos_locs = gen_seq == trg_eos_idx   
                # -- replace the eos with its position for the length penalty use
                seq_lens, _ = self.len_map.masked_fill(~eos_locs, max_seq_len).min(1)
                # -- check if all beams contain eos
                if (eos_locs.sum(1) > 0).sum(0).item() == beam_size:#EOSに達しているものの和がbeamsize個かどうか
                    # TODO: Try different terminate conditions.
                    _, ans_idx = scores.div(seq_lens.float() ** alpha).max(0)
                    ans_idx = ans_idx.item()
                    break
        '''
        tmp = gen_seq.tolist()
        from config import args
        for i in tmp:
            for j in i:
                if j == 0:
                    break
                print(args["INDEX_TO_CHAR"][j], end="")
            print()
        '''
        return gen_seq[ans_idx][:seq_lens[ans_idx]].tolist()
        # return gen_seq, seq_lens


def greedy_translator(model, input, max_len, sosIx, eosIx, blank=0):
    # input=(4T,1,C),(T,1,C)
    DEVICE = model.device
    memory = model.encode(input)
    ys = torch.ones(1,1).fill_(sosIx).type(torch.long).to(DEVICE) #fill tensor of shape(1,1) with sosIx. (L,B)

    for i in range(max_len-1):
        tgt_mask = (model.generate_square_subsequent_mask(ys.size(0), DEVICE).type(torch.bool))
        _, out = model.decode(ys, memory, tgt_mask)#(T,B=1,C)
        out = out.transpose(0,1)#(B,T,C)
        prob = out[:,-1,:]#(B,C)
        _, next_word = torch.max(prob, dim=1)#(B,1) Choose the largest of the C pieces.
        next_word = next_word.item()
        ys = torch.cat([ys, torch.ones(1, 1).type(torch.long).fill_(next_word).to(DEVICE)], dim=0)
        if next_word == eosIx:
            break
    ys = ys.squeeze(dim=1)
    ys = ys.cpu()
    return ys #(max_len,)

'''
def att_greedy_decode(model, src, max_len, sosIx=40, eosIx=39, blank=0):
    a, v = src
    a, v = a.transpose(0,1), v.transpose(0,1)#(B,4T,C),(B,T,C)
    predicts = None
    predict_lens = None
    for (_a, _v) in zip(a, v):
        _a, _v = _a.unsqueeze(1), _v.unsqueeze(1)
        input = (_a, _v)
        predict = greedy_translator(model, input, max_len, sosIx, eosIx, blank=0)
        predict = predict[1:] #bos delete
        predict_len = len(predict)
        predict_len = torch.IntTensor([predict_len])
        if predicts is None:
            predicts = predict
            predict_lens = predict_len
        else:
            predicts = torch.cat([predicts, predict], dim = 0)
            predict_lens = torch.cat([predict_lens, predict_len], dim = 0)
    predicts = predicts.cpu(); predict_lens = predict_lens.cpu()
    return predicts, predict_lens


'''
def att_greedy_decode(model, src, max_len, sosIx=40, eosIx=39, blank=0, device='cuda'):
    # to device
    DEVICE = torch.device(device)
    # src = src.to(DEVICE) 

    # encode
    batch = src[0].shape[1]
    memory = model.encode(src) #(T, B, C=dModel)
    ys = torch.ones(1,batch).fill_(sosIx).type(torch.long).to(DEVICE) #fill tensor of shape(1,B) with sosIx

    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (model.generate_square_subsequent_mask(ys.size(0), DEVICE).type(torch.bool))
        _, out = model.decode(ys, memory, tgt_mask)#(T,B,C)
        out = out.transpose(0,1)#(B,T,C)
        prob = out[:,-1,:]#(B,C)
        _, next_word = torch.max(prob, dim=1)#(B,1) Choose the largest of the C pieces.
        ys = torch.cat([ys, next_word.unsqueeze(0)], dim=0)
        continueflag = False
        for j in next_word:
            if torch.equal(j, torch.tensor(eosIx).to(DEVICE)) == False:#any of them is not eosIx
                continueflag = True
        if continueflag == False:
            break
    ys = ys.transpose(0,1)[:,1:]#(B,L) and bos remove
    predicts = None
    predict_Len = None
    for b in ys:
        len_c = 0
        for i in range(b.shape[0]):
            #追加する                
            len_c += 1
            if predicts is None:#初め
                predicts = b[i].unsqueeze(0)
            else:
                predicts = torch.cat([predicts, b[i].unsqueeze(0)], dim=0)#predicts追加

            #eosがきたら，それかmaxlenで最後にする
            if b[i] == eosIx or i == b.shape[0]-1:
                len_c = torch.IntTensor([len_c])
                if predict_Len is None:
                    predict_Len = len_c
                else:
                    predict_Len = torch.cat([predict_Len, len_c], dim=0)
                break

    predicts = predicts.cpu(); predict_Len = predict_Len.cpu()
    return predicts, predict_Len


def att_search_decode(model, src, max_len, beamSearchParams, lm, sosIx=40, eosIx=39):
    beamWidth = beamSearchParams["beamWidth"]
    alpha = beamSearchParams["alpha"]
    # beta = beamSearchParams["beta"]

    transrator = Translator(model, beam_size=beamWidth, max_seq_len=max_len, trg_pad_idx=0, trg_bos_idx=sosIx, trg_eos_idx=eosIx, lm=lm)

    a, v = src
    a, v = a.transpose(0,1), v.transpose(0,1)#(B,4T,C),(B,T,C)
    predicts = None
    predict_lens = None
    for (_a, _v) in zip(a, v):
        _a, _v = _a.unsqueeze(1), _v.unsqueeze(1)
        input = (_a, _v)
        predict = transrator.translate_sentence(input)
        predict = predict[1:] #bos delete
        predict_len = len(predict)
        predict = torch.IntTensor(predict); predict_len = torch.IntTensor([predict_len])
        if predicts is None:
            predicts = predict
            predict_lens = predict_len
        else:
            predicts = torch.cat([predicts, predict], dim = 0)
            predict_lens = torch.cat([predict_lens, predict_len], dim = 0)
    predicts = predicts.cpu(); predict_lens = predict_lens.cpu()
    return predicts, predict_lens



def ctc_greedy_decode(outputBatch, inputLenBatch, eosIx, blank=0):

    """
    Greedy search technique for CTC decoding.
    This decoding method selects the most probable character at each time step. This is followed by the usual CTC decoding
    to get the predicted transcription.
    Note: The probability assigned to <EOS> token is added to the probability of the blank token before decoding
    to avoid <EOS> predictions in middle of transcriptions. Once decoded, <EOS> token is appended at last to the
    predictions for uniformity with targets.
    """

    outputBatch = outputBatch.cpu()
    inputLenBatch = inputLenBatch.cpu()
    outputBatch[:,:,blank] = torch.log(torch.exp(outputBatch[:,:,blank]) + torch.exp(outputBatch[:,:,eosIx]))
    reqIxs = np.arange(outputBatch.shape[2])
    reqIxs = reqIxs[reqIxs != eosIx]
    outputBatch = outputBatch[:,:,reqIxs]

    predCharIxs = torch.argmax(outputBatch, dim=2).T.numpy()
    inpLens = inputLenBatch.numpy()
    preds = list()
    predLens = list()
    for i in range(len(predCharIxs)):
        pred = predCharIxs[i]
        ilen = inpLens[i]
        pred = pred[:ilen]
        pred = np.array([x[0] for x in groupby(pred)])
        pred = pred[pred != blank]
        pred = list(pred)
        pred.append(eosIx)
        preds.extend(pred)
        predLens.append(len(pred))
    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch



class BeamEntry:
    """
    Class for a single entry in the beam.
    """
    def __init__(self):
        self.logPrTotal = -np.inf
        self.logPrNonBlank = -np.inf
        self.logPrBlank = -np.inf
        self.logPrText = 0
        self.lmApplied = False
        self.lmState = None
        self.labeling = tuple()



class BeamState:

    """
    Class for the beam.
    """

    def __init__(self, alpha, beta):
        self.entries = dict()
        self.alpha = alpha
        self.beta = beta


    def score(self, entry):
        """
        Function to compute score of each entry in the beam.
        """
        labelingLen = len(entry.labeling)
        if labelingLen == 0:
            score = entry.logPrTotal + self.alpha*entry.logPrText
        else:
            score = (entry.logPrTotal + self.alpha*entry.logPrText)/(labelingLen**self.beta)
        return score


    def sort(self):
        """
        Function to sort all the beam entries in descending order depending on their scores.
        """
        beams = [entry for (key, entry) in self.entries.items()]
        sortedBeams = sorted(beams, reverse=True, key=self.score)
        return [x.labeling for x in sortedBeams]



def apply_lm(parentBeam, childBeam, spaceIx, lm, device):

    """
    Applying the language model to obtain the language model character probabilities at a time step
    given all the previous characters.
    """

    if not (childBeam.lmApplied):
        if parentBeam.lmState == None:
            initStateBatch = None
            inputBatch = torch.tensor(spaceIx-1).reshape(1,1)
            inputBatch = inputBatch.to(device)
        else:
            initStateBatch = parentBeam.lmState
            inputBatch = torch.tensor(parentBeam.labeling[-1]-1).reshape(1,1) #1つ，ずれている（38のものは37扱い）
            inputBatch = inputBatch.to(device)
        lm.eval()
        with torch.no_grad():
            outputBatch, finalStateBatch = lm(inputBatch, initStateBatch)
        logProbs = outputBatch.squeeze()
        logProb = logProbs[childBeam.labeling[-1]-1]
        childBeam.logPrText = parentBeam.logPrText + logProb
        childBeam.lmApplied = True
        childBeam.lmState = finalStateBatch
    return



def add_beam(beamState, labeling):
    """
    Function to add a new entry to the beam.
    """
    if labeling not in beamState.entries.keys():
        beamState.entries[labeling] = BeamEntry()



def log_add(a, b):
    """
    Addition of log probabilities.
    """
    result = np.log(np.exp(a) + np.exp(b))
    return result



def ctc_search_decode(outputBatch, inputLenBatch, beamSearchParams, spaceIx, eosIx, lm, device, blank=0):

    """
    Applies the CTC beam search decoding along with a character-level language model.
    Note: The probability assigned to <EOS> token is added to the probability of the blank token before decoding
    to avoid <EOS> predictions in middle of transcriptions. Once decoded, <EOS> token is appended at last to the
    predictions for uniformity with targets.
    """

    outputBatch = outputBatch.cpu()
    inputLenBatch = inputLenBatch.cpu()
    outputBatch[:,:,blank] = torch.log(torch.exp(outputBatch[:,:,blank]) + torch.exp(outputBatch[:,:,eosIx]))
    reqIxs = np.arange(outputBatch.shape[2])
    reqIxs = reqIxs[reqIxs != eosIx]
    outputBatch = outputBatch[:,:,reqIxs]

    beamWidth = beamSearchParams["beamWidth"]
    alpha = beamSearchParams["alpha"]
    beta = beamSearchParams["beta"]
    threshProb = beamSearchParams["threshProb"]

    outLogProbs = outputBatch.transpose(0, 1).numpy()
    inpLens = inputLenBatch.numpy()
    preds = list()
    predLens = list()

    for n in range(len(outLogProbs)):
        mat = outLogProbs[n]
        ilen = inpLens[n]
        mat = mat[:ilen,:]
        maxT, maxC = mat.shape

        #initializing the main beam with a single entry having empty prediction
        last = BeamState(alpha, beta)
        labeling = tuple()
        last.entries[labeling] = BeamEntry()
        last.entries[labeling].logPrBlank = 0
        last.entries[labeling].logPrTotal = 0

        #going over all the time steps
        for t in range(maxT):

            #a temporary beam to store all possible predictions (which are extensions of predictions
            #in the main beam after time step t-1) after time step t
            curr = BeamState(alpha, beta)
            #considering only the characters with probability above a certain threshold to speeden up the algo
            prunedChars = np.where(mat[t,:] > np.log(threshProb))[0]

            #keeping only the best predictions in the main beam
            bestLabelings = last.sort()[:beamWidth]

            #going over all the best predictions
            for labeling in bestLabelings:

                #same prediction (either blank or last character repeated)
                if len(labeling) != 0:
                    logPrNonBlank = last.entries[labeling].logPrNonBlank + mat[t, labeling[-1]]
                else:
                    logPrNonBlank = -np.inf

                logPrBlank = last.entries[labeling].logPrTotal + mat[t, blank]

                add_beam(curr, labeling)
                curr.entries[labeling].labeling = labeling
                curr.entries[labeling].logPrNonBlank = log_add(curr.entries[labeling].logPrNonBlank, logPrNonBlank)
                curr.entries[labeling].logPrBlank = log_add(curr.entries[labeling].logPrBlank, logPrBlank)
                curr.entries[labeling].logPrTotal = log_add(curr.entries[labeling].logPrTotal, log_add(logPrBlank, logPrNonBlank))
                curr.entries[labeling].logPrText = last.entries[labeling].logPrText
                curr.entries[labeling].lmApplied = True
                curr.entries[labeling].lmState = last.entries[labeling].lmState


                #extending the best prediction with all characters in the pruned set
                for c in prunedChars:

                    if c == blank:
                        continue

                    #extended prediction
                    newLabeling = labeling + (c,)

                    if (len(labeling) != 0)  and (labeling[-1] == c):
                        logPrNonBlank = mat[t, c] + last.entries[labeling].logPrBlank
                    else:
                        logPrNonBlank = mat[t, c] + last.entries[labeling].logPrTotal

                    add_beam(curr, newLabeling)
                    curr.entries[newLabeling].labeling = newLabeling
                    curr.entries[newLabeling].logPrNonBlank = log_add(curr.entries[newLabeling].logPrNonBlank, logPrNonBlank)
                    curr.entries[newLabeling].logPrTotal = log_add(curr.entries[newLabeling].logPrTotal, logPrNonBlank)

                    #applying language model
                    if lm is not None:
                        apply_lm(curr.entries[labeling], curr.entries[newLabeling], spaceIx, lm, device)

            #replacing the main beam with the temporary beam having extended predictions
            last = curr

        #output the best prediciton
        bestLabeling = last.sort()[0]
        bestLabeling = list(bestLabeling)
        bestLabeling.append(eosIx)
        preds.extend(bestLabeling)
        predLens.append(len(bestLabeling))

    predictionBatch = torch.tensor(preds).int()
    predictionLenBatch = torch.tensor(predLens).int()
    return predictionBatch, predictionLenBatch
