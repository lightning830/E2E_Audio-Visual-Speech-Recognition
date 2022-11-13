
# joint dim=256
from ntpath import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from conformer.encoder import ConformerEncoder
from conformer.encoder import ConformerBlock
from conformer.convolution import (
    ConformerConvModule,
    Conv2dSubampling,
)



class PositionalEncoding(nn.Module):

    """
    A layer to add positional encodings to the inputs of a Transformer model.
    Formula:
    PE(pos,2i) = sin(pos/10000^(2i/d_model))
    PE(pos,2i+1) = cos(pos/10000^(2i/d_model))
    """

    def __init__(self, dModel, maxLen):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(maxLen, dModel)
        position = torch.arange(0, maxLen, dtype=torch.float).unsqueeze(dim=-1)
        denominator = torch.exp(torch.arange(0, dModel, 2).float()*(math.log(10000.0)/dModel))
        pe[:, 0::2] = torch.sin(position/denominator)
        pe[:, 1::2] = torch.cos(position/denominator)
        pe = pe.unsqueeze(dim=0).transpose(0, 1)
        self.register_buffer("pe", pe)


    def forward(self, inputBatch):
        outputBatch = inputBatch + self.pe[:inputBatch.shape[0],:,:]
        return outputBatch



class AVNet(nn.Module):

    """
    An audio-visual speech transcription model based on the Transformer architecture.
    Architecture: Two stacks of 6 Transformer encoder layers form the Encoder (one for each modality),
                  A single stack of 6 Transformer encoder layers form the joint Decoder. The encoded feature vectors
                  from both the modalities are concatenated and linearly transformed into 512-dim vectors.
    Character Set: 26 alphabets (A-Z), 10 numbers (0-9), apostrophe ('), space ( ), blank (-), end-of-sequence (<EOS>)
    Audio Input: 321-dim STFT feature vectors with 100 vectors per second. Each group of 4 consecutive feature vectors
                 is linearly transformed into a single 512-dim feature vector giving 25 vectors per second.
    Video Input: 512-dim feature vector corresponding to each video frame giving 25 vectors per second.
    Output: Log probabilities over the character set at each time step.
    """

    def __init__(self, dModel, nHeads, numLayers, peMaxLen, inSize, fcHiddenSize, dropout, numClasses, device: torch.device = 'cuda'):
        super(AVNet, self).__init__()
        self.device = device
        # self.audioEncoder = ConformerEncoder(input_dim=inSize, encoder_dim=dModel, num_layers=numLayers)
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        self.embeddings = nn.Embedding(numClasses+1, dModel) # <EOS>は入力になりえないので<BOS>と置き換わるイメージ+zeropadding=40+1
        self.linear_BtoF = nn.Linear(512, dModel)
        self.linear_joint = nn.Linear(2*dModel, dModel)
        self.linear1 = nn.Linear(dModel, 4*dModel)
        self.linear2 = nn.Linear(4*dModel, dModel)
        self.batchnorm = nn.BatchNorm1d(4*dModel)
        self.relu = nn.ReLU()
        decoderLayer = nn.TransformerDecoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.jointDecoder = nn.TransformerDecoder(decoderLayer, num_layers=6)
        self.outputLinear_CTC = nn.Linear(dModel, numClasses)
        self.outputLinear_att = nn.Linear(dModel, numClasses)
        self.jointEncoder = nn.ModuleList([ConformerBlock(
                                encoder_dim=dModel,
                            ) for _ in range(numLayers)])
            
        # conv_subsampling
        self.conv_subsample = Conv2dSubampling(in_channels=1, out_channels=dModel)
        self.input_projection = nn.Sequential(
            nn.Linear(dModel * (((inSize - 1) // 2 - 1) // 2), dModel),
            nn.Dropout(p=0.1),
        )
        return


        # sz*szのサイズでマスクを-inf、他0のtensor型を返す
    def generate_square_subsequent_mask(self, sz, device):
        mask = (torch.triu(torch.ones((sz, sz), device=device)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


    def create_mask(self, tgt, device):
        tgt_seq_len = tgt.shape[0]

        tgt_mask = self.generate_square_subsequent_mask(tgt_seq_len, device)#tgt_seq_len*tgt_seq_lenの自身のmask作成(0と-inf)

        tgt_padding_mask = (tgt == 0).transpose(0, 1)#pad_idx=0の場所をtrueにする
        return tgt_mask, tgt_padding_mask #(S_max, S_max), (S_max)

    # decode
    def encode(self, inputBatch):
        # audioinputBatch (T, B, inSize=80)
        # targetBatch (Lmax, B)
        # videoinputBatch (T, B, 512)
        audioInputBatch, videoInputBatch = inputBatch

        # audio encode
        if audioInputBatch is not None:
            audioInputBatch = audioInputBatch.transpose(0, 1)#(B, T, inSize)
            audioBatch, _ = self.conv_subsample(audioInputBatch, audioInputBatch.size(1))
            audioBatch = self.input_projection(audioBatch)
            audioBatch = audioBatch.transpose(0, 1)# (T, B, encoder_dim)
        else:
            audioBatch = None

        # visual encode
        if videoInputBatch is not None:
            # back-end to front-end
            videoBatch = self.linear_BtoF(videoInputBatch)
        else:
            videoBatch = None

        # joint
        if (audioBatch is not None) and (videoBatch is not None):
            # cat
            jointBatch = torch.cat([audioBatch, videoBatch], dim=2)
            jointBatch = self.linear_joint(jointBatch)
            jointBatch = jointBatch.transpose(0, 1)#(B,T,inSize)
            # conformer fornt-end
            for layer in self.jointEncoder:
                jointBatch = layer(jointBatch)
            jointBatch = jointBatch.transpose(0, 1)#(T,B,inSize)
            # MLPs
            jointBatch = self.linear1(jointBatch)
            jointBatch = jointBatch.transpose(0,1).transpose(1,2)
            jointBatch = self.batchnorm(jointBatch)
            jointBatch = jointBatch.transpose(1,2).transpose(0,1)
            jointBatch = self.relu(jointBatch)
            jointBatch = self.linear2(jointBatch)

        elif (audioBatch is None) and (videoBatch is not None):
            jointBatch = videoBatch
        elif (audioBatch is not None) and (videoBatch is None):
            jointBatch = audioBatch
        else:
            print("Both audio and visual inputs missing.")
            exit()
        
        return jointBatch

    # decode
    def decode(self, targetBatch, memory, tgt_mask=None, tgt_key_padding_mask=None):
        # targetBatch (Lmax, B)

        # embedding
        tgt_emb = self.embeddings(targetBatch) #(L, B, dModel)
        tgt_emb = self.positionalEncoding(tgt_emb)

        # attModule
        output_att = self.jointDecoder(tgt=tgt_emb,memory=memory,tgt_mask=tgt_mask,tgt_key_padding_mask=tgt_key_padding_mask) #(T, B, 2*encoder_dim)
        output_att = self.outputLinear_att(output_att)

        # ctcModule
        output_ctc = self.outputLinear_CTC(memory)
        output_ctc = F.log_softmax(output_ctc, dim=2)

        return output_ctc, output_att


    def forward(self, inputBatch, targetBatch):
        # audioinputBatch (T, B, inSize=80)
        # targetBatch (Lmax, B)
        # videoinputBatch (T, B, 512)

        # create mask
        tgt_mask, tgt_key_padding_mask = self.create_mask(targetBatch, self.device)
        encode_output = self.encode(inputBatch)
        output_ctc, output_att = self.decode(targetBatch=targetBatch, memory=encode_output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask)

        return output_ctc, output_att