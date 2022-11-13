#AVSR row model
from base64 import encode
from binhex import openrsrc
from ntpath import join
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from conformer.encoder import ConformerEncoder
from conformer.encoder import ConformerBlock


def conv1d_3(in_planes, out_planes, stride=1):
    return nn.Conv1d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def downsample_basic_block( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(outplanes),
            )

def downsample_basic_block_v2( inplanes, outplanes, stride ):
    return  nn.Sequential(
                nn.AvgPool1d(kernel_size=stride, stride=stride, ceil_mode=True, count_include_pad=False),
                nn.Conv1d(inplanes, outplanes, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(outplanes),
            )



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, relu_type = 'relu' ):
        super(BasicBlock, self).__init__()

        assert relu_type in ['relu','prelu']

        self.conv1 = conv1d_3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm1d(planes)

        # type of ReLU is an input option
        if relu_type == 'relu':
            self.relu1 = nn.ReLU(inplace=True)
            self.relu2 = nn.ReLU(inplace=True)
        elif relu_type == 'prelu':
            self.relu1 = nn.PReLU(num_parameters=planes)
            self.relu2 = nn.PReLU(num_parameters=planes)
        else:
            raise Exception('relu type not implemented')
        # --------

        self.conv2 = conv1d_3(planes, planes)
        self.bn2 = nn.BatchNorm1d(planes)
        
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu2(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, relu_type = 'relu', gamma_zero = False, avg_pool_downsample = True):
        self.inplanes = 64
        self.relu_type = relu_type
        self.gamma_zero = gamma_zero
        self.downsample_block = downsample_basic_block_v2 if avg_pool_downsample else downsample_basic_block

        super(ResNet, self).__init__()
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AvgPool1d(kernel_size=20, stride=20)

        # default init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                #nn.init.ones_(m.weight)
                #nn.init.zeros_(m.bias)

        if self.gamma_zero:
            for m in self.modules():
                if isinstance(m, BasicBlock ):
                    m.bn2.weight.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):


        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = self.downsample_block( inplanes = self.inplanes, #ここ
                                                 outplanes = planes * block.expansion,#ここ 
                                                 stride = stride )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, relu_type = self.relu_type))#前半
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, relu_type = self.relu_type))#後半

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        # x = x.view(x.size(0), -1)
        return x

# -- auxiliary functions
def threeD_to_2D_tensor(x):
    n_batch, n_channels, s_time, sx, sy = x.shape
    x = x.transpose(1, 2)
    return x.reshape(n_batch*s_time, n_channels, sx, sy)


class AudioFrontend(nn.Module):
    def __init__(self):
        super(AudioFrontend, self).__init__()
        self.trunk = ResNet(BasicBlock, [2, 2, 2, 2], relu_type='prelu')
        self.frontend1D = nn.Conv1d(in_channels=1, padding=38, out_channels=64, kernel_size=80, stride=4)

    def forward(self, x):
        # (N,C,L)
        x = torch.unsqueeze(x, 1).transpose(0,2)
        x = self.frontend1D(x)
        x = self.trunk(x)
        # print('visual',x.shape)
        return x



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
        self.positionalEncoding = PositionalEncoding(dModel=dModel, maxLen=peMaxLen)
        self.embeddings = nn.Embedding(numClasses+1, dModel) # <EOS>は入力になりえないので<BOS>と置き換わるイメージ+zeropadding=40+1
        self.linear_BtoF = nn.Linear(512, dModel)
        self.linear_BtoF_A = nn.Linear(512, dModel)
        self.linear1 = nn.Linear(2*dModel, 4*dModel)
        self.linear2 = nn.Linear(4*dModel, dModel)
        self.batchnorm = nn.BatchNorm1d(4*dModel)
        self.relu = nn.ReLU()
        decoderLayer = nn.TransformerDecoderLayer(d_model=dModel, nhead=nHeads, dim_feedforward=fcHiddenSize, dropout=dropout)
        self.jointDecoder = nn.TransformerDecoder(decoderLayer, num_layers=6)
        self.outputLinear_CTC = nn.Linear(dModel, numClasses)
        self.outputLinear_att = nn.Linear(dModel, numClasses)
        self.videoEncoder = nn.ModuleList([ConformerBlock(
                                encoder_dim=dModel,
                            ) for _ in range(numLayers)])

        self.audioEncoder = nn.ModuleList([ConformerBlock(
                                encoder_dim=dModel,
                            ) for _ in range(numLayers)])

        self.resnet = AudioFrontend()
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


    # for att decode
    def encode(self, inputBatch):
        # audioinputBatch (T, B, inSize=80)
        # videoinputBatch (T, B, 512)
        audioInputBatch, videoInputBatch = inputBatch

        # audio encode
        if audioInputBatch is not None:
            # import numpy as np
            # np.savetxt('tmp.txt',audioInputBatch.cpu().numpy())
            # print(torch.isnan(audioInputBatch).any())
            # print(torch.isinf(audioInputBatch).any())
            audioInputBatch = self.resnet(audioInputBatch)
            # print(audioInputBatch)

            audioInputBatch = audioInputBatch.transpose(1, 2)#(B, T, inSize)
            audioInputBatch = self.linear_BtoF_A(audioInputBatch)
            # conformer fornt-end
            for layer in self.audioEncoder:
                audioInputBatch = layer(audioInputBatch)
            audioInputBatch = audioInputBatch.transpose(0, 1)# (T, B, encoder_dim)
        else:
            audioInputBatch = None
        # visual encode
        if videoInputBatch is not None:
            # back-end to front-end
            videoInputBatch = self.linear_BtoF(videoInputBatch)
            videoInputBatch = videoInputBatch.transpose(0, 1)#(B,T,inSize)
            # conformer fornt-end
            for layer in self.videoEncoder:
                videoInputBatch = layer(videoInputBatch)
            videoInputBatch = videoInputBatch.transpose(0, 1)#(T,B,inSize)
        else:
            videoInputBatch = None

        # joint
        if (audioInputBatch is not None) and (videoInputBatch is not None):
            # cat
            jointBatch = torch.cat([audioInputBatch, videoInputBatch], dim=2)
            del audioInputBatch, videoInputBatch
            # MLPs
            jointBatch = self.linear1(jointBatch)
            jointBatch = jointBatch.transpose(0,1).transpose(1,2)
            jointBatch = self.batchnorm(jointBatch)
            jointBatch = jointBatch.transpose(1,2).transpose(0,1)
            jointBatch = self.relu(jointBatch)
            jointBatch = self.linear2(jointBatch)

        return jointBatch

    # for att decode
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