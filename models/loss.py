import torch
from torch import nn

class CustomLoss(nn.Module):
    def __init__(self, ramda):
        super().__init__()
        # パラメータを設定 
        self.ramda = ramda
        self.loss_ctc = nn.CTCLoss(zero_infinity=False, blank=0)
        self.loss_ce = nn.CrossEntropyLoss(ignore_index=0)

    def forward(self, output_ctc, output_ce, tgt, in_len, tgt_len):
        # ctc
        ctc = self.loss_ctc(output_ctc, tgt, in_len, tgt_len)

        # att
        t_size, b_size, _ = output_ce.size()
        tgt = tgt.to(torch.long)
        # input=output_ce.contiguous().view(b_size*t_size, -1)
        # target=tgt.T.contiguous().view(-1)
        att = self.loss_ce(output_ce.contiguous().view(b_size*t_size, -1), tgt.T.contiguous().view(-1))
        loss = self.ramda*att + (1-self.ramda)*ctc

        return loss, ctc, att