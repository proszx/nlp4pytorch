##defination
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import pdb
#Global variable
PAD=0
UNK=100
BOS=101
EOS=102
MASK=103

PAD_WORD='[PAD]'
UNK_WORD='[UNK]'
BOS_WORD='[CLS]'
EOS_WORD='[SEP]'
#https://github.com/kururuken/BERT-Transformer-for-Summarization/blob/master/transformer/Translator.py
#beam
class Beam():
    def __init__(self, size, device=False):
        self.size=size
        self._done=False

        self.scores=torch.zeros((size,),dtype=torch.float,device=device)

        self.all_score=[]

        self.prev_ks=[]

        self.next_ys=[torch.full((size,),BOS,dtype=torch.long,device=device)]
    def get_curr_state(self):
        return self.get_tentative_hy()
    def get_curr_ori(self):
        return self.prev_ks[-1]
    
    @property
    def done(self):
        return self._done

    def advance(self,word_prob):
        num_words=word_prob.size(1)
        return True
    def sort_scores(self):
        pass
    def get_the_bst_score_n_idx(self):
        pass
    def get_tentative_hy(self):
        pass
    def get_hy(self,k):
        pass
#module
class ScaledDotAttention(nn.Module):
    def __init__(self):
        return super().__init__()
    def forward(self, *input):
        return super().forward(*input)
#sublayer
class MultiHeadAttention(nn.Module):
    def __init__(self):
        return super().__init__()
    def forward(self, *input):
        return super().forward(*input)
class PositionWsFed4word(nn.Module):
    def __init__(self):
        return super().__init__()
    def forward(self, *input):
        return super().forward(*input)
#optim
class ScheduledOptim():
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    def step_and_update_lr(self):
        pass
    def zero_grad(self):
        pass
    def get_lr_scale(self):
        pass
    def update_lr(self):
        pass
#layer
class EncoderLayer(nn.Module):
    def __init__(self):
        return super().__init__()

    def forward(self, *input):
        return super().forward(*input)
class DecoderLayer(nn.Module):
    def __init__(self):
        return super().__init__()
    def forward(self, *input):
        return super().forward(*input)
#model

def get_non_pad(seq):
    assert seq.dim()==2
    return seq.ne(PAD).type(torch.float).unsqueeze(-1)
def get_sinusoid_encoding_table():
    def cal_angle(position,hid_idx):
        pass
    def get_posi_angle(position):
        pass
    pass
def get_attn_key_pad_mask(seq_k,seq_q):
    pass
def get_subsequent_mask(seq):
    pass
class Encoder(nn.Module):
    def __init__(self):
        return super().__init__()
    def forward(self):
        pass
class Decoder(nn.Module):
    def __init__(self):
        return super().__init__()
    def forward(self, *input):
        return super().forward(*input)
class Transformer():
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    def forward(self):
        pass
    
#tranlater


class Translator(object):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
    def forward(self):
        pass
    

