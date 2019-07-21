import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim
from torch.autograd import Variable


dtype=torch.FloatTensor

# S: Symbol that shows starting of decoding input
# E: Symbol that shows starting of decoding output
# P: Symbol that will fill in blank sequence if current batch data size is short than time steps

##在常用的是<'EOS'>,<'UNK'>,<'BOS'>
char_arr = [c for c in 'SEPabcdefghijklmnopqrstuvwxyz']
num_dict = {n: i for i, n in enumerate(char_arr)}

seq_data=[['man','woman'], ['black', 'white'], ['king', 'queen'], ['girl', 'boy'], ['up', 'down'], ['high', 'low']]

n_steps=5
n_hidden=128
len_n=len(num_dict)
batch_size=len(seq_data)


def make_batch(seq_data):
    input_batch,output_batch,target_batch=[],[],[]

    for seq in seq_data:
        for i in range(2):
            seq[i]=seq[i]+'P'*(n_steps-len(seq[i]))
        
        input_=[num_dict[n] for n in seq[0]]
        output_=[num_dict[n] for n in ('S'+seq[1])]
        target_=[num_dict[n] for n in (seq[1]+'E')]

        input_batch.append(np.eye(len_n)[input_])
        output_batch.append(np.eye(len_n)[output_])
        target_batch.append(target_)
    
    input_batch=Variable(torch.Tensor(input_batch))
    output_batch=Variable(torch.Tensor(output_batch))
    target_batch=Variable(torch.LongTensor(target_batch))
    return input_batch,output_batch,target_batch
def translate(word):
    input_batch,output_batch,_=make_batch([[word,'P'*len(word)]])

    hidden=Variable(torch.zeros(1,1,n_hidden))

    output=model(input_batch,hidden,output_batch)

    pred=output.data.max(2,keepdim=True)[1]

    dec=[char_arr[i] for i in pred]

    end=dec.index('E')

    transe=''.join(dec[:end])

    return transe.replace('P','')
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.enc_cell=nn.RNN(input_size=len_n,hidden_size=n_hidden,dropout=0.5)
    def forward(self,enc_input,enc_hidden):
        model=self.enc_cell(enc_input,enc_hidden)
        return model
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.dec_cell=nn.RNN(input_size=len_n,hidden_size=n_hidden,dropout=0.5)
    def forward(self,dec_input,enc_states):
        model=self.dec_cell(dec_input,enc_states)
        return model
class Seq2Seq(nn.Module):
    def __init__(self,enc,dec):
        super(Seq2Seq,self).__init__()

        self.enc_cell=enc
        self.dec_cell=dec

        self.fc=nn.Linear(n_hidden,len_n)
    
    def forward(self,enc_input,enc_hidden,dec_input):
        enc_input=enc_input.transpose(0,1)
        dec_input=dec_input.transpose(0,1)
        _,enc_states=self.enc_cell(enc_input,enc_hidden)

        outputs,_=self.dec_cell(dec_input,enc_states)

        model=self.fc(outputs)

        return model

input_batch,output_batch,target_batch= make_batch(seq_data)
enc=Encoder()
dec=Decoder()
model=Seq2Seq(enc,dec)
#model=Seq2Seq().to('cuda')

criter=nn.CrossEntropyLoss()

optimer=optim.Adam(model.parameters(),lr=0.001)


for ep in range(500):
    hidden=Variable(torch.zeros(1,batch_size,n_hidden))

    optimer.zero_grad()

    output=model(input_batch,hidden,output_batch)
    
    output=output.transpose(0,1)
    loss=0  

    for i in range(0,len(target_batch)):
        loss+=criter(output[i],target_batch[i])

    if(ep+1)%100==0:
        print('ep:','%04d'%(ep+1),'cost=','{:.6f}'.format(loss))
    
    loss.backward()
    optimer.step()


print('test')
print('man ->', translate('man'))
print('mans ->', translate('mans'))
print('king ->', translate('king'))
print('black ->', translate('black'))
print('upp ->', translate('upp'))