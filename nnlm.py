##A Neural Probabilistic Language Model
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype=torch.FloatTensor

sentences=["i like dog", "i love coffee", "i hate milk"]
word_list=" ".join(sentences).split()
word_list=list(set(word_list))

word_dict={w: i for i,w in enumerate(word_list)}
number_dict={i:w for i,w in enumerate(word_list)}
len_n=len(word_list)

n_step=2
n_hidden=2
m=2

def make_batch(sentences):
    input_batch=[]
    output_batch=[]

    for sentence in sentences:
        word=sentence.split()
        inputs=[word_dict[n] for n in word[:-1]]
        target=word_dict[word[-1]]

        input_batch.append(inputs)
        output_batch.append(target)
    
    return input_batch,output_batch

class NNLM(nn.Module):
    def __init__(self):
        super(NNLM,self).__init__()
        self.C=nn.Embedding(len_n,m)
        self.H=nn.Parameter(torch.randn(n_step*m,n_hidden).type(dtype))
        self.W=nn.Parameter(torch.randn(n_step*m,len_n).type(dtype))
        self.d=nn.Parameter(torch.randn(n_hidden).type(dtype))
        self.U=nn.Parameter(torch.randn(n_hidden,len_n).type(dtype))
        self.b=nn.Parameter(torch.randn(len_n).type(dtype))
    
    def forward(self,X):
        X=self.C(X)
        # concatenation of the input word features from the matrix C
        X=X.view(-1,n_step*m)
        tanh=torch.tanh(self.d+torch.mm(X,self.H))
        print("get tanh",torch.mm(X,self.H))

        #y = b+Wx+U tanh(d +Hx)
        output=self.b+torch.mm(X,self.W)+torch.mm(tanh,self.U)
        print("get mm",torch.mm(X,self.W),torch.mm(tanh,self.U))
        print("output",X,tanh,self.b,self.W,self.U,output)
        return output
model=NNLM()

criter=nn.CrossEntropyLoss()
optimer=optim.Adam(model.parameters(),lr=0.001)

input_batch,output_batch=make_batch(sentences)
#将句子处理成torch类型数值
input_batch=Variable(torch.LongTensor(input_batch))
output_batch=Variable(torch.LongTensor(output_batch))


for ep in range(50):
    optimer.zero_grad()
    output=model(input_batch)
    loss=criter(output,output_batch)
    if(ep+1)%10==0:
        print('Ep:','%04d'%(ep+1),'cost=','{:.6f}'.format(loss))
    loss.backward()
    optimer.step()

pred=model(input_batch).data.max(1,keepdim=True)[1]

print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in pred.squeeze()])