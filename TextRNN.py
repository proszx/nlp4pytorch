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
len_n=len(word_dict)

n_step=2
n_hidden=5
batch_size=len(sentences)

def make_batch(sentences):
    input_batch=[]
    output_batch=[]

    for sentence in sentences:
        word=sentence.split()
        inputs=[word_dict[n] for n in word[:-1]] ##input must have 3 dimensions, got 2 bug fix
        target=word_dict[word[-1]]

        input_batch.append(np.eye(len_n)[inputs])
        output_batch.append(target)
    
    return input_batch,output_batch
input_batch,output_batch=make_batch(sentences)
input_batch=Variable(torch.Tensor(input_batch))
output_batch=Variable(torch.LongTensor(output_batch))

class TextRNN(nn.Module):
    def __init__(self):
        super(TextRNN,self).__init__()

        self.rnn=nn.RNN(input_size=len_n,hidden_size=n_hidden)
        self.W=nn.Parameter(torch.randn([n_hidden,len_n]).type(dtype))
        self.b=nn.Parameter(torch.randn([len_n]).type(dtype))

    def forward(self,hidden,X):
        X=X.transpose(0,1)
        outputs,hidden=self.rnn(X,hidden)
        outputs=outputs[-1]
        model=torch.mm(outputs,self.W)+self.b

        return model
    
model=TextRNN()

criter=nn.CrossEntropyLoss()
optimer=optim.Adam(model.parameters(),lr=0.001)

for ep in range(500):
    optimer.zero_grad()
    hidden=Variable(torch.zeros(1,batch_size,n_hidden))

    output=model(hidden,input_batch)
    loss=criter(output,output_batch)
    if(ep+1)%100==0:
        print('Ep:','%04d'%(ep+1),'cost=','{:.6f}'.format(loss))
    loss.backward()
    optimer.step()

hidden = Variable(torch.zeros(1, batch_size, n_hidden))
predict = model(hidden, input_batch).data.max(1, keepdim=True)[1]
print([sen.split()[:2] for sen in sentences], '->', [number_dict[n.item()] for n in predict.squeeze()])