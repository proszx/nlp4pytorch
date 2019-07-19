import numpy as np 
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as func
from torch.autograd import Variable


dtype=torch.FloatTensor

sentence = (
    'Lorem ipsum dolor sit amet consectetur adipisicing elit '
    'sed do eiusmod tempor incididunt ut labore et dolore magna '
    'aliqua Ut enim ad minim veniam quis nostrud exercitation'
)

word_list=enumerate(list(set(sentence.split())))
word_dict={w:i for i,w in word_list}
num_dict={i:w for i,w in word_list}
len_n=len(word_dict)
max_len=len(sentence.split())

n_hidden=5

def make_batch(sentence):
    input_batch,output_batch=[],[]

    words=sentence.split()

    for i,w in enumerate(words[:-1]):
        inputs=[word_dict[n] for n in words[:(i+1)]]
        inputs=inputs+[0]*(max_len-len(inputs))
        target=word_dict[words[i+1]]

        input_batch.append(np.eye(len_n)[inputs])
        output_batch.append(target)
    input_batch=Variable(torch.Tensor(input_batch))
    output_batch=Variable(torch.LongTensor(output_batch))

    return input_batch,output_batch

class BiLSTM(nn.Module):
    def __init__(self):
        super(BiLSTM,self).__init__()

        self.lstm=nn.LSTM(input_size=len_n,hidden_size=n_hidden,bidirectional=True)
        self.W=nn.Parameter(torch.randn([n_hidden*2,len_n]).type(dtype))
        self.b=nn.Parameter(torch.randn([len_n]).type(dtype))

    def forward(self, X):
        inputs=X.transpose(0,1)
        hidden_state=Variable(torch.zeros(1*2,len(X),n_hidden))
        cell_state=Variable(torch.zeros(1*2,len(X),n_hidden))
        output,(_,_)=self.lstm(inputs,(hidden_state,cell_state))

        output=output[-1]

        model=torch.mm(output,self.W)+self.b
        return model


input_batch,output_batch=make_batch(sentence)

model=BiLSTM()

criter=nn.CrossEntropyLoss()
print(model.parameters())
optimer=optim.Adam(model.parameters(),lr=0.001)

for ep in range(10000):
    optimer.zero_grad()
    output=model(input_batch)

    loss=criter(output,output_batch)

    if(ep+1)%1000==0:
         print('Epoch:', '%04d' % (ep + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimer.step()

predict = model(input_batch).data.max(1, keepdim=True)[1]
print(sentence)
print([num_dict[n.item()] for n in predict.squeeze()])
