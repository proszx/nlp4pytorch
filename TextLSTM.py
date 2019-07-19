import numpy as np 
import torch 
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

dtype=torch.FloatTensor

char_arr=[c for c in 'abcdefghijklmnopqrstuvwxyz']
word_dict={w: i for i,w in enumerate(char_arr)}
number_dict={i:w for i,w in enumerate(char_arr)}
len_n=len(word_dict)


seq_data=['make','need','coal','word', 'love', 'hate', 'live', 'home', 'hash', 'star','fork']


n_step=3
n_hidden=128

def make_batch(seq_data):
    input_batch,output_batch=[],[]

    for seq in seq_data:
        inputs=[word_dict[n] for n in seq[:-1]]
        target=word_dict[seq[-1]]

        input_batch.append(np.eye(len_n)[inputs])
        output_batch.append(target)
    input_batch=Variable(torch.Tensor(input_batch))
    output_batch=Variable(torch.LongTensor(output_batch))
    return input_batch,output_batch

class TextLSTM(nn.Module):
    def __init__(self):
        super(TextLSTM,self).__init__()

        self.lstm=nn.LSTM(input_size=len_n,hidden_size=n_hidden)
        self.W=nn.Parameter(torch.randn([n_hidden,len_n]).type(dtype))
        self.b=nn.Parameter(torch.randn([len_n]).type(dtype))

    def forward(self, X):
        inputs=X.transpose(0,1)
        hidden_state=Variable(torch.zeros(1,len(X),n_hidden))
        cell_state=Variable(torch.zeros(1,len(X),n_hidden))
        output,(_,_)=self.lstm(inputs,(hidden_state,cell_state))

        output=output[-1]

        model=torch.mm(output,self.W)+self.b
        return model

input_batch,output_batch=make_batch(seq_data)

model=TextLSTM()

criter=nn.CrossEntropyLoss()
optimer=optim.Adam(model.parameters(),lr=0.001)    

output=model(input_batch)

for ep in range(1000):
    optimer.zero_grad()
    output=model(input_batch)

    loss=criter(output,output_batch)

    if (ep + 1) % 100 == 0:
        print('Epoch:', '%04d' % (ep + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimer.step()
inputs = [sen[:3] for sen in seq_data]

predict = model(input_batch).data.max(1, keepdim=True)[1]
print(inputs, '->', [number_dict[n.item()] for n in predict.squeeze()])

