import numpy as np 
import torch 
import torch.nn as nn 
import torch.optim as optim
import torch.nn.functional as functional 
from torch.autograd import Variable
import matplotlib.pyplot as plt 
dtype=torch.FloatTensor
sentences = ['ich mochte ein bier P', 'S i want a beer', 'i want a beer E']

word_list=list(set(" ".join(sentences).split()))
word_dict={w:i for i,w in enumerate(word_list)}
num_dict={i:w for i,w in enumerate (word_list)}

len_n=len(word_dict)

n_hidden=128
def make_batch(sentences):
    input_batch=Variable(torch.Tensor([np.eye(len_n)[[word_dict[n] for n in sentences[0].split()]]]))
    output_batch=Variable(torch.Tensor([np.eye(len_n)[[word_dict[n] for n in sentences[1].split()]]]))
    target_batch=Variable(torch.LongTensor([[word_dict[n] for n in sentences[2].split()]]))

    return input_batch,output_batch,target_batch

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

class Seq2Seq_attn(nn.Module):
    def __init__(self,enc,dec):
        super(Seq2Seq_attn,self).__init__()

        self.enc_cell=enc
        self.dec_cell=dec

        self.attn=nn.Linear(n_hidden,n_hidden)
        self.linear=nn.Linear(n_hidden*2,len_n)
    def get_attn_score(self,dec_output,enc_output):
        score=self.attn(enc_output)
        #dot运算 [1,2,3],[1,2,3]元素内积 1*1+2*2+3*3
        return torch.dot(dec_output.view(-1),score.view(-1))
    def get_attn_weight(self,dec_output,enc_output):
        n_step=len(enc_output)
        attn_score=Variable(torch.zeros(n_step))
        for i in range(n_step):
            attn_score[i]=self.get_attn_score(dec_output,enc_output[i])
        return functional.softmax(attn_score).view(1,1,-1)
       # return functional.relu(attn_score).view(1,1,-1)
    def forward(self,enc_input,hidden,dec_input):
        enc_input=enc_input.transpose(0,1)
        dec_input=dec_input.transpose(0,1)


        enc_output,enc_hidden=self.enc_cell(enc_input,hidden)

        trained_attn=[]

        hidden=enc_hidden
        n_step=len(dec_input)
        model=Variable(torch.empty([n_step,1,len_n]))

        for i in range(n_step):
            dec_output,hidden=self.dec_cell(dec_input[i].unsqueeze(0),hidden)
            attn_weight=self.get_attn_weight(dec_output,enc_output)
            trained_attn.append(attn_weight.squeeze().data.numpy())

            text=attn_weight.bmm(enc_output.transpose(0,1))
            dec_output=dec_output.squeeze(0)
            text=text.squeeze(1)
            model[i]=self.linear(torch.cat((dec_output,text),1))

        return model.transpose(0,1).squeeze(0),trained_attn

input_batch,output_batch,target_batch=make_batch(sentences)

hidden=Variable(torch.zeros(1,1,n_hidden))

enc=Encoder()
dec=Decoder()

model=Seq2Seq_attn(enc,dec)

criter=nn.CrossEntropyLoss()
optimer=optim.Adam(model.parameters(),lr=0.001)
for epoch in range(2000):
    optimer.zero_grad()
    output, _ = model(input_batch, hidden, output_batch)

    loss = criter(output, target_batch.squeeze(0))
    if (epoch + 1) % 400 == 0:
        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.6f}'.format(loss))

    loss.backward()
    optimer.step()

# Test
test_batch = [np.eye(len_n)[[word_dict[n] for n in 'SPPPP']]]
test_batch = Variable(torch.Tensor(test_batch))
predict, trained_attn = model(input_batch, hidden, test_batch)
predict = predict.data.max(1, keepdim=True)[1]
print(sentences[0], '->', [num_dict[n.item()] for n in predict.squeeze()])

# Show Attention
fig = plt.figure(figsize=(5, 5))
ax = fig.add_subplot(1, 1, 1)
ax.matshow(trained_attn, cmap='viridis')
ax.set_xticklabels([''] + sentences[0].split(), fontdict={'fontsize': 14})
ax.set_yticklabels([''] + sentences[2].split(), fontdict={'fontsize': 14})
plt.show()
