## word2vec skip-gram 语法
import numpy  as  np 
import torch
import torch.nn as nn 
import torch.optim as  optim
from torch.autograd import Variable
import matplotlib.pyplot as plt

dtype=torch.FloatTensor

sentences=["i like dog", "i like cat", "i like animal",
              "dog cat animal", "apple cat dog like", "dog fish milk like",
              "dog cat eyes like", "i like apple", "apple i hate",
              "apple i movie book music like", "cat dog hate", "cat dog like"]
word_seq=" ".join(sentences).split()
word_list=list(set(" " .join(sentences).split()))

word_dict={w:i for i,w in enumerate(word_list)}

batch_size=20
embedding_size=2
voc_size=len(word_list)


def random_batch(data,size):
    random_inputs=[]
    random_labels=[]
    random_index=np.random.choice(range(len(data)),size,replace=False)

    for i in random_index:
        random_inputs.append(np.eye(voc_size)[data[i][0]])
        random_labels.append(data[i][1])

    return random_inputs,random_labels
skip_grams=[]

for i in range(1,len(word_seq)-1):
    target=word_dict[word_seq[i]]
    text=[word_dict[word_seq[i-1]],word_dict[word_seq[i+1]]]

    for w in text:
       # print([target,w])
        skip_grams.append([target,w])
class Word2vec(nn.Module):
    def __init__(self):
        super(Word2vec,self).__init__()
        self.W=nn.Parameter(-2*torch.rand(voc_size,embedding_size)+1).type(dtype)
        self.WT=nn.Parameter(-2*torch.rand(embedding_size,voc_size)+1).type(dtype)

    
    def forward(self,X):
        hidden_layer=torch.matmul(X,self.W)
        output_layer=torch.matmul(hidden_layer,self.WT)
        return output_layer

model=Word2vec()
## 惩罚函数 
criter=nn.CrossEntropyLoss()

optimer=optim.Adam(model.parameters(),lr=0.001)

for ep in range(500):
    input_batch,output_batch=random_batch(skip_grams,batch_size)
    input_batch=Variable(torch.Tensor(input_batch))
    output_batch=Variable(torch.LongTensor(output_batch))

    optimer.zero_grad()
    output=model(input_batch)

    loss=criter(output,output_batch)

    if(ep+1)%100==0:
        print("EP:",'%04d'%(ep+1),'cost=','{:.6f}'.format(loss))
    loss.backward()
    optimer.step()

for i,label in enumerate(word_list):
    W,WT=model.parameters()
    x,y=float(W[i][0]),float(W[i][1])
    #plt.scatter(x, y)
    #plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
#plt.show()

             
