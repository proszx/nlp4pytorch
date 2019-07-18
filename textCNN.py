import  numpy as np
import torch
import torch.nn as nn 
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F

dtype=torch.FloatTensor

embedding_size=2
seq_len=3
num_class=2
filter_sizes=[2,2,2]
num_filters=3

sentences = ["i love you", "he loves me", "she likes baseball", "i hate you", "sorry for that", "this is awful"]
labels = [1, 1, 1, 0, 0, 0]  # 1 is good, 0 is not good.

word_list=list(set(" ".join(sentences).split()))

word_dict={w:i for i,w in enumerate(word_list)}

vocab_size=len(word_dict)

inputs=[]

for sen in sentences:
    inputs.append(np.asarray([word_dict[n] for n in sen.split()]))

targets=[]

for l in labels:
    targets.append(l)
input_batch=Variable(torch.LongTensor(inputs))
output_batch=Variable(torch.LongTensor(targets))

class TextCNN(nn.Module):
    def __init__(self):
        super(TextCNN,self).__init__()
        self.num_filters_total=num_filters*len(filter_sizes)
        self.W=nn.Parameter(torch.empty(vocab_size,embedding_size).uniform_(-1,1)).type(dtype)
        self.Weight=nn.Parameter(torch.empty(self.num_filters_total,num_class).uniform_(-1,1)).type(dtype)
        self.Bias=nn.Parameter(0.1*torch.ones([num_class])).type(dtype)
    
    def forward(self,X):
        embedding_chars=self.W[X]
        embedding_chars=embedding_chars.unsqueeze(1)
        pool_output=[]

        for i in filter_sizes:
            conv=nn.Conv2d(1,num_filters,(i,embedding_size),bias=True)(embedding_chars)
            h=F.relu(conv)

            maxpool=nn.MaxPool2d((seq_len-i+1,1))

            pooled=maxpool(h).permute(0,3,2,1)

            pool_output.append(pooled)
        #xi:i+h−1 
        h_pool=torch.cat(pool_output,len(filter_sizes))
        h_pool_flat=torch.reshape(h_pool,[-1,self.num_filters_total])
        #ci = f(w · xi:i+h−1 + b).
        model=torch.mm(h_pool_flat,self.Weight)+self.Bias
        return model
model=TextCNN()
#This criterion combines nn.LogSoftmax and nn.NLLLoss in one single class.
criter=nn.CrossEntropyLoss()
#Adam: A Method for Stochastic Optimization_.
optimer=optim.Adam(model.parameters(),lr=0.001)

for ep in range(500):
    #input_batch,output_batch=random_batch(skip_grams,batch_size)
    #input_batch=Variable(torch.Tensor(input_batch))
    #output_batch=Variable(torch.LongTensor(output_batch))

    optimer.zero_grad()
    output=model(input_batch)

    loss=criter(output,output_batch)

    if(ep+1)%100==0:
        print("EP:",'%04d'%(ep+1),'cost=','{:.6f}'.format(loss))
    loss.backward()
    optimer.step()

 # Test
test_text = 'sorry hate you'
tests = [np.asarray([word_dict[n] for n in test_text.split()])]
test_batch = Variable(torch.LongTensor(tests))

# Predict
predict = model(test_batch).data.max(1, keepdim=True)[1]
if predict[0][0] == 0:
    print(test_text,"is Bad Mean...")
else:
    print(test_text,"is Good Mean!!")   