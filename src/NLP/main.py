import torch as T
import numpy as np
import pygame as py
import time
import torch.nn as nn
import torch.optim as optim
import string
import unicodedata
from tqdm import tqdm 

class Model(nn.Module):
    def __init__(self,ntokens,batch_size,seq_len):
        super().__init__()
        self.emb = nn.Embedding(ntokens,10)
        self.ntokens = ntokens
        hs =256 
        self.gru = nn.GRU(10,hs,1,batch_first=True)
        self.n1 = nn.Linear(hs,64)
        self.n12 = nn.Linear(64,64)
        self.n2 = nn.Linear(64,ntokens)
        self.bs = batch_size 
        self.seq_len = 1
        self.hidden = T.zeros((self.seq_len,self.bs,hs))
    
    def reset(self):
        hs =256 
        self.hidden = T.zeros((self.seq_len,self.bs,hs))

    def forward(self,x):
        x = self.emb(x)
        u,self.hidden = self.gru(x,self.hidden)
        u = self.n1(u)
        u = nn.functional.relu(u)
        u = self.n12(u)
        u = nn.functional.relu(u)
        u = self.n2(u)
        return u

filename = "temp.txt"
text = ""
bs =80 
seq_len = 30
letters = string.ascii_letters + "\".,?;'-:1234567890 "


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in letters
    )

with open(filename,"r",encoding='utf-8') as f:
    text = f.read()
    out = ""
    for c in text:
        out += unicodeToAscii(c)
    text = out


def convert_tokens(text):
    tokens = []
    for char in text:
        tokens.append(letters.index(char))
    return tokens

n_letters = len(letters)


model = Model(n_letters,bs,seq_len)

def batchify_text(text,seq_size):
    input_data = []
    tar_data = []
    for i in range(0,len(text)-seq_size-1,seq_size):
        clip = text[i:i+seq_size]
        clip1 = text[i+1:i+seq_size+1]
        tokens = convert_tokens(clip)
        tokens1 = convert_tokens(clip1)
        input_data.append(tokens)
        tar_data.append(tokens1)
    return input_data,tar_data 

inp,tar = batchify_text(text,seq_len)
inp = np.array(inp)
tar = np.array(tar)


class Trainer:
    def __init__(self,model):
        self.model = model
        self.optim = optim.Adam(self.model.parameters(),lr=0.01)
        self.criterion = nn.CrossEntropyLoss()
        self.name = str(time.time())+".pth"
    
    def train(self,epochs):
        trange = tqdm(range(epochs))
        self.model.train()
        for _ in trange:
            for i in range(0,len(inp)-bs,bs):
                self.model.reset() 
                x = inp[i:i+bs]
                y = T.tensor(tar[i:i+bs])
                x = T.tensor(x)
                o = self.model(x)
                o = o.reshape((-1,n_letters))
                y = y.reshape(-1)
                loss = self.criterion(o,y.reshape(-1))
                
                trange.set_description(f"B: {i} Loss: {loss.item()}")
                loss.backward()
                self.optim.step()
                self.optim.zero_grad()
            if _%10 == 0:
                self.save_model()

            if _%50==0:
                self.test()
    
    def test(self):
        with T.no_grad():
            self.model.reset()
            self.model.eval()
            hs = 256
            self.model.hidden = T.zeros((1,1,hs))

            start = input("Starting Character: ") 
            leng = int(input("Length:"))
            x = convert_tokens(start) 
            proc = ""
            x = T.tensor([x])
            for _ in range(leng):
                u = self.model(x) 
                x = u
                ind = np.argmax(u.detach().numpy())
                x = T.tensor([[ind]])
                proc += letters[ind]
            print(proc)
        
            
    def save_model(self):
        T.save(self.model.state_dict(),"./logs/models/"+self.name)

    def load_model(self):
        print("Loaded")
        self.model.load_state_dict(T.load("./logs/models/1625655173.583472.pth"))


trainer = Trainer(model)
trainer.load_model()
#trainer.train(200)
for i in range(10):
    trainer.test()



