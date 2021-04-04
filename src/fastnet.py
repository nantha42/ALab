import numpy as np
import time
from tqdm import tqdm
from icecream import ic
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn


inp_dim = 500

class Game:
    def __init__(self):
        self.board = np.array([[0,0,0],[0,0,0],[0,0,0]])
        print(self.board.shape) 
        
    def play(self,r,c,p):
        if self.board[r][c] == 0:
            self.board[r][c] = p/2
            return True
        else: return False

    def gb(self):
        return torch.tensor(self.board,dtype=torch.float32).reshape(1,-1)

class Straight(nn.Module):
    def __init__(self):
        super().__init__()
        self.name="stratight"
        self.a1 = nn.Sequential(
            nn.Linear(inp_dim,20),
            nn.ReLU(),
            nn.Linear(20,10),
            nn.ReLU(),
            nn.Linear(10,2),

        )
    def forward(self,x):
        return self.a1(x)
        
class XO(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "Xo"
        ind = 4
        dim = inp_dim 
        self.a1 = nn.Sequential(
                 nn.Linear(dim,ind),
                 nn.ReLU(),
                 nn.Linear(ind,dim),
                 nn.ReLU())
                 
        self.a2 =  nn.Sequential(
             nn.Linear(dim,ind),
             nn.ReLU(),
             nn.Linear(ind,dim),
             nn.ReLU())
        
    def forward(self,x):
        u = self.a1(x)
        v = self.a2(x)
        r = torch.sum(x*u,dim=-1).view(-1,1)
        s = torch.sum(x*v,dim=-1).view(-1,1)
        out = torch.cat([r,s],dim=-1)
        return out


class product(nn.Module):
    def __init__(self):
        super().__init__()
        self.a1 = nn.Linear(9,5)
        self.name = "product"
        self.a2 = nn.Linear(9,5)
        self.a3 = nn.Linear(5,2)


    def forward(self,x):
        a = self.a1(x)
        b = self.a2(x)
        g = a*b
        c = self.a3(g)
        return c 
        
class Samecatmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "samecatmodel"
        self.a1 = nn.Linear(9,20)
        self.a3 = nn.Linear(20,2)

    def forward(self,x):
        return self.a3(self.a1(x))

class catmodel(nn.Module):
    def __init__(self):
        super().__init__()
        self.name = "catmodel"
        self.a1 = nn.Linear(9,10)
        self.a2 = nn.Linear(9,10)
        self.a3 = nn.Linear(20,2)

    def forward(self,x):
        u1 = self.a1(x)
        u2 = self.a2(x)
        catted = torch.cat((u2,u1),dim=-1)
        o = self.a3(catted)
        return o

class Test():
    def __init__(self,mod):
        self.model = mod
        self.optim = torch.optim.Adam(mod.parameters(),lr=0.001)
        self.loss_fn = torch.nn.CrossEntropyLoss()
        self.data = []
        self.avgtime= []

    def fbtime(self):
        return np.mean(self.avgtime) 

    def step(self,x,y):
        start = time.time() 
        o = self.model(x)
        
        self.avgtime.append(time.time()-start)
        loss = self.loss_fn(o,y)
        loss.backward()
        self.optim.step()
        self.optim.zero_grad()
        self.data.append(loss.item())
        
class TestAll:
    def __init__(self,models):
        self.testers = []
        self.models = models 
        for m in models:
            self.testers.append(Test(m))
            count = sum(p.numel() for p in m.parameters())
            print(m.name," : ",count)
    
    def step(self,x,y):
        for t in self.testers:
            t.step(x,y)
    
    def plot(self):
        legs = []

        for m in self.testers:
            count = sum(p.numel() for p in m.model.parameters())
            print(m.model.name," : ",count," FB time: ",m.fbtime())
 
        for m in self.testers:
            plt.plot(range(len(m.data)),m.data)
            legs.append(m.model.name)
        plt.legend(legs)
        plt.show()
        

if __name__ == '__main__':

    mod = product()
    mod1 = Straight()
    mod2 = catmodel()
    mod3 = XO()
    mod4 = Samecatmodel()
    testall = TestAll([mod1,mod3,])

    ds = 8000
    epoch = 1000
    x = torch.randn((ds,inp_dim))
    g = torch.tensor(torch.randint(0,2,(ds,)),dtype=torch.long)

    pbar = tqdm(range(epoch))
    for i in pbar :
        testall.step(x,g)
    testall.plot()
        # time.sleep(0.01)