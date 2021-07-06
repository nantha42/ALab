import matplotlib.pyplot as plt
import pickle 
import pandas as pd

class RLGraph:
    def __init__(self):
        self.hist = []

    def save(self,name):
        f = open("../../../history/"+name,"wb")
        pickle.dump(self.hist,f)
    
    def load(self,name):
        f = open("../../../history/"+name,"rb")
        pickle.load(f,self.hist)

    def newdata(self,x):
        self.hist.append(x)
    
    def plot(self,name):
        smoothed_rewards = pd.Series.rolling(pd.Series(self.hist), 10).mean()
        smoothed_rewards = [elem for elem in smoothed_rewards]
        plt.plot(self.hist)
        plt.plot(smoothed_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig("../../../graphs/"+name)
        plt.clf()