import matplotlib.pyplot as plt
import pickle 
import torch as T
import pandas as pd
import time
import json

class RLGraph:
    def __init__(self):
        self.hist = []
        self.directory = "logs/"
        self.run_name = str((time.time()))
        print("GRAPH runame = ",self.run_name)
        self.log_message = None
        self.final_reward = 0
        with open(self.directory+"log.json","r")  as f:
            self.logs = json.load(f) 

    def save(self):
        f = open(self.directory+"hists/"+self.run_name+".pkl","wb")
        pickle.dump(self.hist,f)
        with open(self.directory+"log.json","r")  as f:
            self.logs = json.load(f) 
        self.logs[self.run_name] = self.log_message
        with open(self.directory+"log.json","w") as f:
            json.dump(self.logs, f,indent=2)
    
    def load(self,name):
        f = open(self.directory+"hists/"+self.run_name+".pkl","rb")
        pickle.load(f,self.hist)

    def newdata(self,x):
        self.hist.append(x)
    
    def save_model(self,model):
        T.save(model.state_dict(),self.directory+"models/"+self.run_name+".pth")
        print("****** Model Saved ******")
    
    def plot(self):
        smoothed_rewards = pd.Series.rolling(pd.Series(self.hist), 10).mean()
        smoothed_rewards = [elem for elem in smoothed_rewards]
        self.final_reward = smoothed_rewards[-1]
        plt.plot(self.hist)
        plt.plot(smoothed_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Rewards')
        plt.savefig(self.directory+"plots/"+self.run_name+".png")
        plt.clf()