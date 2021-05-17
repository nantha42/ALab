import pygame as py 
import typing
import numpy as np 
from icecream import ic


ENABLE_DRAW = True 
class Agent:
    def __init__(self,pos=[0,0],vsize):
        self.x,self.y = pos
        self.vissize = vsize  #not square number 
    

class PowerGame:
    def __init__(self,gr=10,gc=10,vis=7):
        self.box_size = 20 
        self.font = py.font.SysFont("times",20)
        self.grid = np.zeros((gr,gc))
        self.vissize = vis
        self.w = 50 + gc*self.box_size + 50
        self.h = 50 + gr*self.box_size + 50
        self.win =  py.Surface((self.w,self.h),py.DOUBLEBUF,32)
        self.agents = None 
        self.test = [0,0]
        self.start_position= [50,50]
        self.timer = 0
        self.res = 0
        self.items = 0
        self.collected = 0
        self.processors = 0
        self.agent_pos = np.array([0,0])
        self.enable_draw = ENABLE_DRAW 
        self.visibility = None
        self.processors = []
        self.current_step = 0
        self.total_rewards = 0
        
        #initializing agents
        self.RES = 9
        self.PROCESSOR = 8
        self.ITEM = 7

    def get_state(self,agentid):
        x,y = self.agents[agentid]
        vis = np.ones((self.vissize,self.vissize))*-1
        v,h = self.grid.shape
        for i in range(-2,3):
            for j in range(-2,3):
                if(0 <= x+i < v and 0 <= y+j < h):
                    vis[i+2,j+2] = self.grid[x+i,y+j] 
        self.visibility = vis
        return self.visibility 

