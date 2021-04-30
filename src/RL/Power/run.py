from agents import *
from tqdm import tqdm
import time
from Env import *
from utils import *


def update_policy(policy_network, rewards, log_probs):
    discounted_rewards = []

    for t in range(len(rewards)):
        Gt = 0 
        pw = 0
        for r in rewards[t:]:
            Gt = Gt + GAMMA**pw * r
            pw = pw + 1
        discounted_rewards.append(Gt)
        
    discounted_rewards = torch.tensor(discounted_rewards)
    discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9) # normalize discounted rewards

    policy_gradient = []
    for log_prob, Gt in zip(log_probs, discounted_rewards):
        policy_gradient.append(-log_prob * Gt)
    
    policy_network.optimizer.zero_grad()
    policy_gradient = torch.stack(policy_gradient).sum()
    policy_gradient.backward()
    policy_network.optimizer.step()


def train(config):
# def train(gsize: int,vsize: int,nactions: int,model_name: str,type="Default",load_model=None,nlayers = 2):
    """
        Train()
        Use to train the network
    """

    gsize = config.GSIZE
    vsize = config.VSIZE
    nactions = config.NACTIONS
    model_name = config.MODEL_NAME
    type = config.TYPE 
    nlayers = config.NLAYERS
    load_model = config.LOADMODEL
    HIDDEN_SIZE = config.HIDDEN_SIZE
    HIST_FILENAME = config.HIST_FILENAME
    PLOT_FILENAME = config.PLOT_FILENAME


    hs = 32
    kr,kc = gsize
    game = PowerGame(kr,kc,vsize)
    game.enable_draw = False
    steps = STEPS 
    episodes = EPISODES 
    recorder = RLGraph()

    if type == "Default":
        net = PolicyNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)
    elif type == "Atten":
        net = PolicyAttenNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)
    elif type == "Memory":
        net = PolicyGRUNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE,nlayers=nlayers)

    if load_model:
        net.load_state_dict(torch.load("../../../models/"+model_name))

    for i in range(episodes):
        hard_reset = False 
        game.enable_draw = True if i%5 == 0 else False
        game.reset(hard_reset) 
        log_probs = []
        rewards = []
        state = game.get_state().reshape(-1)
        if type == "Memory" or type=="Atten":
            net.reset()
        pbar = tqdm(range(steps),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        trewards = 0
        for j in pbar: 
            action,log_prob = net.get_action(state)
            avec = np.zeros((nactions));avec[action] = 1
            new_state,reward = game.act(avec)
            log_probs.append(log_prob)
            rewards.append(reward)
            trewards += reward
            state = new_state.reshape(-1)
            game.step()
            pbar.set_description(f"Episodes: {i:4} Rewards: {trewards:2}")
        recorder.newdata(trewards)
        update_policy(net,rewards,log_probs)
        show_once = 1 
        if i% show_once == show_once -1:
            recorder.plot(PLOT_FILENAME)
            recorder.save(HIST_FILENAME)
        torch.save(net.state_dict(),"../../../models/" + model_name) 
    recorder.save(HIST_FILENAME)

def test(config):

# def test(gsize: int,vsize: int,nactions: int,model_name: str,type: str,nlayers=2):
    gsize = config.GSIZE
    vsize = config.VSIZE
    nactions = config.NACTIONS
    model_name = config.MODEL_NAME
    type = config.TYPE 
    nlayers = config.NLAYERS
    HIDDEN_SIZE = config.HIDDEN_SIZE


    print(int(time.time())) 
    np.random.seed(int(time.time()))
    kr,kc = gsize
    game = PowerGame(kr,kc,vsize)
    steps = STEPS 
    episodes = EPISODES 
    if type == "Default":
        net = PolicyNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)
    elif type == "Atten":
        net = PolicyAttenNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)
    elif type == "Memory":
        net = PolicyGRUNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE,nlayers= nlayers)

    net.load_state_dict(torch.load("../../../models/"+ model_name))
    # state = torch.tensor(game.get_state(),dtype=torch.float).reshape(1,-1)
    for i in range(episodes): 
        hard_reset = False 
        game.reset(hard_reset) 
        state = game.get_state().reshape(-1)
        rewards = []
        # if type == "Memory" or type == "Atten":
            # net.reset()
        pbar = tqdm(range(steps),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        trewards = 0
 
        for j in pbar: 
            action,log_prob = net.get_action(state)
            avec = np.zeros((nactions));avec[action] = 1
            new_state,reward = game.act(avec)
            rewards.append(reward)
            trewards += reward
            pbar.set_description(f"Episode: {i:4} Rewards : {trewards}")
            state = new_state.reshape(-1)
            game.step()



class Config:
    def __init__(self,model_name):
        self.MODEL_NAME = model_name + ".pth"
        self.TYPE = "Default" 
        self.NACTIONS = 6 
        self.PLOT_FILENAME = model_name + ".png"
        self.HIST_FILENAME = model_name + ".pkl" 
        self.NLAYERS = 4 
        self.HIDDEN_SIZE = 64 
        self.VSIZE = 5
        self.GSIZE = (14,14)
        self.LOADMODEL = False



if __name__ == '__main__':
    EPISODES = 5000
    STEPS = 50000

    c = Config("TMAgent-S5")
    c.HIDDEN_SIZE =  128 
    c.TYPE = "Memory" 
    c.VSIZE = 5
    c.NACTIONS = 6
    c.NLAYERS = 4
    c.GSIZE= (30,10)
    
    # train(c)
    # HIDDEN_SIZE =  64 

    # MODEL_NAME = "PowerMAgent4Layerv2-S5"
    # TYPE = "Memory" 
    # VSIZE = 5
    # NACTIONS = 6
    # PLOT_FILENAME = MODEL_NAME + ".png" 
    # HIST_FILENAME = MODEL_NAME + ".pkl" 
    # NLAYERS = 4
    # GSIZE = (24,24)
    
    # train(  gsize=(14,14),
    #         vsize=VSIZE,
    #         nactions= NACTIONS,
    #         model_name = MODEL_NAME + ".pth", 
    #         type= TYPE,
    #         load_model = None,
    #         nlayers=4)
            
    test(c)
    # test(GSIZE,VSIZE, NACTIONS, MODEL_NAME + ".pth",type=TYPE,nlayers=NLAYERS)