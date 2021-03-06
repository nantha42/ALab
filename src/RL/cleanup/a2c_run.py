from a2c_agents import *
from tqdm import tqdm
import time
from Env import *
from utils import *


GAMMA = 0.99

def update_policy(policy_network,rewards,values,log_probs,E,new_state,type):
    state = Variable(torch.from_numpy(new_state)).float().unsqueeze(0)
    if type == "Memory":
        state = state.unsqueeze(0)
    Qval,_ = policy_network.forward(state)
    Qval = Qval.detach().numpy()[0,0]
 
    Qvals = np.zeros_like(values)
    for t in reversed(range(len(rewards))):
        Qval = rewards[t] + GAMMA * Qval
        Qvals[t] = Qval
    
    values = torch.FloatTensor(values)
    Qvals = torch.FloatTensor(Qvals)
    log_probs = torch.stack(log_probs)

    advantage = Qvals - values
    actor_loss = (-log_probs * advantage).mean()
    critic_loss = 0.5*advantage.pow(2).mean()

    ac_loss = actor_loss + critic_loss + 0.001* E

    policy_network.optimizer.zero_grad()
    ac_loss.backward()
    policy_network.optimizer.step()


def update_policy_reinforce(policy_network, rewards, log_probs):
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
        net = ActorCritic(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)
    elif type == "Atten":
        net = PolicyAttenNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)
    elif type == "Memory":
        net = ActorCriticGRU1(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE,nlayers=nlayers)

    if load_model :
        net.load_state_dict(torch.load("../../models/"+model_name))

    for i in range(episodes):
        hard_reset = False 
        game.enable_draw = True if i%5==0 else False
        game.reset(hard_reset) 
        log_probs = []
        rewards = []
        values = []
        state = game.get_state().reshape(-1)
        if type == "Memory" or type=="Atten":
            net.reset()
        pbar = tqdm(range(steps),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        trewards = 0
        entropy_term = 0
        for j in pbar: 
            action,value,log_prob,entropy = net.get_action(state)
            avec = np.zeros((nactions));avec[action] = 1
            new_state,reward = game.act(avec)
            rewards.append(reward)
            values.append(value)
            log_probs.append(log_prob)
            entropy_term += entropy
            state = new_state
            trewards += reward
            state = new_state.reshape(-1)
            game.step()
            pbar.set_description(f"Episodes: {i:4} Rewards: {trewards:2}")

        recorder.newdata(trewards)
        update_policy(net,rewards,values,log_probs,entropy_term,state,type)
        show_once = 1 
        if i% show_once == show_once -1:
            recorder.plot(PLOT_FILENAME)
            recorder.save(HIST_FILENAME)
        torch.save(net.state_dict(),"../../models/" + model_name) 
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
        net = ActorCritic(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)
    elif type == "Atten":
        net = PolicyAttenNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)
    elif type == "Memory":
        net = ActorCriticGRU1(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE,nlayers= nlayers)

    net.load_state_dict(torch.load("../../models/"+ model_name))
    # state = torch.tensor(game.get_state(),dtype=torch.float).reshape(1,-1)
    for i in range(episodes): 
        hard_reset = False 
        game.reset(hard_reset) 
        state = game.get_state().reshape(-1)
        rewards = []
        if type == "Memory" or type == "Atten":
            net.reset()
        pbar = tqdm(range(steps),bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')
        trewards = 0
 
        for j in pbar: 
            action,value,log_prob,_ = net.get_action(state)
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
    ##GLOBAL##
    EPISODES = 5000
    STEPS = 500

    c = Config("A2C/TM4LAgentv2-S5")
    c.HIDDEN_SIZE =  128 
    c.TYPE = "Memory" 
    c.VSIZE = 5
    c.NACTIONS = 6
    c.NLAYERS = 4
    c.GSIZE= (14,14)

    train(c)
           
    # test(c)
    # test((14,14),VSIZE, NACTIONS, MODEL_NAME + ".pth",type=TYPE,nlayers=NLAYERS)