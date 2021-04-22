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



def train(gsize: int,vsize: int,nactions: int,model_name: str,type="Default",load_model=None):
    """
        Train()
        Use to train the network
    """
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
        net = PolicyGRUNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)

    if load_model is not None:
        net.load_state_dict(torch.load("../../models/"+load_model))

    for i in range(episodes):
        hard_reset = False 
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
        show_once = 100 
        if i% show_once == show_once -1:
            recorder.plot(PLOT_FILENAME)
            recorder.save(HIST_FILENAME)
        torch.save(net.state_dict(),"../../models/" + model_name) 
    recorder.save(HIST_FILENAME)

def test(gsize: int,vsize: int,nactions: int,model_name: str,type: str):
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
        net = PolicyGRUNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)

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
            action,log_prob = net.get_action(state)
            avec = np.zeros((nactions));avec[action] = 1
            new_state,reward = game.act(avec)
            rewards.append(reward)
            trewards += reward
            pbar.set_description(f"Episode: {i:4} Rewards : {trewards}")
            state = new_state.reshape(-1)
            game.step()

if __name__ == '__main__':
    EPISODES = 5000
    STEPS = 10000
    HIDDEN_SIZE =  64 
    MODEL_NAME = "PowerMAgentv2-S7"
    
    PLOT_FILENAME = MODEL_NAME + ".png" 
    HIST_FILENAME = MODEL_NAME + ".pkl" 

    # train(  gsize=(14,14),
    #         vsize=7,
    #         nactions=6,
    #         model_name = MODEL_NAME + ".pth", 
    #         type="Memory",
    #         load_model = None)


    # train((14,14),7,6,"PowerAgentv2-S7.pth")
    # input()
    # train((10,10),5,"PAAgent-S5.pth",mem=True,load_model="PAAgent-S5.pth")
    # train((20,30),5,"PowerAgent-S5.pth")
    # test((30,30),5,"PowerAgent-S5.pth",)
    test((14,14),7,6, MODEL_NAME + ".pth",type="Memory")