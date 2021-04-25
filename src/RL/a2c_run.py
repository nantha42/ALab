from a2c_agents import *
from tqdm import tqdm
import time
from Env import *
from utils import *


GAMMA = 0.99

def update_policy(policy_network,rewards,values,log_probs,E,new_state):
    state = Variable(torch.from_numpy(new_state)).float().unsqueeze(0)
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


def train(gsize: int,vsize: int,nactions: int,model_name: str,type="Default",load_model=None,nlayers = 2):
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
        net = ActorCritic(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)
    elif type == "Atten":
        net = PolicyAttenNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE)
    elif type == "Memory":
        net = PolicyGRUNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE,nlayers=nlayers)

    if load_model is not None:
        net.load_state_dict(torch.load("../../models/"+load_model))

    for i in range(episodes):
        hard_reset = False 
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
        update_policy(net,rewards,values,log_probs,entropy_term,state)
        show_once = 100 
        if i% show_once == show_once -1:
            recorder.plot(PLOT_FILENAME)
            recorder.save(HIST_FILENAME)
        torch.save(net.state_dict(),"../../models/" + model_name) 
    recorder.save(HIST_FILENAME)

def test(gsize: int,vsize: int,nactions: int,model_name: str,type: str,nlayers=2):
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
        net = PolicyGRUNetwork(num_inputs=vsize*vsize,num_actions=nactions,hidden_size=HIDDEN_SIZE,nlayers= nlayers)

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

if __name__ == '__main__':
    EPISODES = 5000
    STEPS = 700
    HIDDEN_SIZE =  64 
    MODEL_NAME = "A2C/Agent-S7"
    TYPE = "Default" 
    VSIZE = 7
    NACTIONS = 6
    PLOT_FILENAME = MODEL_NAME + ".png" 
    HIST_FILENAME = MODEL_NAME + ".pkl" 
    NLAYERS = 2
    
    # train(  gsize=(14,14),
    #         vsize=VSIZE,
    #         nactions= NACTIONS,
    #         model_name = MODEL_NAME + ".pth", 
    #         type= TYPE,
    #         load_model = None,
    #         nlayers=NLAYERS)
            

    test((14,14),VSIZE, NACTIONS, MODEL_NAME + ".pth",type=TYPE,nlayers=NLAYERS)