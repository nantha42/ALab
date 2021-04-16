from agents import *
from Env import *

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



def train():
    game = PowerGame()
    game.enable_draw = False
    steps = 400 
    episodes = 500
    net = PolicyNetwork(num_inputs=7*7,num_actions=5,hidden_size=128)
    # state = torch.tensor(game.get_state(),dtype=torch.float).reshape(1,-1)
    for i in range(episodes):
        game.reset() 
        log_probs = []
        rewards = []
        state = game.get_state().reshape(-1)
        for i in range(steps):
            action,log_prob = net.get_action(state)
            avec = np.zeros((5));avec[action] = 1
            new_state,reward = game.act(avec)
            log_probs.append(log_prob)
            rewards.append(reward)
            state = new_state.reshape(-1)
            game.step()
        print("Total Reward: ",sum(rewards))
        update_policy(net,rewards,log_probs)
        torch.save(net.state_dict(),"../../models/PowerAgent.pth")

def test():
    game = PowerGame(20,20)
    steps = 1000 
    episodes = 500
    net = PolicyNetwork(num_inputs=7*7,num_actions=5,hidden_size=128)
    net.load_state_dict(torch.load("../../models/PowerAgent.pth"))
    # state = torch.tensor(game.get_state(),dtype=torch.float).reshape(1,-1)

    for i in range(episodes):
        game.reset() 
        state = game.get_state().reshape(-1)
        rewards = []
        for i in range(steps):
            action,log_prob = net.get_action(state)
            avec = np.zeros((5));avec[action] = 1
            new_state,reward = game.act(avec)
            rewards.append(reward)
            state = new_state.reshape(-1)
            game.step()
        print("Total Reward: ",sum(rewards))



if __name__ == '__main__':
    # train()
    test()