import sys
import torch  
import gym
import numpy as np  
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt

# Constants
GAMMA = 0.9
class PolicyNetwork(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_size, learning_rate=3e-4):
        super(PolicyNetwork, self).__init__()
        self.num_actions = num_actions
        self.hidden_size = hidden_size
        self.gru = nn.GRU(num_inputs,hidden_size,2)
        self.hidden = torch.zeros((2,1,hidden_size))
        self.linear1 = nn.Linear(hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, num_actions)
        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
    
    def reset(self):
        self.hidden = torch.zeros((2,1,self.hidden_size))

    def forward(self, state):
        x,self.hidden = self.gru(state.unsqueeze(0),self.hidden)
        x = F.relu(self.linear1(x.squeeze(0)))
        x = F.softmax(self.linear2(x), dim=1)
        return x 
    
    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob

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
    policy_gradient.backward(retain_graph = True)
    policy_network.optimizer.step()

def main():
    env = gym.make('CartPole-v0')
    policy_net = PolicyNetwork(env.observation_space.shape[0], env.action_space.n, 128)
    
    max_episode_num = 5000
    max_steps = 1000
    numsteps = []
    avg_numsteps = []
    all_rewards = []

    for episode in range(max_episode_num):
        state = env.reset()
        log_probs = []
        rewards = []
        policy_net.reset()

        for steps in range(max_steps):
            env.render()
            action, log_prob = policy_net.get_action(state)
            new_state, reward, done, _ = env.step(action)
            log_probs.append(log_prob)
            rewards.append(reward)

            if steps % 20 == 19 or done:
                update_policy(policy_net, rewards, log_probs)
                policy_net.zero_grad()
                if done:
                    numsteps.append(steps)
                    avg_numsteps.append(np.mean(numsteps[-10:]))
                    all_rewards.append(np.sum(rewards))
                    if episode % 1 == 0:
                        sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
                    break
            # if done:
            #     update_policy(policy_net, rewards, log_probs)
            #     numsteps.append(steps)
            #     avg_numsteps.append(np.mean(numsteps[-10:]))
            #     all_rewards.append(np.sum(rewards))
            #     if episode % 1 == 0:
            #         sys.stdout.write("episode: {}, total reward: {}, average_reward: {}, length: {}\n".format(episode, np.round(np.sum(rewards), decimals = 3),  np.round(np.mean(all_rewards[-10:]), decimals = 3), steps))
            #     break
            
            state = new_state
        
        plt.plot(numsteps)
        plt.plot(all_rewards)
        plt.xlabel('Episode')
        plt.ylabel('rewards')
        plt.savefig("../../graphs/reinforcegru_race.png")


if __name__ == '__main__':
    main()