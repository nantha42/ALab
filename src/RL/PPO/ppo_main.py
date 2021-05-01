import gym 
import numpy as np 
from ppo import Agent
import matplotlib.pyplot as plt
#from utils import plot_learning_curve

if __name__ == '__main__':
	env = gym.make('CartPole-v0')
	N = 20
	batch_size = 5
	n_epochs = 4
	alpha = 0.0003
	agent = Agent(n_actions=env.action_space.n,
			batch_size = batch_size,
			alpha = alpha,
			n_epochs = n_epochs,
			input_dims = env.observation_space.shape)

	# agent.load_models()
	n_games = 300
	figure_file = '../../../graphs/ppo_cartpol.png'
	best_score = env.reward_range[0]
	score_history = []

	learn_iters = 0
	avg_score = 0

	for i in range(n_games):
		observation = env.reset()
		done = False
		score = 0
		n_steps = 0
		while not done:
			action,prob,val = agent.choose_action(observation)
			env.render()
			observation_ ,reward, done, info = env.step(action)
			n_steps +=1 
			score += reward
			agent.remember(observation, action, prob, val,reward, done)
			if n_steps % N == 0:
				agent.learn()
				learn_iters += 1
			observation = observation_
		score_history.append(score)
		avg_score = np.mean(score_history[-100:])

		if avg_score > best_score:
			best_score = avg_score
			# agent.save_model()
			
		print(f"episode: {i} score {score:3} avg {avg_score:4} steps {n_steps:4}")

		x = [i+1 for i in range(len(score_history))]
		plt.plot(x,score_history)
		plt.savefig(figure_file)
#		plot_learning_curve(x,score_history,figure_file)
