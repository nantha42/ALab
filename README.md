# RL - Power Environment
Agent needs to collect resources and build processor where the processor will produce items. Collecting this items gives more reward to the agent.

![Demo Run](images/3.png?raw=true "Demo 1") 

Check previous master commits. 

Run `main.py` in `src/RL`. Code your own agent and use the algorithm from the algorithm package to train the agent. Set the train to `False` while testing the agent.


```
s = Simulator(
        agent,env,trainer,
        nactions=6,
        log_message="Testing this high performance model", 
        visual_activations= True )
    print(s.visual_activations)
    s.run(1000,5000,train=False,render_once=10,saveonce=7)
```

Training results are stored in the logs folder. The name of the plots are given based on the timestamp the program is run. Make sure you give a appropriate log name for each training session, so you can later identify the model using the timestamp. This log messages are stored in the `src/RL/logs/log.json` file.

# RL -Multi Agent Training Using Gatherer Environment
![Multi Agent](images/4.gif?raw=true "Train Results") 

Latest master commit.
Use the class algorithm.reinforce.MultiEnvironmnetSimulator. 
```
#MULTI ENVIRONMENT TESTING
boxsize = 10
na = 3
n_envs = 16 
environments = [GathererState(gr=10,gc=10,vis=5,nagents=na,boxsize=boxsize,spawn_limit=5) for i in range(n_envs)]

model = StateRAgent(input_size=100,state_size=3,containers=len(environments))
model1 = StateRAgent(input_size=100,state_size=3,containers=len(environments))
model2 = StateRAgent(input_size=100,state_size=3,containers=len(environments))

models = [model,model1,model2]
s = MultiEnvironmentSimulator(
    models,environments,nactions=6,
    log_message="Testing with 4 Environments",
    visual_activations=True)

train = True 
s.run(1000,500,train=train,render_once=1,saveonce=2)
```

