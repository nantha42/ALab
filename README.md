# RL - PowerAgent  
Agent needs to collect resources and build processor where the processor will produce items. Collecting this items gives more reward to the agent.

![Demo Run](images/3.png?raw=true "Demo 1") 

Run `main.py` in `src/RL`. 
Code your own agent and use the algorithm from the algorithm package to train the agent. Set the train to `False` while testing the agent.


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

------

![Train Results](src/RL/logs/plots/1620290798.png?raw=true "Train Results") 
Plot of rewards obtained in a training from logs/plot

-----------------

