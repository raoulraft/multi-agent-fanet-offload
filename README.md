# multi-agent-fanet-offload# UAV Offloading optimization using Deep Reinforcement Learning
## Introduction
In this scenario, each **UAV** is equipped with a **computing element (CE)** that provides computing facilities to **packets** arriving from devices (sensors, smartphones, etc.) installed on the ground area it is covering. 

<p align="center">
  <img src="https://github.com/raoulraft/uav_rl_stable_baselines/blob/main/images/reference_system.PNG" width="350" class="center">
</p>

All the UAVs have identical CEs. The mean packet arrival rate to each UAV from the zone it is covering depends on the current **state of the zone**: the higher the activity of a zone, the higher the packet arrival rate. 

When a packet is received from an UAV, it is processed by the CE installed onboard. If the CE is processing another job, the arriving ones are enqueued in the **Processing Queue, QP** . Here the packet waits until all the other packets (that previously arrived to the queue) are processed by the CE, according to a First-In-First-Out (FIFO) policy. 

However, during high-activity periods of some zones, Processing Queues of UAVs covering them can saturate, causing unacceptable delays and even losses. This would cause high variability in processing delays, with some jobs experiencing very high delays while other ones low delays. In addition, for the same reasons, waste of computing capacities can occur in UAVs when the zones they are covering are in low-activity states. 

For this reason, in order to smooth performance and decrease computing resource waste, we allow **horizontal offload** among UAVs. This consists in giving the possibility to overloaded UAVs to **offload** jobs to other, less-loaded, UAVs, with a resulting **load balancing**. We call the UAV that has received the job from the ground the Dwell UAV, whereas the UAV where the job is processed the Processing UAV.

When the link from the Dwell UAV to the Processing UAV is busy transmitting a packet, new packets to be offloaded are enqueued in the **Offloading Queue, Qol** . Therefore, each offloaded packets suffers an overall delay that it the sum of an offloading delay, that is the delay suffered in the Offloading Queue, and the Processing Queue delay, i.e. the delay suffered in the Processing Queue QP of the "target" Processing UAV.
<p align="center">
  <img src="https://github.com/raoulraft/uav_rl_stable_baselines/blob/main/images/drone.PNG" width="300" class="center">
</p>

Challenge: offloaded packets suffer an additional delay due to the transmission from the one UAV to another one. Optimize the offloading scheduling process.


## Main system parameters
- **N**: number of UAVs

- **Q_p**: processing queue

- **Q_o**: offloading queue

- **μ_p**: computing element frequency (Qp departure rate)

- **μ_o**: transmission rate (Qol departure rate)

- **K_p**: max processing queue Qp capacity

- **K_o**: max processing queue Qo capacity

- **State**: [Qp1, Qo1, Zi1], [Qp2, Qo2, Zi2], ..., [QpN, QoN, ZiN]

- **Action**: choose, for each drone, the packet offloading probability.

- **Reward**: - (number of packets in the FANET)

## Code

1. Environment is initialized (using values from input.py)
2. Agent is initialized
3. First observation is returned to the agent using state = env.reset()
4. The agent selects an action using model.predict(state)
5. The action is executed inside the environment, next_state and reward are returned next_state, reward, done = env.step(action)

Loop 4. and 5. until done = True (episode ends when t reaches time_limit, which is set in input_data)

The agent can also be trained using a fixed number of steps. In that case, the run could end before t reaching time_limit

### Events

during env.step there are 4 possible events:
1. Packet arrival
2. Packet processing
3. Packet offloading
4. Change Zone state

###### Packet arrival
A new packet arrives at the *i* UAV, where *i = row* (explained in TimeMatrix).
The scheduler inside the drone decides whether to offload or process locally the packet based on the *offloading_probability*.
- If the packet is offloaded, then
  1. if the offloading queue is *not* empty, then the packet is inserted in the offloading queue
  2. if the offloading queue is empty, then *TimeMatrix.update_matrix( )* is called immediately. A new offloading event will then be generated. Offloading will be completed when *env.t* reaches that t_event
- If the packet is processed, then
  1. if the processing queue is *not* empty, then the packet is inserted in the processing queue
  2. if the processing queue is empty, then *TimeMatrix.update_matrix( )* is called immediately. A new processing event will then be generated. Processing will be completed when *env.t* reaches that t_event

Finally, *zone.schedule_next_arrival( )* is called, which in turn calls *TimeMatrix.update_matrix( )* to schedule the next job arrival event

###### Packet processing 
A packet is processed by the CE. The packet is popped from the processing queue Qp. Delay metrics are then saved.
- If the packet was processed, then the *total delay, processing delay, offloading delay* are saved using env.update_metrics(...)
- Otherwise, only the *total delay* and the *processing delay* are saved using env.update_metrics(...)

All these metrics are reetrieved through *Packet.get_delay( ), Packet.get_processing_delay( ) and Packet.get_offloading_delay( )

- If the processing queue is not empty, then *TimeMatrix.update_matrix( )* is called immediately. A new processing event will then be generated. Processing (of the next packet in the queue) will be completed when *env.t* reaches that t_event
- If the processing queue is empty, *TimeMatrix.update_matrix( )* is called, in order to set the next processing event to *inf* (this will change to some value a new job arrives from the ground devices and has to be processed)

###### Packet offloading
A packet is offloaded by the CE. The packet is offloaded from the offloading queue Qo. Offloading delay metric is then saved inside the packet through *Packet.set_offloaded( )*.
- If the offloading queue is not empty, then *TimeMatrix.update_matrix( )* is called immediately. A new offloading event will then be generated. Offloading (of the next packet in the queue) will be completed when *env.t* reaches that t_event
- If the offloading queue is empty,*TimeMatrix.update_matrix( )* is called, in order to set the next offloading event to *inf* (this will change to some value a new job arrives from the ground devices and has to be offloaded)

###### Change Zone state
The state of the zone changes (from 0 to 1 and viceversa). This in turn also causes the packet arrival rate from the zones to change. Also, *Time.Matrix.update_matrix( )* is called, in order to schedule the next change zone state event

###### End of run
When env.t reaches its maximum value, which is usually set inside input.py, env.step returns *done = True* (in addition to the usual next_state and reward). When this happens, *env.save_history( )* is called, which saves the main episode metrics in *wandb*.

### Time Matrix
The way the time matrix works is the following:

A *(N,4)* array is instantiated, where *N* is equal to the number of drones, and 4 is the number of events (zone change state, packet arrival, processing, and offloading)

Specifically:
- zone change state = 0
- packet arrival = 1
- packet processing = 2
- packet offloading = 3

At the start of the run, only the *packet arrival* and the *zone change state* column will already have non-infinite value assigned. This is due to the fact that there are no packets inside the drones queues. When a packet arrives, a new event will be scheduled at the corrispondent column (e.g. if a packet arrives at the drone #0, then the offloading event would be scheduled in *row 0 column 3* (0 since it is drone 0, 3 since offloading is the 3rd type of event)).

E.g.

env.init()

*env.t = 0s*

    1     inf   inf   1.245
    1.2   inf   inf   1.194
    1.434 inf   inf   1.343
    1.98  inf   inf   1.443
 
 env.step() 
 
 *env.t goes to 1s*
 
 *first event is row 0 and column 0*
 
 *update_matrix is called the next event (0,0) is now scheduled at time 2.3s*
            
 
    2.3   inf   inf   1.245
    1.2   inf   inf   1.194
    1.434 inf   inf   1.343
    1.98  inf   inf   1.443
    
 env.step()
 
 *second event is row 1 column 3 (since it has the lowest value among all), which is a job arrival. Suppose the scheduler decides to offload the packet. Then, an offload event is scheduled at 1.54s*
 
  
    2.3   inf   inf   1.245
    1.2   inf   1.54  2.431
    1.434 inf   inf   1.343
    1.98  inf   inf   1.449

## Module

###### Drone.py

- init  (...)
- job_arrival (...)
- job_processing (...)
- job_offloading (...)
- others

###### Packet.py

- init (...)
- get_delay (...)
- get_processing_delay (...)
- get_offloading_delay (...)
- set_offloaded (...)

###### Zone.py

- init (...)
- change_zone_state (...)
- schedule_next_arrival (...)
- others

###### DronesEnv.py

- init (...)
- reset (...)
- step (...)
- get_obs (...)
- get_reward (...)
- update_metrics (...)
- save_history (...)

###### TimeMatrix.py

- init (...)
- search_next_event (...)
- update_martix (...)
- others

## Python packages to install
Install wandb to keep track of environment and agent metrics
```
pip install wandb
wandb login CREDENTIALS
```
Install gym to standardize the environment
```
pip install gym
```

Install stable-baselines to use state-of-art agents such as PPO, A2C and DQN
```
pip install stable-baselines3
```

## Things to try out
- Change the state
- Change the action
- Change the reward
- Change the observation epoch (currently the agent acts every single timestep, which means that the agent acts even when a zone state change event is executed)
