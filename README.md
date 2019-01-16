# Eager in Haste
train.py --hybrid --vr --iter 4500


# nnpso
Training of Neural Network using Particle Swarm Optimization.

- Losses: [8.012271, 8.002567, 8.042032, 8.00327] Iteration: 0
- Losses: [6.176217, 4.361371, 8.0, 4.4884667] Iteration: 10000
- Losses: [1.7860475, 1.6976731, 4.1016097, 1.7039604] Iteration: 20000
- Losses: [1.6730804, 1.6676208, 1.702151, 1.6678984] Iteration: 30000
- Losses: [0.2826424, 1.6667058, 1.6677036, 1.0983028] Iteration: 40000
- Losses: [0.10317229, 1.6666698, 1.6667093, 0.16274434] Iteration: 50000
- Losses: [0.038965203, 1.6666677, 1.6666709, 0.062464874] Iteration: 60000
- Losses: [0.01462996, 1.6666673, 1.6666676, 0.023709508] Iteration: 70000
- Losses: [0.0054597864, 1.6666669, 1.6666669, 0.008893641] Iteration: 80000
- Losses: [0.002021174, 1.666667, 1.6666667, 0.003305968] Iteration: 90000

With Regular Neural Networks you just hope that you hadn't chosen the second network. If 20000 iterations took 20 days. Even after 20 days are you really sure that you got the best optimum loss and would further training improve network performance.

Thus we propose a new hybrid approach one that scales up crazy parallely. Particle Swarm Optimization with Neural Networks.

# Advantages
- Easy multi GPU training. More GPU's more training. Only the global best weights need to be synchronised. That can be done asynchronously on the CPU. Each GPU could be a single particle is case of huge networks or each GPU could instead be driving 100's of smaller particles
- Higher Learning Rates are Possible. ( > 100x when compared to traditional training). We literally trained with learning rate = 0.1. When's the last time you used that high a learning rate.
- Network is far less likely to get stuck on local optimum
- Faster on GPU depending upon how much GPU power you have and the network size.
- Better initialization. Use this for initialization and then use regular old gradient decent if network size is huge.

# Hyperparameter Help

 - GlobalBest Factor : The higher it is the more the particles stick together. Control Particles from going from too further. Based on my tests it is better to set Global Best Factor > Local Best Factor for  training purposes.
 - LocalBest Factor: The higher it is the less the particle will move towards the global best. And lesser chance of convergence. Set it to higher values if you wish to increase the exploration. Set a high value if using for initialization and not training
 - Velocity Decay: Velocity Decay is best between 0.75 -1. The decay prevents the network weights going too far away.
 - Velocity Max : Maximum velocity for a particle along a dimension. It useful to prevent network from seeing a pendulum effect.
 - Velocity Max Decay : It is a good idea to have the velocity max go down with iteration for finer control especially if you are not using the hybrid approach
 - Learning Rate: A much higher learning rate can be used with hybrid approach for learning weights.
 - Number of Particles : The more the merrier. As much as your GPU can support
 - Number of Iterations: Until you get bored.

# Training Neural Network only with PSO
Not that a great idea. Current testing based on fully connected networks, training with only PSO isn't a sufficient. Works great for Initializations. Increasing hidden layers however makes pso converge quicker. An increase in network size is compensated with lesser number of iterations.

# Training with Hybrid Approach
Way to go. Faster than traditional approaches if you have sufficient power.
Robust against Local Optimas

# Bragging Rights

With the 32 particles the sample network can be trained under 20K iteration with a final loss of 1.9013219e-06 using a Hybrid Approach. Oops did we mention the learning rate 0.1 (Ouch). And yes it never gets stuck up on local minimas. A much bigger search space is explored by the particles as compared to a single particle.

While using traditional approaches we get stuck up 50% of the times with a learning rate of 0.001. Even after 90000 iteration our losses for the best particle were at 0.002021174.

# Usage
Cite this work if used for academic purposes. For commercial purposes contact me preferably via mail.

# Project Guidelines

- Testing : Rigorous
- Comments : More the Merrier
- Modules : Split em up
- Variable Names : Big is OKay, Non-informative ain't
- Code Format: Python PEP 8

# Future Work
 - Trying out other PSO variants
 - Testing on MNIST
 - Adding support for more Layers

# Similar Projects
## Cublas Version of this Project
- [Local Best PSO without backprop](https://github.com/chintans111/ANNPSO)
- Non hybrid Approach. Regular old PSO
- Doesn't store activation of all Layers during Feed Forward

## Test on Iris bundled with QT code
- https://github.com/asbudhkar/ANN-PSO
- Some part of core code rewritten by author
- Tested on Iris

