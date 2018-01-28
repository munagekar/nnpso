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

With Regular Neural Networks you just hope that you hadn't chosen the second network. If 20000 iterations took 20 days. Even after 20 days are you really sure that you got the best optimum loss and would # nnpso
Training of Neural Network using Particle Swarm Optimization.

# Hyperparameter Help

 - GlobalBest Factor : The higher it is the more the particles stick together. Control Particles from going from too further. Based on my tests it is better to set Global Best Factor > Local Best Factor
 - LocalBest Factor: The higher it is the less the particle will move towards the global best. And lesser chance of convergence. Set it to higher values if you wish to increase the exploration.
 - Velocity Decay: Velocity Decay is best between 0.7 -0.95. The decay prevents the network weights going too far away.

# Training Neural Network only with PSO
Not that a great idea. Current testing based on fully convolution networks, training with only PSO isn't a sufficient. 

# Huh, Then Why Should I Use it ?

 - Kicking off a network from Local Minima. Initialize one of the particles at the current minima the rest wherever you want. You might find a better point.
 - Network Initialization : This is a really good way to initialize your network. With a few iterations a very large number of points in huge dimensional space can be explored.
 - Multi-GPU training : If you have enought compute power and memory PSO could work really well. With 1000's networks moving like crazy in the search space you could get a really good set of weights and biases.

# Future Work

 - Experimentation with the Hyperparameters
 - Limiting the velocity
 - Tryping out other PSO variants
 - Exploring Hybird Approaches.

further training help. Enter ParticleFlow (Please Play Grand Red Carpet Welcome Music). Built on top of Tensorflow, Particle Flow does much more.

# Hyperparameter Help

 - GlobalBest Factor : The higher it is the more the particles stick together. Control Particles from going from too further. Based on my tests it is better to set Global Best Factor > Local Best Factor
 - LocalBest Factor: The higher it is the less the particle will move towards the global best. And lesser chance of convergence. Set it to higher values if you wish to increase the exploration.
 - Velocity Decay: Velocity Decay is best between 0.7 -0.95. The decay prevents the network weights going too far away.

# Training Neural Network only with PSO
Not that a great idea. Current testing based on fully connected networks, training with only PSO indicates that PSO alone is unfortunately not sufficient.  

# Huh, Then Why Should I Use it ?
 - Kicking off a network from Local Minima. Initialize one of the particles at the current minima the rest wherever you want. You might find a better point.
 - Network Initialization : This is a really good way to initialize your network. With a few iterations a very large number of points in huge dimensional space can be explored.
 - Multi-GPU training : If you have enought compute power and memory PSO could work really well. With 1000's networks moving like crazy in the search space you could get a really good set of weights and biases.

# Future Work
 - Experimentation with the Hyperparameters
 - Limiting the velocity
 - Tryping out other PSO variants
 - Exploring Hybird Approaches.

