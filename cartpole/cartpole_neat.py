import gym
import gym.wrappers
import numpy as np
import neat
import visualize

env= gym.make('CartPole-v1')
env.reset()
print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))
    
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
#        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = get_fitness(net)
        
        

def get_fitness(net):
    observation = env.reset()
    done = False
    runs = 0
    totalReward = 0
    while not done:
        runs += 1
#        env.render()
        output = net.activate(observation)
        action = np.argmax(output)
        observation, reward, done, info = env.step(int(action))
        totalReward += reward
    return totalReward
    
    


# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run until a solution is found. 
winner = p.run(eval_genomes)

print('\nBest genome:\n{!s}'.format(winner))

winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

visualize.draw_net(config, winner, False)

observation = env.reset()
for i in range(1000):
    env.render()
    output = winner_net.activate(observation)
    action = np.argmax(output)
    observation, reward, done, info = env.step(int(action))
    if done == True:
        break

env.close()



