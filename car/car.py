import gym
import gym.wrappers
import numpy as np
import neat


env= gym.make('MountainCar-v0')
env.reset()
print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))
    
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
#        genome.fitness = 4.0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = get_fitness(net)
        
def do_random_stuff():
    env.reset()
    totalReward = 0
    action = []
    action.append(1)
    for i in range(201):
        env.render()
        action = 1#env.action_space.sample()
        observation, reward, done, info = env.step(action)
        totalReward += reward
        print(observation)
    env.close()

#this solution is stupid just try out random networks untill it reaches the top,
#no feedback other than wether or not it made it....
def get_fitness(net):
    observation = env.reset()
    done = False
    totalReward = 0
    frames = 0
#   done is true when frames > 200 so we stay below that
    for _ in range(198):
#        env.render()
        output = net.activate(observation)
        action = np.argmax(output)
#        print(action)
        observation, reward, done, info = env.step(action)
        frames += 1
        if done:
            totalReward = 200 - frames
            break
    return totalReward
    
    
# Load configuration.
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
#stats = neat.StatisticsReporter()
#p.add_reporter(stats)

# Run until a solution is found. 
winner = p.run(eval_genomes, 50)

print('\nBest genome:\n{!s}'.format(winner))

winner_net = neat.nn.FeedForwardNetwork.create(winner, config)

input("Press Enter to continue...")

observation = env.reset()
for i in range(1000):
    env.render()
    output = winner_net.activate(observation)
    action = np.argmax(output)
    observation, reward, done, info = env.step(action)
    if done == True:
        break

env.close()



