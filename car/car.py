import gym
import gym.wrappers
import numpy as np
import neat
from JSAnimation.IPython_display import display_animation
from matplotlib import animation
import matplotlib.pyplot as plt
from IPython.display import display



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

do_random_stuff()
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

frames = []

observation = env.reset()
for i in range(1000):
    frames.append(env.render(mode = 'rgb_array'))
    output = winner_net.activate(observation)
    action = np.argmax(output)
    observation, reward, done, info = env.step(action)
    if done == True:
        break

env.close()

def save_frames_as_gif(frames):
    """
    Displays a list of frames as a gif, with controls
    """
    plt.figure(figsize=(frames[0].shape[1] / 72.0, frames[0].shape[0] / 72.0), dpi = 72)
    patch = plt.imshow(frames[0])
    plt.axis('off')

    def animate(i):
        patch.set_data(frames[i])

    anim = animation.FuncAnimation(plt.gcf(), animate, frames = len(frames), interval=50)
#    display(display_animation(anim, default_mode='loop'))
    anim.save('animation.gif', writer='imagemagick', fps=60)

save_frames_as_gif(frames)

