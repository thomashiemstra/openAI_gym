import gym
import neat
from matplotlib import animation
import matplotlib.pyplot as plt
import visualize


env = gym.make('BipedalWalker-v2')
env.reset()

print("action space: {0!r}".format(env.action_space))
print("observation space: {0!r}".format(env.observation_space))


def do_random_stuff():
    env.reset()
    totalReward = 300
    action = []
    action.append(1)
    for i in range(100):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        totalReward += reward
        print(action)
    env.close()
         
def eval_genomes(genomes, config):
    for genome_id, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        genome.fitness = get_fitness(net)

def get_fitness(net):
    observation = env.reset()
    done = False
    totalReward = 300
    temp = []
#   done is true when frames > 200 so we stay below that
    for i in range(500):
#        env.render()
        action = net.activate(observation)
#        print(action)
        observation, reward, done, info = env.step(action)
        temp.append(reward)
#        if i > 100:
#            if np.absolute(temp[i] - temp[i-1]) <= 0.00001:
#                done = True
        totalReward += reward
        if done:
            totalReward -= 300
            break
    return totalReward

def run():
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
    winner = p.run(eval_genomes, 400)
    print('\nBest genome:\n{!s}'.format(winner)) 
    
    visualize.draw_net(config, winner, True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)    
    
    return winner, config
    
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

def save_gif(winner,config):

    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    
    input("Press Enter to continue...")
    
    frames = []
    
    observation = env.reset()
    for i in range(1000):
        frames.append(env.render(mode = 'rgb_array'))
        action = winner_net.activate(observation)
        observation, reward, done, info = env.step(action)
        print(reward)
        if done == True:
            break
    
    env.close()
    
    save_frames_as_gif(frames)
        
if __name__ == '__main__':
    winner, config = run()
    save_gif(winner,config)

    
    
    
