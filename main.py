from Factories import Factory

numSets = 1
numEpisodes = 500
numTimeSteps = 500

stepInf_CartPole0 = 0.5
numSplit_CartPole0 = 12
explorationCoeff_CartPole0 = 0.5

stepInf_MountainCar0 = 0.5
numSplit_MountainCar0 = 32
explorationCoeff_MountainCar0 = 1

stepInf_LunarLander2 = 0.5
numSplit_LunarLander2 = 8
explorationCoeff_LunarLander2 = 0.3

results = []
environments = []

#environments.append(Factory.prepareData('LunarLander-v2', numSplit_LunarLander2, stepInf_LunarLander2, explorationCoeff_LunarLander2))
environments.append(Factory.prepareData('MountainCar-v0', numSplit_MountainCar0, stepInf_MountainCar0, explorationCoeff_MountainCar0))
#environments.append(Factory.prepareData('CartPole-v0', numSplit_CartPole0, stepInf_CartPole0, explorationCoeff_CartPole0))

for environment in environments:
    fo = open(environment.generateFileName(), "a")
    fo.write(environment.generateDescription())
    fo.write("Episode\tReward\tNumSteps\t")
    for i in range(environment.getActionSpaceSize()):
        fo.write("StatesW_{}_Actions\t".format(i + 1))
    environment.clear()
    for i_episode in range(numEpisodes):
        environment.resetEnvironment()
        rew = 0
        stateData = []
        for t in range(numTimeSteps):
            environment.render()
            done = environment.makeStep()
            if done:
                rew = environment.update()
                fo.write("{}\t{}\t{}\t".format(i_episode, rew, t))
                for j in range(environment.getActionSpaceSize()):
                    stateData.append(environment.getNumKnownStates(j + 1))
                    fo.write("{}\t".format(environment.getNumKnownStates(j + 1)))
                fo.write("\n")
                if rew >= environment.getThresholdValue():
                    print("Environment {}, episode {} is solved in {} steps, total reward is {}".format(
                        environment.getName(), i_episode, t + 1, rew))
                else:
                    print("Environment {}, episode {} unsolved after {} steps, total reward is {}".format(
                        environment.getName(), i_episode, t + 1, rew))
                print("States: {}".format(stateData))
                break
        else:
            rew = environment.update()
            print("Environment {}, episode {} is solved in {} steps, total reward is {}".format(environment.getName(), i_episode, numTimeSteps, rew))
            print("States: {}".format(stateData))
        solved, rewAv = environment.isSolved()
        if solved:
            print("Environment {} is solved after {} episodes. Total reward is {}, average is {}".format(environment.getName(), i_episode, rew, rewAv))
            break
        else:
            print("Environment {} - average reward after {} episodes is {}".format(environment.getName(), i_episode, rewAv))

    fo.close()
# for env in environments:
#     print("Problem {}. Average reward for {} episodes - {}, first time solved at step {}".format(env.getName(), env.getNumberEpisodes(), env.getAverageReward(), env.firstTimeSolved()))