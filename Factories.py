import gym

from Converters import StateConverterBase, StateConverterInf
from Policy import Policy_Base, Policy_WithMemory
from EnvironmentDataSet import EnvironmentDataSet

class Factory:
    """"""

    numEpisodesToSolve_CartPole0 = 100
    actionsSpaceSize_CartPole0 = 2
    thresholdValue_CartPole0 = 195

    numEpisodesToSolve_MountainCar0 = 100
    actionsSpaceSize_MountainCar0 = 3
    thresholdValue_MountainCar0 = -110

    numEpisodesToSolve_LunarLander2 = 100
    actionsSpaceSize_LunarLander2 = 4
    thresholdValue_LunarLander2 = 200

    @staticmethod
    def prepareData(envType, numSplit, infStep, explorCoeff):
        env = gym.make(envType);
        fileName = Factory.createFileName(envType, numSplit, infStep, explorCoeff)
        descr = Factory.createDescription(envType, numSplit, infStep, explorCoeff)
        if envType == 'CartPole-v0':
            policy = Policy_WithMemory(explorCoeff, Factory.actionsSpaceSize_CartPole0)
            # policy = Policy_Base(explorCoeff, Factory.actionsSpaceSize_CartPole0)
            policy.setActionsSpace(Factory.actionsSpaceSize_CartPole0)
            converter = StateConverterInf(env.observation_space.low, env.observation_space.high, infStep, numSplit)
            retval = EnvironmentDataSet(envType, Factory.thresholdValue_CartPole0, env, policy, converter,
                                        Factory.numEpisodesToSolve_CartPole0, fileName, descr)
        elif envType == 'MountainCar-v0':
            policy = Policy_WithMemory(explorCoeff, Factory.actionsSpaceSize_MountainCar0)
            # policy = Policy_Base(explorCoeff, Factory.actionsSpaceSize_MountainCar0)
            policy.setActionsSpace(Factory.actionsSpaceSize_MountainCar0)
            converter = StateConverterInf(env.observation_space.low, env.observation_space.high, infStep,
                                          numSplit)
            retval = EnvironmentDataSet(envType, Factory.thresholdValue_MountainCar0, env, policy, converter,
                                        Factory.numEpisodesToSolve_MountainCar0, fileName, descr)
        elif envType == 'LunarLander-v2':
            policy = Policy_Base(explorCoeff, Factory.actionsSpaceSize_LunarLander2)
            policy.setActionsSpace(Factory.actionsSpaceSize_LunarLander2)
            converter = StateConverterInf(env.observation_space.low, env.observation_space.high, infStep,
                                          numSplit)
            retval = EnvironmentDataSet(envType, Factory.thresholdValue_LunarLander2, env, policy, converter,
                                        Factory.numEpisodesToSolve_LunarLander2, fileName, descr)
        return retval

    @staticmethod
    def createFileName(envType, numSplit, infStep, explorCoeff):
        result = "{}_{}_{}_{}".format(envType, numSplit, infStep, explorCoeff)
        return result

    @staticmethod
    def createDescription(envType, numSplit, infStep, explorCoeff):
        result = "{}\t{}\t{}\t{}\n".format(envType, numSplit, infStep, explorCoeff)
        return result