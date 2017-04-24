from math import inf

import datetime

from EpisodeObservers import EpisodeObserver
from Functions import ActionValueFunction, ActionValueFunction_WithMemory
from Misc import RewardProcessingModes


class EnvironmentDataSet_Base:
    """"""

    def __init__(self, envName, envThresholdVal, environment, policy, stateConverter, numEpisodesToAverage, fileName, descr):
        self.solved = False
        self.environment = environment
        self.policy = policy
        self.converter = stateConverter
        self.avFunction = ActionValueFunction_WithMemory()
        self.episodeObserver = EpisodeObserver()
        self.rewards = []
        self.rewardTotal = 0
        self.name = envName
        self.thresholdValue = envThresholdVal
        self.numberOfEpisodesToAverage = numEpisodesToAverage

        dateTime = datetime.datetime.now().strftime("%Y_%B_%d_%I_%M")
        self.fileName = "{}_{}.txt".format(fileName, dateTime)
        self.description = descr

    def generateFileName(self):
        return self.fileName

    def generateDescription(self):
        return self.description

    def getActionSpaceSize(self):
        return self.policy.getActionSpaceSize()

    def getNumKnownStates(self, power):
        return self.avFunction.getTotalNumberOfKnownStates(power)

    def clear(self):
        self.policy.clear()
        self.avFunction.clear()
        self.episodeObserver.clear()

    def getMaxReward(self):
        res = -inf
        for val in self.rewards:
            res = max(res, val)
        return res

    def getThresholdValue(self):
        return self.thresholdValue

    def getName(self):
        return self.name

    def informAboutPrematureFinish(self):
        self.solved = True

    def isSolvedNow(self):
        return self.solved

    def isSolved(self):
        num = max(0, self.rewards.__len__() - self.numberOfEpisodesToAverage)
        val = self.getAverageReward(num)
        return (val >= self.thresholdValue, val)

    def getAverageReward(self, num):
        res = 0
        for index in range(num, self.rewards.__len__()):
            res += self.rewards[index]
        res /= (self.rewards.__len__() - num)
        return res

    def getNumberEpisodes(self):
        return self.rewards.__len__()

    def resetEnvironment(self):
        self.currentState = self.converter.convert(self.environment.reset())

    def render(self):
        self.environment.render()

    def makeStep(self):
        action = self.policy.makeStep(self.currentState)
        state, reward, done, info = self.environment.step(action)
        self.episodeObserver.recordStep(self.currentState, action, reward)
        self.currentState = self.converter.convert(state)
        self.rewardTotal += reward
        return done

    def update(self):
        retval = self.rewardTotal
        self.rewards.append(self.rewardTotal)
        self.rewardTotal = 0
        self.avFunction = self.episodeObserver.updateStates(self.avFunction)
        self.policy.updatePolicy(self.avFunction.getImprovedPolicy())
        self.episodeObserver.reset()
        self.solved = False
        return retval


class EnvironmentDataSet_2Stages(EnvironmentDataSet_Base):
    def __init__(self, envName, envThresholdVal, environment, rewardProcesingMode, policy, stateConverter, numEpisodesToAverage, fileName, descr):
        super().__init__(envName, envThresholdVal, environment, policy, stateConverter, numEpisodesToAverage, fileName, descr)
        self.rewardProcessingMode = rewardProcesingMode

    def informAboutPrematureFinish(self):
        self.solved = True
        self.rewardProcessingMode = RewardProcessingModes.RPM_NO_CHANGE

    def makeStep(self):
        action = self.policy.makeStep(self.currentState)
        state, reward, done, info = self.environment.step(action)
        rewardTmp = self.processReward(self.currentState, self.converter.convert(state), reward)
        self.episodeObserver.recordStep(self.currentState, action, rewardTmp)
        self.currentState = self.converter.convert(state)
        self.rewardTotal += reward
        return done

    def update(self):
        retval = self.rewardTotal
        self.rewards.append(self.rewardTotal)
        self.rewardTotal = 0
        self.avFunction = self.episodeObserver.updateStates(self.avFunction)
        self.policy.updatePolicy(self.avFunction.getImprovedPolicy())
        self.episodeObserver.reset()
        av, ma = self.policy.analyzeReturns()
        print("Policy with {} states, average return - {}, max return is {}".format(self.policy.getSize(), av, ma))
        self.solved = False
        return retval

    def processReward(self, stateOld, stateNew, reward):
        retval = reward
        if self.rewardProcessingMode == RewardProcessingModes.RPM_NO_CHANGE:
            retval = reward
        elif self.rewardProcessingMode == RewardProcessingModes.RPM_NEW_STATES_PLUS:
            if self.policy.hasStateMoreUntestedActions(stateNew):
                retval = abs(reward) * 5
            else:
                retval = reward
        elif self.rewardProcessingMode == RewardProcessingModes.RPM_RARE_STATES_PLUS:
            numNew = self.policy.getNumberOfVisits(stateNew)
            numOld = self.policy.getNumberOfVisits(stateOld)
            if numNew < numOld:
                retval = abs(reward) * 10
            else:
                retval = reward
        return retval