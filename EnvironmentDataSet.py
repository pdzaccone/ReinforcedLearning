from math import inf

import datetime

from EpisodeObserver import EpisodeObserver
from Functions import ActionValueFunction, ActionValueFunction_WithMemory


class EnvironmentDataSet:
    """"""

    def __init__(self, envName, envThresholdVal, environment, policy, stateConverter, numEpisodesToAverage, fileName, descr):
        self.environment = environment
        self.policy = policy
        self.converter = stateConverter
        self.avFunction = ActionValueFunction_WithMemory()
        self.episodeObserver = EpisodeObserver()
        self.rewards = []
        self.rewardTotal = 0
        self.name = envName
        self.thresholdValue = envThresholdVal
        # self.whenFirstSolved = -1
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

    # def firstTimeSolved(self):
    #     return self.whenFirstSolved

    def getNumberEpisodes(self):
        return self.rewards.__len__()

    def resetEnvironment(self):
        self.currentState = self.converter.convert(self.environment.reset())

    def render(self):
        self.environment.render()

    def makeStep(self):
        action = self.policy.makeStep(self.currentState)
        state, reward, done, info = self.environment.step(action)
        if self.policy.hasStateMoreUntestedActions(self.converter.convert(state)):
            rewardTmp = 100
        else:
            rewardTmp = reward
        self.episodeObserver.recordStep(self.currentState, action, rewardTmp)
        self.currentState = self.converter.convert(state)
        self.rewardTotal += reward
        # if self.rewardTotal >= self.thresholdValue and self.whenFirstSolved == -1:
        #     self.whenFirstSolved = self.rewards.__len__()
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
        return retval