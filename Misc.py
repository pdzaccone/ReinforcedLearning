import operator
from enum import Enum
from math import inf, floor
from random import randrange

from numpy.matlib import rand


class State:
    def __init__(self, state: object) -> object:
        self.state = state
        self.actionsArray = {}

    def addAction(self, action, reward):
        if action in self.actionsArray.keys():
            self.actionsArray[action].update(reward)
        else:
            self.actionsArray[action] = reward

    def getBestAction(self):
        val = -inf
        act = -inf
        for key in self.actionsArray.keys():
            if self.actionsArray[key].value > val:
                val = self.actionsArray[key].value
                act = key
        return act

    def getSortedActionsList(self):
        result = []
        temp = {}
        for key in self.actionsArray.keys():
            temp[key] = self.actionsArray[key].value
        sortedActions = sorted(temp.items(), key=operator.itemgetter(1), reverse=True)
        for val in sortedActions:
            result.append(val[0])
        return result

    def getNumberOfActions(self):
        return self.actionsArray.__len__()

class ActionSorter_TopReturn:
    def __init__(self):
        self.actionsAll = {}
        self.actionsTop = []
        self.returnTop = -inf

    def update(self, action, returnVal):
        if action not in self.actionsAll:
            self.actionsAll[action] = (1, returnVal)
        else:
            self.actionsAll[action] = (self.actionsAll[action][0] + 1, (self.actionsAll[action][1] + returnVal) / self.actionsAll[action][0] + 1)

    def updateBestActions(self):
        self.actionsTop.clear()
        self.returnTop = -inf
        for action in self.actionsAll:
            self.returnTop = max(self.returnTop, self.actionsAll[action][1])
        for action in self.actionsAll:
            if self.actionsAll[action][1] == self.returnTop:
                self.actionsTop.append(action)

    def getBestAction(self):
        if self.actionsTop.__len__() == 1:
            return self.actionsTop[0]
        else:
            num = randrange(0, self.actionsTop.__len__())
            return self.actionsTop[num]

    def getAllActions(self):
        result = []
        for action in self.actionsAll:
            result.append(action)
        return result

    def hasUntestedActions(self, numberTotal):
        return self.actionsAll.__len__() < numberTotal

    def getNumberOfActions(self):
        return self.actionsAll.__len__()

    def getNumberOfVisits(self):
        retval = 0
        for action in self.actionsAll:
            retval += self.actionsAll[action][0]
        return retval

    def analyzeReturns(self):
        valAv = 0
        for action in self.actionsAll:
            valAv += self.actionsAll[action][1]
        return (self.actionsAll.__len__(), valAv / self.actionsAll.__len__(), self.returnTop)

class ValueNumPair:
    def __init__(self, value, number):
        self.value = value
        self.number = number

    def update(self, newVal):
        average = (self.value * self.number + newVal.value * newVal.number) / (self.number + newVal.number)
        self.value = average
        self.number += newVal.number

class RewardProcessingModes(Enum):
    RPM_NO_CHANGE = 1
    RPM_NEW_STATES_PLUS = 2
    RPM_RARE_STATES_PLUS = 3