import operator
from math import inf, floor


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


class ValueNumPair:
    def __init__(self, value, number):
        self.value = value
        self.number = number

    def update(self, newVal):
        average = (self.value * self.number + newVal.value * newVal.number) / (self.number + newVal.number)
        self.value = average
        self.number += newVal.number