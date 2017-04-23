from random import randrange

from gym.spaces import Discrete
from numpy.matlib import rand, inf


class Policy_Base:
    """
    Base class for all policies
    
    Parameters
    ----------
    actionsMap : Dictionary
                 This map decides which actions should be taken at each known state
                 Keys are states (lists of values), values are actions (integers)
                 
    actionSpace : Discrete
                  Space class, defining allowed actions
                  
    explorationCoeff : float
                       Exploration coefficient - how likely it is that the policy veers into the great unknown [0.0 - 1.0)
    """

    def __init__(self, exploration, actionSpaceSize):
        """
        Constructor
        :param exploration: float
                            exploration coefficient (between 0.0 and 1.0)
        """
        self.actionsMap = {}
        self.setExplorationCoefficient(exploration)
        self.setActionsSpace(actionSpaceSize)

    def updatePolicy(self, statesActionsMap):
        """
        This method sets a policy
        :param statesActionsMap: Dictionary
                                 This map decides which actions should be taken at each known state
                                 Keys are states (lists of values), values are actions (integers)
        :return: 
        """
        self.actionsMap = statesActionsMap

    def setActionsSpace(self, val):
        """
        This method specifies an action space
        :param val: integer
                    Number of actions, used to create Discrete(val)
        :return: 
        """
        self.actionSpaceSize = val
        self.actionSpace = Discrete(val)

    def getActionSpaceSize(self):
        return self.actionSpaceSize

    def setExplorationCoefficient(self, exploration):
        """
        This method specifies exploration coefficient
        :param exploration: float
                            Exploration coefficient - how likely it is that the policy veers into the great unknown [0.0 - 1.0)
        :return: 
        """
        self.explorationCoeff = exploration

    def makeStep(self, observation):
        """
        In this method the policy applies its internal logic to information about the current state to choose an action to take
        :param observation: List
                            List of values, defining current state
        :return: integer
                 Resulting action according to the policy 
        """
        if self.actionsMap.__len__() != 0 and observation in self.actionsMap.keys():
            return self.calculateAction(observation)
        else:
            return self.actionSpace.sample()

    def calculateAction(self, observation):
        """
        This internal method decides whether to choose the already existing "optimal" action or to explore
        :param observation: List
                            List of values, defining current state                            
        :return: integer
                 Resulting action according to the policy 
        """
        action = self.actionsMap[observation][0]
        result = action
        if self.actionsMap[observation].__len__() < self.actionSpaceSize:
            val = rand(1)
            if self.explorationCoeff != 0 and val < self.explorationCoeff:
                while result == action: result = self.actionSpace.sample()
        return result

    def clear(self):
        self.actionsMap.clear()

class Policy_WithMemory:
    """

    Parameters
    ----------
    actionsMap : Dictionary
                 This map decides which actions should be taken at each known state
                 Keys are states (lists of values), values are actions (integers)

    actionSpace : Discrete
                  Space class, defining allowed actions

    explorationCoeff : float
                       Exploration coefficient - how likely it is that the policy veers into the great unknown [0.0 - 1.0)
    """

    def __init__(self, exploration, actionSpaceSize):
        """
        Constructor
        :param exploration: float
                            exploration coefficient (between 0.0 and 1.0)
        """
        self.actionsMap = {}
        self.setExplorationCoefficient(exploration)
        self.setActionsSpace(actionSpaceSize)

    def updatePolicy(self, statesActionsMap):
        """
        This method sets a policy
        :param statesActionsMap: Dictionary
                                 This map decides which actions should be taken at each known state
                                 Keys are states (lists of values), values are actions (integers)
        :return: 
        """
        self.actionsMap = statesActionsMap

    def setActionsSpace(self, val):
        """
        This method specifies an action space
        :param val: integer
                    Number of actions, used to create Discrete(val)
        :return: 
        """
        self.actionSpaceSize = val
        self.actionSpace = Discrete(val)

    def getActionSpaceSize(self):
        return self.actionSpaceSize

    def setExplorationCoefficient(self, exploration):
        """
        This method specifies exploration coefficient
        :param exploration: float
                            Exploration coefficient - how likely it is that the policy veers into the great unknown [0.0 - 1.0)
        :return: 
        """
        self.explorationCoeff = exploration

    def makeStep(self, observation):
        """
        In this method the policy applies its internal logic to information about the current state to choose an action to take
        :param observation: List
                            List of values, defining current state
        :return: integer
                 Resulting action according to the policy 
        """
        if self.actionsMap.__len__() != 0 and observation in self.actionsMap.keys():
            return self.calculateAction(observation)
        else:
            return self.actionSpace.sample()

    def calculateAction(self, observation):
        """
        This internal method decides whether to choose the already existing "optimal" action or to explore
        :param observation: List
                            List of values, defining current state                            
        :return: integer
                 Resulting action according to the policy 
        """

        actionPicker = self.actionsMap[observation]
        result = 0
        if actionPicker.hasUntestedActions(self.actionSpaceSize):
            val = rand(1)
            if self.explorationCoeff != 0 and val < self.explorationCoeff:
                tested = actionPicker.getAllActions()
                result = tested[0]
                while result in tested: result = self.actionSpace.sample()
        else:
            result = actionPicker.getBestAction()

        # action = self.actionsMap[observation][0]
        # result = action
        # if self.actionsMap[observation].__len__() < self.actionSpaceSize:
        #     val = rand(1)
        #     if self.explorationCoeff != 0 and val < self.explorationCoeff:
        #         while result == action: result = self.actionSpace.sample()
        return result

    def hasStateMoreUntestedActions(self, state):
        if self.actionsMap.__len__() == 0:
            return True
        else:
            if state not in self.actionsMap:
                return True
            else:
                actionPicker = self.actionsMap[state]
                if actionPicker.hasUntestedActions(self.actionSpaceSize):
                    return True
        return False

    def clear(self):
        self.actionsMap.clear()

    def getSize(self):
        return self.actionsMap.__len__()

    def analyzeReturns(self):
        numTotal = 0
        valAv = 0
        valMax = -inf
        for state in self.actionsMap:
            num, valAvTmp, valMaxTmp = self.actionsMap[state].analyzeReturns()
            numTotal += num
            valAv += valAvTmp
            valMax = max(valMax, valMaxTmp)
        valAv /= numTotal
        return (valAv, valMax)