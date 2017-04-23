from Misc import State, ActionSorter_TopReturn


class ActionValueFunction:
    """
    This class implements the action-value function
    """
    def __init__(self):
        self.allStateData = {}

    def updateStateActionPair(self, episodeNum, state, action, returnVal):
        val = State(state)
        if episodeNum == 1:
            val.addAction(action, returnVal)
        else:
            if state in self.allStateData:
                val = self.allStateData[state]
            val.addAction(action, returnVal)
        self.allStateData[state] = val

    def getImprovedPolicy(self):
        results = {}
        for state in self.allStateData:
            if results.__len__() == 0 or state not in results.keys():
                results[state] = self.allStateData[state].getSortedActionsList()
        return results

    def getTotalNumberOfKnownStates(self, numberOfActions):
        retval = 0
        for key in self.allStateData.keys():
            if self.allStateData[key].getNumberOfActions() >= numberOfActions:
                retval += 1
        return retval

    def clear(self):
        self.allStateData.clear()

class ActionValueFunction_WithMemory:
    def __init__(self):
        self.allStateData = {}

    def updateStateActionPair(self, state, action, returnVal):
        if state not in self.allStateData:
            actions = ActionSorter_TopReturn()
            actions.update(action, returnVal)
            self.allStateData[state] = actions
        else:
            self.allStateData[state].update(action, returnVal)

    def finalizeStateUpdate(self):
        for state in self.allStateData:
            self.allStateData[state].updateBestActions()

    def getImprovedPolicy(self):
        return self.allStateData

    def getTotalNumberOfKnownStates(self, numberOfActions):
        retval = 0
        for state in self.allStateData:
            if self.allStateData[state].getNumberOfActions() >= numberOfActions:
                retval += 1
        return retval

    # def isNewState(self, stateToCheck):
    #     if self.allStateData.__len__() == 0:
    #         return True
    #     else:
    #         for state in self.allStateData.keys():
    #             if state == stateToCheck:
    #                 return False
    #         return True

    def clear(self):
        self.allStateData.clear()