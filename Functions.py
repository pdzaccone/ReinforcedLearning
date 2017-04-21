from Misc import State


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