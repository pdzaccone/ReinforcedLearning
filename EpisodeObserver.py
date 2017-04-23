class EpisodeObserver:
    """
    This class stores episode states and rewards and optimizes the policy
    """
    def __init__(self):
        self.episodeNum = 1
        self.totalReward = 0
        self.returns = {}

    def recordStep(self, state, action, reward):
        self.totalReward += reward
        if self.returns.__len__() == 0 or state not in self.returns.keys():
            self.returns[state] = (action, self.totalReward)

    def updateStates(self, avFunction):
        if self.returns.__len__() != 0:
            for state in self.returns.keys():
                self.returns[state] = (self.returns[state][0], self.totalReward - self.returns[state][1] + 1)
                action = self.returns[state][0]
                avFunction.updateStateActionPair(state, action, self.returns[state][1])
                # avFunction.updateStateActionPair(self.episodeNum, state, action, ValueNumPair(self.returns[state][1], 1))
            avFunction.finalizeStateUpdate()
        return avFunction

    def reset(self):
        self.episodeNum += 1
        self.totalReward = 0
        self.returns.clear()

    def clear(self):
        self.reset()
        self.episodeNum = 1