from math import floor


class StateConverterBase:
    coeff = 1000

    elementsMins = []
    elementsMaxs = []
    infMin = 0
    infMax = 0
    steps = []
    numSteps = 0

    def __init__(self, mins, maxs, stepDef, numSteps):
        self.infMin = -stepDef * (numSteps - 2) / 2
        self.infMax = stepDef * (numSteps - 2) / 2
        for index, val in enumerate(mins):
            if mins[index] < -StateConverterBase.coeff and maxs[index] > StateConverterBase.coeff:
                self.elementsMins.append(-StateConverterBase.coeff)
                self.elementsMaxs.append(StateConverterBase.coeff)
                self.steps.append(stepDef)
            else:
                self.elementsMins.append(mins[index])
                self.elementsMaxs.append(maxs[index])
                self.steps.append((maxs[index] - mins[index]) / numSteps)
        self.numSteps = numSteps

    def convert(self, data):
        resultList = []
        for i, val in enumerate(data):
            if self.elementsMins[i] == -StateConverterBase.coeff:
                if val < self.infMin:
                    num = 0
                elif val > self.infMax:
                    num = self.numSteps - 1
                else:
                    num = (val + self.steps[i] - self.infMin) / self.steps[i];
            else:
                num = (val - self.elementsMins[i]) / self.steps[i]
            resultList.append(floor(num))
        return tuple(resultList)


class StateConverterInf:
    coeff = 1000

    def __init__(self, mins, maxs, stepDef, numSteps):
        self.elementsMins = []
        self.elementsMaxs = []
        self.infMin = 0
        self.infMax = 0
        self.steps = []
        self.numSteps = 0

        self.infMin = -stepDef * (numSteps - 2) / 2
        self.infMax = stepDef * (numSteps - 2) / 2
        for index, val in enumerate(mins):
            self.elementsMins.append(mins[index])
            self.elementsMaxs.append(maxs[index])
            if mins[index] < -StateConverterBase.coeff and maxs[index] > StateConverterBase.coeff:
                self.steps.append(stepDef)
            else:
                self.steps.append((maxs[index] - mins[index]) / numSteps)
        self.numSteps = numSteps

    def convert(self, data):
        resultList = []
        for i, val in enumerate(data):
            if self.elementsMins[i] < -StateConverterBase.coeff:
                num = floor(val / self.steps[i])
                if val < 0:
                    num = num * 2 + 1
                else:
                    num = num * 2
            else:
                num = (val - self.elementsMins[i]) / self.steps[i]
            resultList.append(floor(num))
        return tuple(resultList)
