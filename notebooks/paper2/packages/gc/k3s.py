from .lda import LDA
from .plotter import Plotter

class K3S(LDA):


    def __init__(self, datasetProcessor):
        super().__init__(datasetProcessor)
        self.max = 0
        self.setMaxRadius()
        return


    def setMaxRadius(self, attribute = 'number_of_blocks'):
        for word in self.wordInfo:
            if self.max < self.wordInfo['word'][attribute]:
                self.max = self.wordInfo['word'][attribute]

        return


    def getPoints(self, totalToDisplay = 100, attribute = 'number_of_blocks'):
        thetaIncrement = 360 / totalToDisplay
        currentTheta = 0
        currentIndex = 0
        for word in self.wordInfo:
            self.wordInfo[word]['topic'] = self.topics[self.wordInfo[word]['index']]
            currentIndex += 1
            if currentIndex < totalToDisplay:
                processedWordInfo.append(self.wordInfo[word])

        return processedWordInfo