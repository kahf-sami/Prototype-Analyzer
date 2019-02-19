from .k3s import K3S
import numpy

class Peripheral(K3S):


    def __init__(self, datasetProcessor):
        super().__init__(datasetProcessor)
        self.totalAngle = 360
        self.startAngle = 0
        return

    
    def setTotalAngle(self, angle):
        self.totalAngle = angle
        return


    def setStartAngle(self, start):
        self.startAngle = start
        return


    def getPoints(self, totalToDisplay = 100, attribute = 'number_of_blocks'):
        thetaIncrement = self.totalAngle / totalToDisplay
        currentTheta = self.startAngle
        currentIndex = 0
        processedWordInfo = []
        for word in self.vocab:
            if (self.topicFilter is not None) and (self.topicFilter != self.topics[word]):
                continue

            if currentIndex >= totalToDisplay:
                break

            self.vocab[word]['radius'] =  self.max - self.vocab[word][attribute]
            self.vocab[word]['topic'] = self.topics[word]
            self.vocab[word]['theta'] = currentTheta
            currentTheta += thetaIncrement
            self.vocab[word]['x'] = self.vocab[word]['radius'] * numpy.cos(numpy.deg2rad(self.vocab[word]['theta']))
            self.vocab[word]['y'] = self.vocab[word]['radius'] * numpy.sin(numpy.deg2rad(self.vocab[word]['theta']))
            
            currentIndex += 1
            
            processedWordInfo.append(self.vocab[word])

        self.startAngle = currentTheta
        return processedWordInfo