from .k3s import K3S
import numpy

class Peripheral(K3S):


    def getPoints(self, totalToDisplay = 100, attribute = 'number_of_blocks'):
        thetaIncrement = 360 / totalToDisplay
        currentTheta = 0
        currentIndex = 0
        processedWordInfo = []
        for word in self.vocab:
            self.vocab[word]['radius'] =  self.max - self.vocab[word][attribute]
            self.vocab[word]['topic'] = self.topics[word]
            self.vocab[word]['theta'] = currentTheta
            currentTheta += thetaIncrement
            self.vocab[word]['x'] = self.vocab[word]['radius'] * numpy.cos(numpy.deg2rad(self.vocab[word]['theta']))
            self.vocab[word]['y'] = self.vocab[word]['radius'] * numpy.sin(numpy.deg2rad(self.vocab[word]['theta']))
            
            currentIndex += 1
            if currentIndex <= totalToDisplay:
                processedWordInfo.append(self.vocab[word])

        return processedWordInfo