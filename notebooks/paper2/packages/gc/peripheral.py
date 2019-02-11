from .k3s import K3S

class Peripheral(K3S):


    def getPoints(self, totalToDisplay = 100, attribute = 'number_of_blocks'):
        thetaIncrement = 360 / totalToDisplay
        currentTheta = 0
        currentIndex = 0
        for word in self.wordInfo:
            self.wordInfo[word]['radius'] =  self.max - self.wordInfo[word][attribute]
            self.wordInfo[word]['topic'] = self.topics[self.wordInfo[word]['index']]
            self.wordInfo[word]['theta'] = currentTheta
            currentTheta += thetaIncrement
            self.wordInfo[word]['x'] = self.wordInfo[word]['radius'] * numpy.cos(numpy.deg2rad(self.wordInfo[word]['theta'))
            self.wordInfo[word]['y'] = self.wordInfo[word]['radius'] * numpy.sin(numpy.deg2rad(self.wordInfo[word]['theta'))
            
            currentIndex += 1
            if currentIndex < totalToDisplay:
                processedWordInfo.append(self.wordInfo[word])

        return processedWordInfo