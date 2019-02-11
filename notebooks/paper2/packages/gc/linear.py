from .k3s import K3S

class Linear(K3S):


    def getPoints(self, totalToDisplay = 100, attribute = 'number_of_blocks'):

        currentIndex = 0
        for word in self.wordInfo:
            self.wordInfo[word]['y'] =  self.max - self.wordInfo[word][attribute]
            self.wordInfo[word]['topic'] = self.topics[self.wordInfo[word]['index']]
            self.wordInfo[word]['x'] = self.wordInfo[word]['timestamp']
            
            currentIndex += 1
            if currentIndex < totalToDisplay:
                processedWordInfo.append(self.wordInfo[word])

        return processedWordInfo