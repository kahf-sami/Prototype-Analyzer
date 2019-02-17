from .k3s import K3S

class Linear(K3S):


    def getPoints(self, totalToDisplay = 100, attribute = 'number_of_blocks'):
        processedWordInfo = []
        currentIndex = 0
        for word in self.vocab:
            self.vocab[word]['y'] =  self.max - self.vocab[word][attribute]
            self.vocab[word]['topic'] = self.topics[word]
            self.vocab[word]['x'] = self.vocab[word]['index']
            
            currentIndex += 1
            if currentIndex <= totalToDisplay:
                processedWordInfo.append(self.vocab[word])

        return processedWordInfo