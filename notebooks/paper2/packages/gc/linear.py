from .k3s import K3S
import time

class Linear(K3S):


    def __init__(self, datasetProcessor):
        super().__init__(datasetProcessor)
        self.startX = 0
        self.now = time.time()
        return


    def setStartX(self, start):
        self.startX = start
        return


    def getPoints(self, totalToDisplay = 100, attribute = 'number_of_blocks'):
        processedWordInfo = []
        currentIndex = 0
        lastX = 0

        for word in self.vocab:
            if (self.topicFilter is not None) and (self.topicFilter != self.topics[word]):
                continue

            if currentIndex >= totalToDisplay:
                break

            self.vocab[word]['y'] =  self.vocab[word][attribute]
            self.vocab[word]['topic'] = self.topics[word]
            self.vocab[word]['x'] = self.vocab[word]['index']
            
            currentIndex += 1
            
            processedWordInfo.append(self.vocab[word])

        return processedWordInfo