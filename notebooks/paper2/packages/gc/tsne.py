from .lda import LDA

class TSNE(LDA):


    def __init__(self, dataProcessor):
		super().__init__(dataProcessor)
		self.perplexity = 3
		self.numberOfComponents = 2
		self.numberOfIterations = 250
		self.learnedEmbeddings = None
		return


    def train(self):
		vectors = self.wordCoOccurenceVector
		tsne = TSNE(perplexity = self.perplexity, 
			n_components = self.numberOfComponents, 
			init = 'pca', 
			n_iter = self.numberOfIterations, 
			method='exact')
		self.learnedEmbeddings = tsne.fit_transform(vectors)
		return


    def getPoints(self, totalToDisplay = 100, attribute = ''):

        currentIndex = 0
        for word in self.wordInfo:
            index = self.wordInfo[word]['index']
            self.wordInfo[word]['topic'] = self.topics[self.wordInfo[word]['index']]
            self.wordInfo[word]['x'] = self.learnedEmbeddings[index, 0]
            self.wordInfo[word]['x'] = self.learnedEmbeddings[index, 1]
            
            currentIndex += 1
            if currentIndex < totalToDisplay:
                processedWordInfo.append(self.wordInfo[word])

        return processedWordInfo