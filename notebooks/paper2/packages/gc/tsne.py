from .lda import LDA
from sklearn.manifold import TSNE as skTSNE
import utility
import numpy as np

class TSNE(LDA):


	def __init__(self, dataProcessor):
		super().__init__(dataProcessor)
		self.perplexity = 10
		self.numberOfTopics = 10
		self.iteration = 250
		self.learnedEmbeddings = None
		return


	def train(self):
		vectors = self.wordCoOccurenceVector.todense()
		tsne = skTSNE(perplexity=self.perplexity, 
			n_components=self.numberOfTopics, 
			init='pca', 
			n_iter=self.iteration, 
			method='exact')
		self.learnedEmbeddings = tsne.fit_transform(vectors)
		self.__saveTSNE()
		print('Trained for TSNE')
		return


	def getPoints(self, totalToDisplay = 100, attribute = ''):
		currentIndex = 0
		processedWordInfo = []
		for word in self.vocab:
			if (self.topicFilter is not None) and (self.topicFilter != self.topics[word]):
				continue

			if currentIndex >= totalToDisplay:
				break

			index = self.vocab[word]['index']
			self.vocab[word]['topic'] = self.topics[word]
			self.vocab[word]['x'] = self.learnedEmbeddings[index, 0]
			self.vocab[word]['y'] = self.learnedEmbeddings[index, 1]

			currentIndex += 1
					
			processedWordInfo.append(self.vocab[word])

		return processedWordInfo

		
	def _load(self):
		super()._load()
		self.__loadTSNE()
		return

	def __saveTSNE(self):
		self._saveNumpy('tsne.npz', self.learnedEmbeddings)
		return


	def __loadTSNE(self):
		embeddingFromFile = self._loadNumpy('tsne.npz')
		self.learnedEmbeddings = None
		if embeddingFromFile is None:
			return

		for fileRef in embeddingFromFile:
			self.learnedEmbeddings =  embeddingFromFile[fileRef]

		return