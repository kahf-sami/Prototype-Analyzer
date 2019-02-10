import numpy
from . import Base
from sklearn.manifold import TSNE

class TSNELC(Base):


	def __init__(self, text, filterRate = 0.2):
		super().__init__(text, filterRate)
		self.loadSentences(text)
		self.perplexity = 3
		self.numberOfComponents = 2
		self.numberOfIterations = 250
		self.learnedEmbeddings = None
		return


	def setPerplexity(self, perplexity):
		self.perplexity = perplexity
		return


	def setNumberOfComponents(self, numberOfComponents):
		self.numberOfComponents = numberOfComponents
		return

	
	def setNumberOfIterations(self, numberOfIterations):
		self.numberOfIterations = numberOfIterations
		return


	def getWordCoOccurenceVectors(self):
		vocabSize = len(self.wordInfo)
		vectors = numpy.zeros((vocabSize, vocabSize))
		
		filteredWords = self.filteredWords.keys()

		for sentence in self.sentences:
			for word1 in sentence:
				word1 = self.stemmer.stem(word1.lower())
				if word1 not in filteredWords:
					continue
				for word2 in sentence:
					if (word1 == word2) or (word2 not in filteredWords):
						continue
					word2 = self.stemmer.stem(word2.lower())
					word1Index = self.wordInfo[word1]['index']
					word2Index = self.wordInfo[word2]['index']
					vectors[word1Index][word2Index] += 1

		return vectors


	def train(self):
		vectors = self.getWordCoOccurenceVectors()
		tsne = TSNE(perplexity = self.perplexity, 
			n_components = self.numberOfComponents, 
			init = 'pca', 
			n_iter = self.numberOfIterations, 
			method='exact')
		self.learnedEmbeddings = tsne.fit_transform(vectors)
		return


	def _getX(self, word):
		index = self.filteredWords[word]['index']
		return self.learnedEmbeddings[index, 0]


	def _getY(self, word):
		index = self.filteredWords[word]['index']
		return self.learnedEmbeddings[index, 1]