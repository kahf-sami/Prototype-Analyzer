from __future__ import division
import numpy as np
from .vocab import Vocab
from scipy.sparse import csr_matrix
import utility
from sklearn.decomposition import LatentDirichletAllocation

class LDA(Vocab):

	def __init__(self, datasetProcessor):
		super().__init__(datasetProcessor)
		self.iteration = 100
		self.verbose = 1
		self.perplexity = 10
		self.numberOfTopics = 10
		self.wordCoOccurenceVector = None
		self.topics = {}
		self._load()
		self.loadWordCoOccurenceVectorsFromFile()
		self.loadTopics()
		return

	def setNumberOfIterations(self, number):
		self.iteration = number
		return


	def setPerplexity(self, perplexity):
		self.perplexity = perplexity
		return


	def setVerbose(self, verbose):
		self.verbose = verbose
		return


	def loadWordCoOccurenceVectorsFromFile(self):
		self.wordCoOccurenceVector = self.__loadSparseCsr()
		print('-- loadWordCoOccurenceVectorsFromFile --')
		print(self.wordCoOccurenceVector)
		return self.wordCoOccurenceVector


	def loadTopics(self):
		self.__loadLdaTopics()
		return self.topics

	def buildWordCoOccurenceVectors(self):
		if not self.datasetProcessor:
			print('Failed to prepare word co-occurance matrix. Undefined dataset processor.')
			return

		vocabSize = len(self.vocab)
		print(vocabSize)
		self.wordCoOccurenceVector = np.zeros((vocabSize, vocabSize))
		print(self.processedSentences)
		if len(self.processedSentences) == 0:
			return

		for sentence in self.wordCoOccurenceVector:
			for word1Index in sentence:
				for word2Index in sentence:
					if word1Index == word2Index:
						continue
					else:
						self.wordCoOccurenceVector[word1Index][word2Index] += 1


		self.__convertToSparseMatrix()
		self.__saveSparseCsr(self.wordCoOccurenceVector)
		return self.wordCoOccurenceVector


	def train(self):
		print(self.wordCoOccurenceVector)
		lda = LatentDirichletAllocation(n_components=self.numberOfTopics, max_iter=self.iteration, learning_method='online', learning_offset=1.0,random_state=0).fit(self.wordCoOccurenceVector)
		wordScores = {}

		vocabId2Word = {}
		for word in self.vocab:
			vocabId2Word[self.vocab[word]['index']] = word

		self.topics = {}
		for topic_idx, topics in enumerate(lda.components_):
			for i in topics.argsort():
				word = vocabId2Word[i]
				if word in self.topics.keys():
					if self.topics[word] < topics[i]:
						self.topics[word] = topic_idx
					else:
						self.topics[word] = topic_idx

		
		self.__saveLdaTopics()
		return


	def __saveSparseCsr(self, vectors):
		path = self.datasetProcessor.getDatasetPath()
		filePath = utility.File.join(path, 'word_cooccurence.npz')
		file = utility.File(filePath)
		file.remove()
		np.savez(filePath, data=vectors.data, indices=vectors.indices, indptr=vectors.indptr, shape=vectors.shape)
		return


	def __saveLdaTopics(self):
		path = self.datasetProcessor.getDatasetPath()
		filePath = utility.File.join(path, 'lda.npz')
		file = utility.File(filePath)
		file.remove()
		np.savez(filePath, self.topics)
		return


	def __loadLdaTopics(self):
		path = self.datasetProcessor.getDatasetPath()
		filePath = utility.File.join(path, 'lda.npz')
		file = utility.File(filePath)
		if not file.exists():
			return
		self.topics = np.load(filePath)
		return


	def __loadSparseCsr(self):
		path = self.datasetProcessor.getDatasetPath()
		filePath = utility.File.join(path, 'word_cooccurence.npz')
		file = utility.File(filePath)
		if(not file.exists()):
			return None

		loader = np.load(filePath)
		if ((loader['shape'][0] != loader['shape'][1])):
			return None

		return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])


	def __convertToSparseMatrix(self):
		vocabSize = len(self.vocab)
		data = []
		rows = []
		columns = []
		for i in range(0, vocabSize):
			for j in range(0, vocabSize):
				if self.wordCoOccurenceVector[i][j] > 0:
					rows.append(i)
					columns.append(j)
					data.append(self.wordCoOccurenceVector[i][j])

		self.wordCoOccurenceVector = csr_matrix((data, (rows, columns)), shape=(vocabSize, vocabSize))
		return

