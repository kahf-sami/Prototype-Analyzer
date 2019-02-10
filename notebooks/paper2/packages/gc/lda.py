from .common import Common
import numpy, sys, os
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.datasets import fetch_20newsgroups
from sklearn.decomposition import NMF, LatentDirichletAllocation
import iterators, utility
from scipy.sparse import csr_matrix

class LDA(Common):


	def __init__(self):
		super().__init__()
		config = utility.Config()
		self.filePath = utility.File.join('/home/apache/hosts/WSBprojects/SP-PySystem', 'data', config.getProjectId())

		directory = utility.Directory(self.filePath)
		if not directory.exists():
			directory.create()
		self.fileLabelPath = ''
		self.lda = None
		self.labels = []
		return


	def getTermTermVectors(self):
		filePath = utility.File.join(self.filePath, 'lda.npz')
		self.fileLabelPath = utility.File.join(self.filePath, 'label-lda.npz')

		wordIds = self.lCStorer.getAllImportantIds()
		total = len(wordIds)
		savedVectors = self.loadSparseCsr(filePath, total)		

		if str(type(savedVectors)) == "<class 'scipy.sparse.csr.csr_matrix'>":
			return savedVectors

		cursor = self.wordStorer.getWordsByBatch('wordid,word', True)

		data = []
		rows = []
		columns = []
		self.labels = []
		rowIndex = 0
		
		for word in cursor:
			columnIndex = 0
			for otherWord in wordIds:
				totalRelated = self.lCStorer.howManyTimes(word[0], otherWord[0])
				if totalRelated and totalRelated[0]:
					columns.append(columnIndex)
					rows.append(rowIndex)
					data.append(totalRelated[0])
				columnIndex = columnIndex + 1
			
			self.labels.append(word[0])
			rowIndex = rowIndex + 1
		
		vectors = csr_matrix((data, (rows, columns)), shape=(total, total))
		self.saveSparseCsr(filePath, vectors)
		numpy.savez(self.fileLabelPath, self.labels)
		return vectors


	def getTermDocument(self):
		filePath = utility.File.join(self.filePath, 'td-lda.npz')
		self.fileLabelPath = utility.File.join(self.filePath, 'label-td-lda.npz')
		
		contentIds = self.contentStorer.getAllIds()
		total = len(contentIds)
		
		savedVectors = self.loadSparseCsr(filePath, total)
		
		if str(type(savedVectors)) == "<class 'scipy.sparse.csr.csr_matrix'>":
			return savedVectors

		data = []
		rows = []
		columns = []
		rowIndex = 0
		indexes = {}
		newIndex = 0
		self.labels = []
		for contentId in contentIds:
			words = self.lCStorer.getWordsInAContent(contentId[0])
			for word in words:
				if word[0] in indexes.keys():
					columns.append(indexes[word[0]])
				else:
					self.labels.append(word[0])
					indexes[word[0]] = newIndex
					columns.append(newIndex)
					newIndex = newIndex + 1

				rows.append(rowIndex)
				data.append(1)
			
			rowIndex = rowIndex + 1 

		vectors = csr_matrix((data, (rows, columns)), shape=(total, newIndex))
		self.saveSparseCsr(filePath, vectors)
		numpy.savez(self.fileLabelPath, self.labels)
		return vectors
 

	def loadLabels(self):
		self.labels = numpy.load(self.fileLabelPath)
		return


	def process(self, useTermTermVector = False):
		vectors = None
		if useTermTermVector:
			vectors = self.getTermTermVectors()
		else:
			vectors = self.getTermDocument()

		if not self.labels:
			self.loadLabels()

		self.lda = LatentDirichletAllocation(n_components=100, max_iter=500, learning_method='online', learning_offset=5.,random_state=0).fit(vectors)
		wordScores = {}

		for topic_idx, topics in enumerate(self.lda.components_):
			for i in topics.argsort():
				if i in wordScores.keys():
					if wordScores[i]['score'] < topics[i]:
						wordScores[i]['score'] = topics[i]
						wordScores[i]['topic'] = topic_idx
				else:
					wordScores[i] = {}
					wordScores[i]['score'] = topics[i]
					wordScores[i]['topic'] = topic_idx
					wordScores[i]['word'] = self.labels[i]

		topicWords = {}
		for i in wordScores.keys():
			if wordScores[i]['topic'] not in topicWords.keys():
				topicWords[wordScores[i]['topic']] = []
			topicWords[wordScores[i]['topic']].append(wordScores[i]['word'])
			data = {}
			data['wordid'] = self.labels[i]
			data['color'] = wordScores[i]['topic']

			self.wordStorer.save(data, False)

		
		for topicIndex in topicWords:
			print(topicWords[topicIndex])
			print('-----------------------------------------------')

		print('Total Clusters: ' + str(len(topicWords)))
		
		return

	def saveSparseCsr(self, filename, array):
		numpy.savez(filename, data=array.data, indices=array.indices, indptr=array.indptr, shape=array.shape)

		
	def loadSparseCsr(self, filename, total):
		file = utility.File(filename)
		if(not file.exists()):
			return None


		loader = numpy.load(filename)
		if ((loader['shape'][0] != loader['shape'][1]) or (loader['shape'][0] != total)):
			return None

		
		return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])



