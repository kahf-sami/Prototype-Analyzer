from .lda import LDA
import numpy, sys
from sklearn.manifold import TSNE
import utility
from scipy.sparse import csr_matrix

class gcTSNE(LDA):


	def __init__(self):
		super().__init__()
		self.iteration = 250
		self.nComponents = 2
		self.verbose = 1
		self.perplexity = 20
		#self.graphTypes = ['compact', 'term']
		self.graphTypes = ['term']
		config = utility.Config()
		self.filePath = utility.File.join('/home/apache/hosts/WSBprojects/SP-PySystem', 'data', config.getProjectId())

		directory = utility.Directory(self.filePath)
		if not directory.exists():
			directory.create()

		self.fileLabelPath = ''
		return


	def storePointsPerGraphType(self):
		for graphType in self.graphTypes:
			self.wordPointProcessor.deleteAllByGraphType(graphType)
			self.storeWordPointsforGraphType(graphType)
		return


	def storePointsPerGraphType(self):
		for graphType in self.graphTypes:
			self.wordPointProcessor.deleteAllByGraphType(graphType)
			
		if 'compact' in self.graphTypes:
			self.storeWordPointsforCompact()

		if 'term' in self.graphTypes:
			self.storeWordPointsTermDocument()
		return

	'''	
	def storeWordPointsTermDocument(self):
		graphType = 'compact'
		wordIds = self.lCStorer.getAllImportantIds()
		print(len(wordIds))
		cursor = self.wordStorer.getWordsByBatch('wordid,word,color', True)

		data = []
		labels = []
		colors = []
		
		for word in cursor:
			print()
			row = []
			for otherWord in wordIds:
				totalRelated = self.lCStorer.howManyTimes(word[0], otherWord[0])
				row.append(totalRelated[0])
				

			data.append(row)
			labels.append(word[1])
			colors.append(word[2])

		self.analyzeAndSaveVectors(data, labels, colors, 'term')
		return
	'''

	def storeWordPointsTermDocument(self):
		filePath = utility.File.join(self.filePath, 'td-tsne.npz')
		fileLabelPath = utility.File.join(self.filePath, 'label-td-tsne.npz')
		colorFilePath = utility.File.join(self.filePath, 'color-td-tsne.npz')

		contentIds = self.contentStorer.getAllIds()
		total = len(contentIds)
		
		#savedVectors = self.loadSparseCsr(filePath, total)
		print(filePath + '-------')
		file = utility.File(filePath)
		'''
		if(file.exists()):
			print('---------')
			data = numpy.load(filePath)
			colors = numpy.load(colorFilePath)
			labels = numpy.load(fileLabelPath)
			self.analyzeAndSaveVectors(data, labels, colors, 'term')
			return
		'''
		print('--------999999999-----------')
		wordIds = self.lCStorer.getAllImportantIds()
		contentIds = self.lCStorer.getAllContentIds(100)

		data = []
		colors = []
		labels = []

		index = 0;
		for wordId in wordIds:
			print(wordId)
			row = []
			word = self.wordStorer.getDetailsById(wordId[0])

			for contentId in contentIds:
				if self.lCStorer.contentHasWord(contentId[0], wordId[0]):
					row.append(1)
				else:
					row.append(0)

			data.append(row)
			labels.append(word[0][1])
			colors.append(word[0][2])
			print(row)
			print(index)
			index += 1


			print(labels)
			numpy.savez(filePath, data)
			numpy.savez(fileLabelPath, labels)
			numpy.savez(colorFilePath, colors)

		print(data)
		self.analyzeAndSaveVectors(data, labels, colors, 'term')
		return data
 


	def storeWordPointsforCompact(self):
		cursor = self.wordStorer.getWordsByBatch('wordid,word,count_avg,number_of_blocks,tf_idf_avg,signature,local_avg,zone,color', True)
		vectors = []
		labels = []
		colors = []
		
		for word in cursor:
			data = []
			data.append(word[0])
			data.append(word[2])
			data.append(word[3])
			data.append(word[4])
			data.append(word[5])
			data.append(word[6])
			#data.append(word[7])
			vectors.append(data)
			labels.append(word[1])
			colors.append(word[8])
		print('--compact--');
		self.analyzeAndSaveVectors(vectors, labels, colors, 'compact')
		
		return


	def analyzeAndSaveVectors(self, vectors, labels, colors, graphType):
		print(vectors)
		embedded = TSNE(n_components = self.nComponents, perplexity = self.perplexity, verbose = self.verbose, n_iter = self.iteration).fit_transform(vectors)
		index = 0
		for item in vectors:
			data = {}
			data['wordid'] = item[0]
			data['graph_type'] = graphType
			data['label']	= labels[index]
			data['x'] = numpy.float64(embedded[index][0])
			data['y'] =  numpy.float64(embedded[index][1])
			data['color'] = colors[index]
			print(data)
			self.wordPointProcessor.save(data)

			index += 1	
		return