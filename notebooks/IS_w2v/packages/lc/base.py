import re, sys, numpy
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
from .. import utility
from sklearn.cluster import KMeans
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams


class Base():


	def __init__(self, text, filterRate = 0):
		self.rawText = text
		self.text = self.__clean(text)
		self.stopWords = utility.Utility.getStopWords()
		self.stemmer = PorterStemmer()
		self.wordInfo = {}
		self.featuredWordInfo = {}
		self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
		self.minWordSize = 2
		self.sentences = []
		self.punctuationTypes = ['.', '?', '!']
		self.maxCount = 1
		self.filterRate = filterRate
		self.topScorePercentage = filterRate
		self.filteredWords = {}
		self.contributors = []
		return


	'''
	allOptions = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
	'''
	def setAllowedPosTypes(self, allowedPOSTypes):
		self.allowedPOSTypes = allowedPOSTypes
		return


	def setFilterWords(self, filterRate = 0.2):
		self.filterRate = filterRate
		self.loadFilteredWords()
		return


	def setTopScorePercentage(self, topScorePercentage):
		self.topScorePercentage = topScorePercentage
		return


	def getRawText(self):
		return self.rawText


	def getCleanText(self):
		return self.text


	def getContrinutors(self):
		return self.contributors


	def loadFilteredWords(self):
		minAllowedScore = self.maxCount * self.filterRate
		self.filteredWords = {}
		for word in self.wordInfo:
			if self.wordInfo[word]['count'] <= minAllowedScore:
				continue

			index = len(self.filteredWords)
			self.filteredWords[word] = self.wordInfo[word]
			self.filteredWords[word]['index'] = index

		print("Total vocab: ", len(self.wordInfo))
		print("Filtered vocab: ", len(self.filteredWords))
		return self.filteredWords


	def loadSentences(self, text):
		words = self.__getWords(text, True)
		self.sentences = []
		currentSentence = []
		for word in words:
			(word, type) = word
			word = self.__cleanWord(word)
			if type in self.punctuationTypes:
				if len(currentSentence) > 1:
					# If more than one word than add as sentence
					self.sentences.append(currentSentence)
				currentSentence = []
			if len(word) < self.minWordSize:
				continue
			
			wordKey = self._addWordInfo(word, type)
			if wordKey and (wordKey not in currentSentence):
				currentSentence.append(word)

        # Processing last sentence
		if len(currentSentence) > 1:
			# If more than one word than add as sentence
			self.sentences.append(currentSentence)

		self.filteredWords = self.wordInfo
		return self.sentences


	def displayPlot(self):
		#rcParams['figure.figsize']=15,10
		points = self.getPoints()
		if not points:
			print('No points to display')
			return

		plt.figure(figsize=(20, 20))  # in inches(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, data=None, **kwargs)[source]
		for point in points:
			plt.scatter(point['x'], point['y'], c = point['color'])
			plt.annotate(point['label'], 
				xy=(point['x'], point['y']), 
				xytext=(5, 2), 
				textcoords='offset points', 
				ha='right', 
				va='bottom')
				
		plt.show()
		return


	def getPoints(self):
		if not len(self.wordInfo):
			return None

		topWordScores = self.maxCount * self.topScorePercentage

		points = []
		for word in self.filteredWords:
			point = {}
			point['x'] = self._getX(word)
			point['y'] = self._getY(word)
			point['color'] = 'green'
			point['label'] = self.filteredWords[word]['pure_word']
			if self.filteredWords[word]['count'] >= topWordScores:
				point['color'] = 'red'
				self.contributors.append(word)

			points.append(point)

		return points

	def _getX(self, word):
		return 0


	def _getY(self, word):
		return 0	


	def _addWordInfo(self, word, type):
		if not word or (type not in self.allowedPOSTypes):
			return None

		localWordInfo = {}
		localWordInfo['pure_word'] = word
		wordKey = self.stemmer.stem(word.lower())
		localWordInfo['stemmed_word'] = wordKey

		if localWordInfo['stemmed_word'] in self.wordInfo.keys():
			self.wordInfo[wordKey]['count'] += 1
			if self.maxCount < self.wordInfo[wordKey]['count']:
				self.maxCount = self.wordInfo[wordKey]['count']
			return wordKey

		localWordInfo['count'] = 1
		localWordInfo['index'] = len(self.wordInfo)
		self.wordInfo[wordKey] = localWordInfo

		return wordKey


	def __getWords(self, text, tagPartsOfSpeach = False):
		words = word_tokenize(text)

		if tagPartsOfSpeach:
			return pos_tag(words)

		return words


	def __cleanWord(self, word):
		return re.sub('[^a-zA-Z0-9]+', '', word)


	def __clean(self, text):
		text = re.sub('<.+?>', '. ', text)
		text = re.sub('&.+?;', '', text)
		text = re.sub('[\']{1}', '', text)
		text = re.sub('[^a-zA-Z0-9\s_\-\?:;\.,!\(\)\"]+', ' ', text)
		text = re.sub('\s+', ' ', text)
		text = re.sub('(\.\s*)+', '. ', text)
		return text

