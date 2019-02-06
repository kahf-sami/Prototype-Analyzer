import re, sys, numpy
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
from .. import utility
from sklearn.cluster import KMeans

class Base():


	def __init__(self, text, filterRate = 0.2):
		self.rawText = text
		self.text = self.__clean(text)
		self.stopWords = utility.Utility.getStopWords()
		self.stemmer = PorterStemmer()
        self.wordInfo = {}
        self.featuredWordInfo = {}
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.minWordSize = 2
		return


    '''
    allOptions = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    '''
    def setAllowedPosTypes(self, allowedPOSTypes):
        self.allowedPOSTypes = allowedPOSTypes
        return


    def getRawText(self):
		return self.rawText


	def getCleanText(self):
		return self.text


    def __addWordInfo(self, word):
        if not word:
            return

        localWordInfo = []
        localWordInfo['pure_word'] = word
        wordKey = self.stemmer(word.lower())
        localWordInfo['stemmed_word'] = wordKey
        
        if localWordInfo['stemmed_word'] in self.wordInfo.keys():
            self.wordInfo[wordKey]['count'] += 1
            return

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


	def getPoints(self):
		if not len(self.scores):
			return None

		minAllowedScore = self.getMinAllowedScore()
		totalWords = len(self.scores)
		thetaGap = 360 / (totalWords)
		nodes = []
		importantNodes = []
		theta = 0
		processedWords = []
		
		for word in  self.scores.keys():
			if word in processedWords:
				continue

			processedWords.append(word)
			node = {}
			node['label'] = self.pureWords[word] 	

			radius = self.max - self.scores[word]
			node['x'] = radius * numpy.cos(numpy.deg2rad(theta))
			node['y'] = radius * numpy.sin(numpy.deg2rad(theta))
			node['score'] = self.scores[word]

			if word not in self.wordColors.keys():
				node['color'] = self.defaultColor
			else:
				node['color'] = self.wordColors[word]
			nodes.append(node)

			theta += thetaGap
		
		return nodes

