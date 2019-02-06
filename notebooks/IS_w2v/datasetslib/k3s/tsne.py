import numpy
from . import Base

class TSNE(Base):


	def __init__(self, text, filterRate = 0.2):
        super().__init__()
		self.sentences = []
        self.punctuationTypes = '.', '?', '!']
		return


	def loadSentences(self, text):
		words = self.__getWords(text, True)
		currentSentence = []
		for word in words:
			(word, type) = word
			word = self.cleanWord(word)
			if type in self.punctuationTypes:
				if len(currentSentence) > 1:
					# If more than one word than add as sentence
					self.sentences.append(currentSentence)
				currentSentence = []
			if len(word) < self.minWordSize:
				continue
			if type in allowedPOSTypes:
                wordKey = self.__addWordInfo(word)
				if wordKey not in currentSentence:
					currentSentence.append(word)

        # Processing last sentence
		if len(currentSentence) > 1:
			# If more than one word than add as sentence
			self.sentences.append(currentSentence)

		return self.sentences
        

	def getWordCoOccurenceVectors(self, text):
		self.loadSentences(text)

		vocabSize = len(self.wordInfo)
		vectors = numpy.zeros((vocabSize, vocabSize))
		
		for sentence in self.sentences:
			for word1 in sentence:
				for word2 in sentence:
					if word1 == word2:
						continue
					word1Index = self.wordInfo[word1]['index']
					word2Index = self.wordInfo[word2]['index']
					vectors[word1Index][word2Index] += 1

		return vectors
