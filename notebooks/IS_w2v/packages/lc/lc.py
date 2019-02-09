import re, sys, numpy
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
from .. import utility
#import store
from sklearn.cluster import KMeans

class LC():


	def __init__(self, text, filterRate = 0.2):
		self.rawText = text
		self.text = self.clean(text)
		self.stopWords = utility.Utility.getStopWords()
		self.stemmer = PorterStemmer()
		self.scores = {}
		self.count = {}
		self.pureWords = {}
		self.properNouns = []
		self.wordData = {}
		self.wordTypes = {}
		self.wordPosition = {}
		self.wordColors = {}
		self.filterRate = filterRate
		self.occuranceContributingFactor = 1
		self.positionContributingFactor = 10
		self.properNounContributingFactor = 100
		self.currentPosition = None
		self.contributingWords = None
		self.max = None
		self.min = None
		self.totalWordsToProcess = 0
		self.colors = ['crimson', 'fuchsia', 'pink', 'plum', 
			'violet', 'darkorchid', 'royalblue', 
			'dodgerblue', 'lightskyblue', 'aqua', 'aquamarine', 'green', 
			'yellowgreen', 'yellow', 'lightyellow', 'lightsalmon', 
			'coral', 'tomato', 'brown', 'maroon', 'gray']
		self.mostImportantColor = 'tomato'
		self.defaultColor = 'yellowgreen'
		self.ignoreLastIndex = False


		self.sentences = []
		self.word2Index = {}
		return


	def setPositionContributingFactor(self, contributingFactor):
		self.positionContributingFactor = contributingFactor
		return


	def setOccuranceContributingFactor(self, contributingFactor):
		self.occuranceContributingFactor = contributingFactor
		return
    
	def setProperNounContributingFactor(self, contributingFactor):
		self.properNounContributingFactor = contributingFactor
		return
		
	def getRawText(self):
		return self.rawText


	def getCleanText(self):
		return self.text


	def getScores(self):
		return self.scores


	def getCount(self):
		return self.count
		

	def getPureWords(self):
		return self.pureWords


	def getProperNouns(self):
		return self.properNouns


	def getTotalContributer(self):
		return len(self.contributingWords)


	def getSentences(self):
		return self.sentences


	def getMinAllowedScore(self):
		if not self.max:
			return 0


		return self.min + ((self.max - self.min) * self.filterRate)


	def getContributers(self):
		if not self.contributingWords:
			return

		properNounElements = self.getProperNounElements()
		mostImportantWords = []
		usedProperNouns = []
		minAllowedScore = self.getMinAllowedScore()

		self.wordData = {}
        
		for word, score in self.contributingWords:
			mostImportantWord = word
			if self.scores[word] <= minAllowedScore:
				break

			if word in properNounElements.keys():
				mostImportantWord = properNounElements[word]
				usedProperNouns.append(mostImportantWord)

			if mostImportantWord not in mostImportantWords:
				mostImportantWords.append(mostImportantWord)
				

			self.wordColors[word] = self.mostImportantColor
		
		mostImportantPureWords = []
		if mostImportantWords:
			for word in mostImportantWords:
				self.wordData[word] = {}
				if word in self.pureWords.keys():
					mostImportantPureWords.append(self.pureWords[word])
					self.wordData[word]['word'] = self.pureWords[word]
					self.wordData[word]['count'] = self.count[word]
					self.wordData[word]['local_avg'] = self.scores[word]
					
				else:
					mostImportantPureWords.append(word)
					self.wordData[word]['word'] = word
					parts = word.split(' ')
					firstPart = self.stemmer.stem(parts[0].lower())
					self.wordData[word]['count'] = self.count[firstPart]
					self.wordData[word]['local_avg'] = self.scores[firstPart]
		
		otherProperNouns = list(utility.Utility.diff(self.properNouns, usedProperNouns))

		if otherProperNouns:
			for properNoun in otherProperNouns:
				parts = properNoun.split(' ')
				firstPart = self.stemmer.stem(parts[0].lower())
				self.wordData[firstPart] = {}
				self.wordData[firstPart]['word'] = self.pureWords[firstPart]
				self.wordData[firstPart]['count'] = self.count[firstPart]
				self.wordData[firstPart]['local_avg'] = self.scores[firstPart]

		for word in self.scores.keys():
			if word in self.wordData.keys():
				continue

			firstPart = self.stemmer.stem(parts[0].lower())
			if word in self.wordData.keys():
				continue

			self.wordData[word] = {}
			if word in self.pureWords.keys():
				self.wordData[word]['word'] = self.pureWords[word]
			else:
				self.wordData[word]['word'] = word
			self.wordData[word]['count'] = self.count[word]
			self.wordData[word]['local_avg'] = self.scores[word]


		'''
		if mostImportantPureWords and otherProperNouns:
			return mostImportantPureWords + otherProperNouns
		elif mostImportantWords:
			return mostImportantPureWords
		elif otherProperNouns:
			return otherProperNouns
		return []
		'''
		return [mostImportantPureWords, otherProperNouns]



	def process(self):
		afterPartsOfSpeachTagging = self.getWords(self.text, True);

		self.totalWordsToProcess = len(afterPartsOfSpeachTagging)
		self.processWords(afterPartsOfSpeachTagging)
		
		self.contributingWords = [(k, self.scores[k]) for k in sorted(self.scores, key=self.scores.get, reverse=True)]
		return


	def processWords(self, words):
		self.currentPosition  = len(words)
		
		lastNounProperNoun = False
		for item in words:
			wordType = item[1]
			mainWord = item[0].lower()
			if (wordType not in ['NNP', 'NNPS', 'NN', 'NNS']) or (len(item[1]) == 1) or (len(item[0]) <= 1) or  (mainWord in self.stopWords):
				if item[0] in ['bin', 'ibn']:
					wordType = 'NNP'
				else:
					# Puntuation or other type of word
					lastNounProperNoun = False
					continue

			if item[0] in ['bin', 'ibn']:
					wordType = 'NNP'

						
			wordIsProperNoun = False
			if wordType in ['NNP', 'NNPS']:
				#print(wordType + ' ' + mainWord)
				self.addToProperNoun(lastNounProperNoun, item[0])
				lastNounProperNoun = True
				wordIsProperNoun = True
			else:
				lastNounProperNoun = False

			#print(mainWord + '--' + str(lastNounProperNoun))


			word = self.stemmer.stem(mainWord)	
			self.pureWords[word] = item[0]
			self.wordTypes[word] = wordType
			self.wordPosition[word] = self.currentPosition
			self.increaseOccurance(word)
			self.increaseScore(word, wordIsProperNoun)
			self.currentPosition -= 1

		return


	def addToProperNoun(self, lastNounProperNoun, mainWord):
		if lastNounProperNoun:
			if self.ignoreLastIndex:
				self.properNouns.append(mainWord)
			else:
				lastIndex = len(self.properNouns) - 1
				self.properNouns[lastIndex] = self.properNouns[lastIndex] + ' ' + mainWord

			self.ignoreLastIndex = False
		elif mainWord not in self.properNouns:	
			self.properNouns.append(mainWord)
			self.ignoreLastIndex = False
		else:
			self.ignoreLastIndex = True
		
		return
		
	
	def increaseOccurance(self, word):
		if word not in self.count.keys():
			self.count[word] = 0

		self.count[word] += 1 
		return


	def increaseScore(self, word, properNoun = False):
		if word not in self.scores.keys():
			self.scores[word] = 0

		self.scores[word] += 1 * self.occuranceContributingFactor + self.currentPosition * self.positionContributingFactor
		if properNoun:
			self.scores[word] += self.properNounContributingFactor

		if not self.max or (self.max < self.scores[word]):
			self.max = self.scores[word]

		if not self.min or (self.min > self.scores[word]):
			self.min = self.scores[word]
		return



	def getWords(self, text, tagPartsOfSpeach = False):
		words = word_tokenize(text)

		if tagPartsOfSpeach:
			return pos_tag(words)

		return words


	def getProperNounElements(self):
		properNounElements = {}

		if self.properNouns:
			for properNoun in self.properNouns:
				parts = properNoun.split(' ')
				for part in parts:
					stemmedWord =  self.stemmer.stem(part.lower())
					properNounElements[stemmedWord] = properNoun

		return properNounElements


	def clean(self, text):
		text = re.sub('<.+?>', '. ', text)
		text = re.sub('&.+?;', '', text)
		text = re.sub('[\']{1}', '', text)
		text = re.sub('[^a-zA-Z0-9\s_\-\?:;\.,!\(\)\"]+', ' ', text)
		text = re.sub('\s+', ' ', text)
		text = re.sub('(\.\s*)+', '. ', text)
		return text

	'''
	def storeWords(self):	
		self.getContributers()
		if not self.wordData:
			return

		wordProcessor = store.Word()
		for key in self.wordData:
			wordProcessor.save(self.wordData[key])

		self.storeCoreWords()
		self.storeLocalContext()
		return


	def storeLocalContext(self):
		if not self.scores:
			return

		config = utility.Config()
		contentId = config.getContentId()
		wordProcessor = store.Word()
		lc = store.LocalContext()
		for stemmed_word in self.scores:
			wordData = {}
			wordData['stemmed_word'] = stemmed_word

			word = wordProcessor.read(wordData)

			if not word:
				word = wordProcessor.readPart(stemmed_word)

			data = {}
			data['contentid'] =  contentId
			data['wordid'] = word[0][0]
			data['word'] = stemmed_word
			data['weight'] =  self.scores[stemmed_word]
			data['count'] =  self.count[stemmed_word]
			if stemmed_word in self.wordColors.keys():
				data['important'] = 1
			lc.save(data)
		return


	def storeCoreWords(self):
		if not self.pureWords:
			return

		coreWordProcessor = store.CoreWord()
		for stemmed_word in self.pureWords:
			data = {}
			data['word'] = self.pureWords[stemmed_word]
			data['stemmed_word'] = stemmed_word
			data['count'] = self.count[stemmed_word]
			coreWordProcessor.save(data)
		
		return
	'''

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


	def loadSentences(self, text):
		stemmer = PorterStemmer()
		processedWords = []
		#allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
		allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
		
		words = self.getWords(text, True)
		currentSentence = []
		for word in words:
			(word, type) = word
			word = re.sub('[^a-zA-Z]+', '', word)
			if type in ['.', '?', '!']:
				if len(currentSentence) > 1:
					# If more than one word than add as sentence
					self.sentences.append(currentSentence)
				currentSentence = []
			if len(word) < 2:
				continue
			if type in allowedPOSTypes:
				#print(type + ' ' + word)
				word = word.lower()
				word = stemmer.stem(word)
				processedWords.append(word)
				if word not in currentSentence:
					currentSentence.append(word)
				if word not in self.word2Index.keys():
					self.word2Index[word] = len(self.word2Index)
			
		if len(currentSentence) > 1:
			# If more than one word than add as sentence
			self.sentences.append(currentSentence)


		return self.word2Index

	def getWordToIndex(self):
		return self.word2Index

	def textToVector(self, text):
		self.loadSentences(text)

		vocabSize = len(self.word2Index)
		vectors = numpy.zeros((vocabSize, vocabSize))
		
		for sentence in self.sentences:
			for word1 in sentence:
				for word2 in sentence:
					if word1 == word2:
						continue
					word1Index = self.word2Index[word1]
					word2Index = self.word2Index[word2]
					vectors[word1Index][word2Index] += 1


		return vectors
