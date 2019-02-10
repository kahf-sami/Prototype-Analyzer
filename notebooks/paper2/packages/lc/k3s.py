import re, sys, numpy
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
import utility
from . import Base

class K3S(Base):


	def __init__(self, text, filterRate = 0.2):
		super().__init__(text, filterRate)
		self.occuranceContributingFactor = 1
		self.positionContributingFactor = 10
		self.properNounContributingFactor = 100
		self.maxPosition = 0
		self.properNouns = []
		self.lastNounProperNoun = False
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


	def getProperNouns(self):
		return self.properNouns


	def _addWordInfo(self, word, type):
		if type not in ['NNP', 'NNPS']:
			self.lastNounProperNoun = False

		if not word or (type not in self.allowedPOSTypes):
			return None

		wordKey = super()._addWordInfo(word, type)
		keys = self.wordInfo[wordKey].keys()
		if 'positions' not in keys:
			self.wordInfo[wordKey]['positions'] = []

		if 'proper_noun' not in keys:
			self.wordInfo[wordKey]['proper_noun'] = 0

		self.wordInfo[wordKey]['positions'].append(self.maxPosition)
		if type in ['NNP', 'NNPS']:
			self.wordInfo[wordKey]['proper_noun'] += 1
			self.addToProperNoun(word)
			self.lastNounProperNoun = True
		else:
			self.lastNounProperNoun = False

		self.maxPosition += 1
		return wordKey


	def addToProperNoun(self, mainWord):
		if self.lastNounProperNoun:
			lastIndex = len(self.properNouns) - 1
			self.properNouns[lastIndex] = self.properNouns[lastIndex] + ' ' + mainWord
		elif mainWord not in self.properNouns:	
			self.properNouns.append(mainWord)
		
		return

