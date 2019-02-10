from . import K3S
import numpy

class Peripheral(K3S):


	def __init__(self, text, filterRate = 0.2):
		super().__init__(text, filterRate)
		self.maxScore = 0
		self.loadSentences(text)
		return


	def train(self):
		totalWords = len(self.filteredWords)
		
		if totalWords == 0:
			return

		currentTheta = 0
		unitTheta = 360 / totalWords
		for word in self.filteredWords:
			self.filteredWords[word]['score'] = self.__getScore(self.filteredWords[word])
			self.filteredWords[word]['theta'] = currentTheta
			currentTheta += unitTheta
			
		return

	
	def __getScore(self, word):
		score = 0
		score += word['count'] * self.occuranceContributingFactor
		if 'proper_noun' in word.keys():
			score += word['proper_noun'] * self.properNounContributingFactor
		for position in word['positions']:
			score += (self.maxPosition - position) * self.positionContributingFactor

		if self.maxScore < score:
			self.maxScore = score
		return score


	def _getX(self, word):
		score = self.filteredWords[word]['score']
		radius = self.maxScore - score
		return radius * numpy.cos(numpy.deg2rad(self.filteredWords[word]['theta']))


	def _getY(self, word):
		score = self.filteredWords[word]['score']
		radius = self.maxScore - score
		return radius * numpy.sin(numpy.deg2rad(self.filteredWords[word]['theta']))


