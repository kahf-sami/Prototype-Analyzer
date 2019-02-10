import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import SnowballStemmer 
from nltk import word_tokenize, pos_tag
import re
import sys
from .utility import Utility


class NLP():


	def __init__(self, textBlock = None):
		self.textBlock = textBlock

		return

	def removePunctuation(self, textBlock = None):
		if not textBlock:
			textBlock = self.textBlock

		if not textBlock:
			return None

		textBlock = re.sub('\(\)', '', str(textBlock))
		textBlock = re.sub('\'s', '', str(textBlock))
		textBlock = re.sub('\'', '', str(textBlock))
		textBlock = re.sub('-\n', '', str(textBlock))
		textBlock = re.sub('[' + string.punctuation + ']', ' ', str(textBlock))
		textBlock = re.sub('\s+', ' ', str(textBlock))

		
		return textBlock


	def lower(self, textBlock = None):
		if not textBlock:
			textBlock = self.textBlock

		if not textBlock:
			return None

		return textBlock.lower()

	def removeNewLine(self, textBlock = None):
		if not textBlock:
			textBlock = self.textBlock

		if not textBlock:
			return None

		return textBlock.replace("\n", "")


	def removeHtmlTags(self, textBlock = None):
		if not textBlock:
			textBlock = self.textBlock

		if not textBlock:
			return None

		# First we remove inline JavaScript/CSS:
		cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", textBlock)

		# Then we remove html comments. This has to be done before removing regular
		# tags since comments can contain '>' characters.
		cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)

		# Next we can remove the remaining tags:
		cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
		

		# Finally, we deal with whitespace
		cleaned = re.sub(r"&[a-z0-9]+;", " ", cleaned)
		cleaned = re.sub(r"\s+", " ", cleaned)

		return cleaned


	def removeStopWord(self, textBlock = None):
		if not textBlock:
			textBlock = self.textBlock

		if not textBlock:
			return None

		words =  word_tokenize(textBlock)
		filteredWords = [word for word in words if word not in stopwords.words('english')]
		filteredWords = [word for word in words if word not in ['etc', 'part', 'term']]
		return filteredWords


	def stem(self, filteredWords = None, algorithm = 'Snowball'):
		if not filteredWords:
			filteredWords = self.filteredWords

		if not filteredWords:
			return None

		if algorithm == 'Porter':
			stemmer = PorterStemmer()
		elif algorithm == 'Lancasters':
			stemmer = LancasterStemmer()
		else:
			stemmer = SnowballStemmer('english')

		stemmedWords = [stemmer.stem(word) for word in filteredWords]
	
		return stemmedWords


	def getFiltered(self, textBlock = None):
		if not textBlock:
			textBlock = self.textBlock

		if not textBlock:
			return None

		textBlock = self.removePunctuation(textBlock)
		textBlock = self.lower(textBlock)
		textBlock = self.removeNewLine(textBlock)
		textBlock = self.removeHtmlTags(textBlock)
		filteredWords = self.removeStopWord(textBlock)
		filteredWords = self.stem(filteredWords)
		return " ".join(filteredWords)


	def getAsciiSum(self, textBlock):
		words =  word_tokenize(textBlock)
		words = Utility.unique(words)
		wordsString = ''.join(words)
		
		asciiSum = 0
		for char in wordsString:
			asciiSum += ord(char)

		return asciiSum


	def getCapitals(self, textBlock):
		words =  word_tokenize(textBlock)
		capitals = []
		normal	= []
		#for word in words:



		return


	'''
	1.	CC	Coordinating conjunction
	2.	CD	Cardinal number
	3.	DT	Determiner
	4.	EX	Existential there
	5.	FW	Foreign word
	6.	IN	Preposition or subordinating conjunction
	7.	JJ	Adjective
	8.	JJR	Adjective, comparative
	9.	JJS	Adjective, superlative
	10.	LS	List item marker
	11.	MD	Modal
	12.	NN	Noun, singular or mass
	13.	NNS	Noun, plural
	14.	NNP	Proper noun, singular
	15.	NNPS	Proper noun, plural
	16.	PDT	Predeterminer
	17.	POS	Possessive ending
	18.	PRP	Personal pronoun
	19.	PRP$	Possessive pronoun
	20.	RB	Adverb
	21.	RBR	Adverb, comparative
	22.	RBS	Adverb, superlative
	23.	RP	Particle
	24.	SYM	Symbol
	25.	TO	to
	26.	UH	Interjection
	27.	VB	Verb, base form
	28.	VBD	Verb, past tense
	29.	VBG	Verb, gerund or present participle
	30.	VBN	Verb, past participle
	31.	VBP	Verb, non-3rd person singular present
	32.	VBZ	Verb, 3rd person singular present
	33.	WDT	Wh-determiner
	34.	WP	Wh-pronoun
	35.	WP$	Possessive wh-pronoun
	36.	WRB	Wh-adverb
	'''
	def getNouns(self, textBlock):
		afterPartsOfSpeachTagging = self.getWords(textBlock, True)
		words = {}
		words['NNP'] = []
		words['NNPS'] = []
		words['NN'] = []
		words['NNS'] = []
		
		stopWords = self.getLocalStopWords()
		stemmer = PorterStemmer()
		for item in afterPartsOfSpeachTagging:
			word = item[0].lower()
			wordType = item[1]
			if (item[1] in ['NNPS', 'NNS']):
				word = stemmer.stem(word)

			if (word in stopWords) or (len(word) <= 2):
				continue

			if (item[1] in ['NNP', 'NNPS', 'NN', 'NNS']) and (word not in words):
				words[item[1]].append(word)
				stopWords.append(word)

		filteredWords = words['NNP'] + words['NNPS'] + words['NN'] + words['NNS']
		return filteredWords



	def getLocalStopWords(self):
		return ['etc', 'part', 'term', 'number', 'i.e', 'whose', 'whenever', 'need', 's', 
			'o', 'none', 'him', 'nobody', 'anything', 'your', 'means', 'do', 'did', 'yes', 'no']


	def getWordsByType(self, textBlock, type = None):
		afterPartsOfSpeachTagging = self.getWords(textBlock, True)
		
		if not type:
			return afterPartsOfSpeachTagging

		words = []
		for item in afterPartsOfSpeachTagging:
			word = item[0]
			wordType = item[1]

			if wordType == type:
				words.append(word)

		return words


	def getWords(self, textBlock, tagPartsOfSpeach = False):
		words = word_tokenize(textBlock)

		if tagPartsOfSpeach:
			return pos_tag(words)

		return words
