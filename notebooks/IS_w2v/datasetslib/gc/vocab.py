import os, re, numpy
from .. import utility
from nltk import word_tokenize, pos_tag
from nltk.stem.porter import PorterStemmer
import operator


class Vocab():

    def __init__(self, dataSetPath = None):
        self.dataSetPath = dataSetPath
        self.__reset()
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        print('Dataset path: ', self.dataSetPath)
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
    def setAllowedPOSType(self, allowedPOSTypes):
        #self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        #self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.allowedPOSTypes = allowedPOSTypes
        return

    
    def setMinCount(self, min):
        self.minCount = min
        return


    def loadFromFile(self):
        self._load()
        return


    def getId2Words(self):
        return self.vocabId2Word
        

    def buildVocab(self, saveInFile = False):
        if not self.dataSetPath:
            print('Failed to prepare vocab. Undefined dataset path.')
            return

        if self.vocabWord2Id:
            return

        fileNames = self._getListOfTextFiles()
        if fileNames:
            for fileName in fileNames:
                self._processFile(fileName)

        self.__sort()
        self.__reIndex()

        #print('------- Vocab (word -> id) -----')
        #print(self.vocabWord2Id)
        #print('------- Vocab Counter (word -> occurence) -----')
        #print(self.counter)

        if self.vocabWord2Id:
            self.vocabId2Word = dict(zip(self.vocabWord2Id.values(), self.vocabWord2Id.keys()))

        print('--- Finishing building vocab ------')
        self._save()
        return self.vocabWord2Id


    def _save(self):
        filePath = utility.File.join(self.dataSetPath, 'vocab.npz')
        file = utility.File(filePath)
        file.remove()
        numpy.savez(filePath, self.vocabWord2Id)

        filePath = utility.File.join(self.dataSetPath, 'counter.npz')
        file = utility.File(filePath)
        file.remove()
        numpy.savez(filePath, self.counter)
        return


    def _load(self):
        filelPath = utility.File.join(self.dataSetPath, 'vocab.npz')
        file = utility.File(filePath)
        if not file.exists():
            return
        self.vocabWord2Id = numpy.load(filelPath)

        if self.vocabWord2Id:
            self.vocabId2Word = dict(zip(self.vocabWord2Id.values(), self.vocabWord2Id.keys()))

        filelPath = utility.File.join(self.dataSetPath, 'counter.npz')
        file = utility.File(filePath)
        if not file.exists():
            return
        self.counter = numpy.load(self.fileLabelPath)
        return


    def _processFile(self, fileName = None):
        if not fileName:
            return

        text = self._getFileText(fileName)            
        self.sentences = []
        textWords = self._getFilteredWords(text)
        if textWords:
            self.__appendToVocab(textWords)

        return


    def _getFileText(self, fileName):
        filePath = os.path.join(self.dataSetPath, fileName)
        fileHandler = utility.File(filePath)
        text = fileHandler.read()
        return text


    def _getListOfTextFiles(self):
        self.totalFiles = 0
        textFiles = []
        for root, dirs, files in os.walk(self.dataSetPath):
            for file in files:
                if file.endswith(".txt"):
                    textFiles.append(file)
                    self.totalFiles += 1
                                
        print('Total files', self.totalFiles)
        print('-------------------------------')
        return textFiles


    def _getFilteredWords(self, text):
        stemmer = PorterStemmer()
        processedWords = []
        words = self.__getWords(text, True)
        currentSentence = []
        for word in words:
            (word, type) = word
            word = re.sub('[^a-zA-Z0-9\-_]+', '', word)
            if type in ['.', '?', '!']:
                if len(currentSentence) > 1:
                    # If more than one word than add as sentence
                    self.sentences.append(' '.join(currentSentence))
                currentSentence = []
            if len(word) < 2:
                continue

            if type in self.allowedPOSTypes:
                #print(type + ' ' + word)
                word = word.lower()
                word = stemmer.stem(word)
                processedWords.append(word)
                currentSentence.append(word)

        if len(currentSentence) > 1:
            # If more than one word than add as sentence
            self.sentences.append(' '.join(currentSentence))

        return processedWords


    def __reIndex(self):
        if not self.counter.items():
            return

        self.vocabWord2Id = {}
        self.vocabSize = len(self.vocabWord2Id)
        for word, count in self.counter.items():
            if count > self.minCount:
                self.__appendToVocab([word], False)
        
        return

    def __sort(self):
        if not self.counter:
            return
        
        sortedCounter = {}
        for key, value in sorted(self.counter.items(), key=operator.itemgetter(1), reverse=True):
            sortedCounter[key] = value

        self.counter = sortedCounter
        return


    def __getWords(self, text, tagPartsOfSpeach = False):
        words = word_tokenize(text)
        
        if tagPartsOfSpeach:
            return pos_tag(words)
            
        return words

    
    def __appendToVocab(self, textWords, count = True):
        if not self.vocabWord2Id.items():
            self.vocabWord2Id['<Undefined>'] = self.vocabSize
            if count:
                self.counter['<Undefined>'] = 0
            self.vocabSize += 1
            
        for word in textWords:
            if word in self.vocabWord2Id:
                self.counter[word] += 1
                continue
            # Assigning a word an unique id
            self.vocabWord2Id[word] = self.vocabSize
            if count:
                self.counter[word] = 1
            self.vocabSize += 1

        return


    def __reset(self):
        self.vocabWord2Id = {}
        self.counter = {}
        self.vocabId2Word = {}
        self.vocabSize = len(self.vocabWord2Id)
        self.totalFiles = 0
        self.sentences = []
        self.minCount = 2
        return





