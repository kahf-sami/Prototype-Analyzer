import os
import tarfile
import numpy as np
import sys
from . import utility
from . import k3s
import re
from nltk.stem.porter import PorterStemmer
import pandas

#from . import util
class Data():
    
    def __init__(self, datasets_root = None):
        if datasets_root:
            self.datasets_root = datasets_root
        else:
            self.datasets_root = './datasets'
            
        self.init_part()
        
        self.dataset_name = 'brexit'
        self.dataset_home = os.path.join(self.datasets_root, self.dataset_name)
        print('Dataset path:', self.dataset_home)
        
        self.vocabWord2Id = {}
        self.vocabId2Word = None
        self.vocab_size = len(self.vocabWord2Id) 
        self.sentences = []
        self.fileNames = {
            'train': None,
            'valid': None,
            'test': None
        }
        self.text = {
            'train': [],
            'valid': [],
            'test': []
        }
        
        self.indexSentence = {
            'train': 0,
            'valid': 0,
            'test': 0
        }

        self.batchSize = 128
        self.skipWindow = 2
        print('-- End of constructor ---')
        
    def resetIndex(self):
        self.indexSentence = {
            'train': 0,
            'valid': 0,
            'test': 0
        }
        return
    
    
    def getVocab(self):
        return self.vocabWord2Id
        
    def loadData(self,force=False):
        print('-- In load data ---')
        self.processTextFiles()
        print('-- Returning from load data --')
        return self.part['train'], self.part['valid'], self.part['test']
    
    def processTextFiles(self):
        print('Processing directory: ', self.dataset_home)
        totalFiles = 0
        textFiles = []
        for root, dirs, files in os.walk(self.dataset_home):
            for file in files:
                if file.endswith(".txt"):
                    textFiles.append(file)
                    totalFiles += 1
                    
                    
        print('Total files', totalFiles)
        print('-------------------------------')
        
        totalTrainingFiles = int(totalFiles * 1);
        self.fileNames['train'] = textFiles[:totalTrainingFiles]
        print('Training file count: ', totalTrainingFiles)
        print('Total files in list: ', len(self.fileNames['train']))
        print('-------------------------------')
        '''
        totalValidationFiles = int(totalFiles * 0.1);
        totalTrainValid = totalTrainingFiles + totalValidationFiles
        self.fileNames['valid'] = textFiles[totalTrainingFiles:totalTrainValid]
        print('Total validation file count', totalValidationFiles)
        print('Total files in list: ', len(self.fileNames['valid']))
        print('-------------------------------')
        
        totalTestFiles = totalFiles - (totalTrainingFiles + totalValidationFiles)
        self.fileNames['test'] = textFiles[totalTrainValid:]
        print('Total test file count', totalTestFiles)
        print('Total files in list: ', len(self.fileNames['test']))
        print('-------------------------------')
        '''
        self.loadDataByType('train');
        #self.loadDataByType('valid');
        #self.loadDataByType('test');
        
        self.vocabId2Word = dict(zip(self.vocabWord2Id.values(), self.vocabWord2Id.keys()))
        return
    
    
    def loadDataByType(self, type):
        if type not in self.fileNames.keys():
            return
        
        
        for fileName in self.fileNames[type]:
            filePath = os.path.join(self.dataset_home, fileName)
            fileHandler = utility.File(filePath)
            text = fileHandler.read()
            self.sentences = []
            textWords = self.getFilteredWords(text)
            self.buildVocabulary(textWords)
            
            for sentence in self.sentences:
                
                sentence = sentence.split(" ")
     
                words = np.array([self.vocabWord2Id[word] for word in sentence])
                self.text[type].append(words)
                
                
        print('-------------------------------')
        print('Vocab size(', type, '): ', self.vocab_size)    
        #print(self.text[type])
        print('-------------------------------')
        print(self.vocabWord2Id)
        return

    
    def next_batch_cbow(self, type = 'train'):
        skip2 = self.skipWindow * 2
        span = skip2 + 1
       
        target = np.ndarray(shape=[self.batchSize], dtype=np.int32)
        context = np.ndarray(shape=[self.batchSize,skip2], dtype=np.int32)
        
        #print([self.batchSize,skip2])

        totalSentences = len(self.text[type])
        wordIndex = 0
        for sentenceIndex in range(self.indexSentence[type], totalSentences):
            #print('Sentence index: ', sentenceIndex)
           
            sentence = self.text[type][self.indexSentence[type]]
            #print('Sentence: ', sentence)
            sentenceWordCounter = 0
            for word in self.text[type][self.indexSentence[type]]:
                #print('Word: ', word)
                if wordIndex == self.batchSize:
                    return target, context
                
                totalWordsInSentence = len(sentence)
                #print('totalWordsInSentence: ', totalWordsInSentence)
                target[wordIndex] = sentence[sentenceWordCounter]
                for contextIndexer in range(skip2):
                    context[wordIndex][contextIndexer] = -1

                for contextIndexer in range(self.skipWindow):
                    #print('sentenceWordCounter: ', sentenceWordCounter)
                    #print('contextIndexer: ', contextIndexer)
                    previousWordIndex = sentenceWordCounter - contextIndexer
                    #print('previousWordIndex: ', previousWordIndex)
                    if (previousWordIndex > 0) :
                        context[wordIndex][self.skipWindow - contextIndexer -1] = sentence[previousWordIndex - 1]
                        
                    nextWordIndex = sentenceWordCounter + contextIndexer + self.skipWindow
                    #print('nextWordIndex: ', nextWordIndex)
                    if (nextWordIndex < (totalWordsInSentence - 1)):
                        context[wordIndex][self.skipWindow + contextIndexer] = sentence[nextWordIndex - 1]
                        
                #print(context[wordIndex])
                #print('------------------------')
                    
                sentenceWordCounter += 1
                wordIndex += 1
                
            self.indexSentence[type] += 1
              
        return target, context
    
    def next_batch_sg(self, type = 'train'):
        skip2 = self.skipWindow * 2
       
        totalLength = self.batchSize * skip2
        #print('total length ', totalLength)
        target = np.zeros(shape=[totalLength], dtype=np.int32)
        context = np.zeros(shape=[totalLength], dtype=np.int32)

        totalSentences = len(self.text[type])
        wordIndex = 0
        #print('Word index ', wordIndex)
        for sentenceIndex in range(self.indexSentence[type], totalSentences):
            #print('Sentence index: ', sentenceIndex)
           
            sentence = self.text[type][self.indexSentence[type]]
            totalInASentence = len(sentence)
            
            #print(sentence)
            #print('totalInASentence: ', totalInASentence)
            sentenceWordCounter = 0
            for word1 in sentence:
                #print('Word: ', word1)
                if wordIndex == totalLength:
                    #print('return ----- 1')
                    return target, context
                
                start = 0
                if sentenceWordCounter >= self.skipWindow:
                    start = sentenceWordCounter - self.skipWindow
                
                end = sentenceWordCounter + self.skipWindow + 1
                if end > totalInASentence:
                    end = totalInASentence - 1
                slicedSectence = sentence[start:end]
                #print(slicedSectence)
                
                for word2 in slicedSectence:
                    if wordIndex == totalLength:
                        #print('return ----- 2')
                        return target, context
                
                    if word1 == word2:
                        continue
                    #print('Word: ', word2)        
                    target[wordIndex] = word1
                    context[wordIndex] = word2
                    wordIndex += 1
                    
                sentenceWordCounter += 1
                #print('-------------------------')
                
                
            self.indexSentence[type] += 1
            #print('return ----- 3')
        return target, context
    
    def to2d(self, x,unit_axis=1):
        if unit_axis==1: # one column
            col = 1
            row = -1
        else:
            col = -1
            row = 1
        return np.reshape(x,[row,col])
    
    def n_batches_wv(self):
        #skip2 = self.skipWindow * 2
        #span = skip2 + 1
        #return ((self.vocab_size - skip2) * skip2) // self.batchSize
        return self.vocab_size // self.batchSize
    
    
    def writeVocab(path):
        os.remove(path)
        df = pandas.Series(self.vocabWord2Id).to_frame()
        df.to_csv(path, mode='w', header = None)
        return
    
    
    def writeSentences(path):
        #np.savetxt('xgboost.txt', a.values, fmt='%d', delimiter="\t", header="X\tY\tZ\tValue") 
        df = pandas.DataFrame(self.sentences)
        #print(df)
        df.to_csv(path, mode='w', header = None)
        return
    
    
    def buildVocabulary(self, text):
        if not self.vocabWord2Id.items():
            self.vocabWord2Id['<Undefined>'] = self.vocab_size
            self.vocab_size += 1
        for word in text:
            if word in self.vocabWord2Id:
                continue
            # Assigning a word an unique id
            self.vocabWord2Id[word] = self.vocab_size
            self.vocab_size += 1
    
    
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
    def getFilteredWords(self, text):
        stemmer = PorterStemmer()
        processedWords = []
        #allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
        allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']

        lc = k3s.LC(text)
        words = lc.getWords(text, True)
        currentSentence = []
        for word in words:
            (word, type) = word
            word = re.sub('[^a-zA-Z]+', '', word)
            if type in ['.', '?', '!']:
                if len(currentSentence) > 1:
                    # If more than one word than add as sentence
                    self.sentences.append(' '.join(currentSentence))
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

        if len(currentSentence) > 1:
            # If more than one word than add as sentence
            self.sentences.append(' '.join(currentSentence))

        return processedWords
    
        
    def init_part(self):
        self.part = {
            'X'        : None,
            'Y'        : None,
            'X_train'  : None,
            'Y_train'  : None,
            'X_valid'  : None,
            'Y_valid'  : None,
            'X_test'   : None,
            'Y_test'   : None,
            'train'    : None,
            'test'     : None,
            'valid'    : None,
        }
        self.index={
            'train'    : 0,
            'test'     : 0,
            'valid'    : 0,
        }