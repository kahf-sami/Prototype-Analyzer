import os
import lc
import operator
import utility
import numpy
from .store import Store
import datetime

class Vocab(Store):
    
    def __init__(self, datasetProcessor):
        super().__init__(datasetProcessor)
        self.vocab = {}
        self.index = 0
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.processedSentences = []
        self._load()
        self.sequenceIndex = 0
        return

    
    def setDatasetPath(self, path):
        self.datasetPath = path
        return


    def getVocab(self):
        return self.vocab


    '''
    allOptions = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    '''
    def setAllowedPosTypes(self, allowedPOSTypes):
        self.allowedPOSTypes = allowedPOSTypes
        return


    def buildVocab(self):
        self.datasetProcessor.resetFileIndex()
        details = self.datasetProcessor.getNextTextBlockDetails('all')
        while details:
            self.__processText(details)
            details = self.datasetProcessor.getNextTextBlockDetails()
        
        self.__sort()
        self._save()
        print('Total vocab: ', len(self.vocab))
        print('Total sentences: ', len(self.processedSentences))
        print('Finished processing')
        return


    def _load(self):
        self.__loadVocab()
        self.__loadSentences()
        return


    def _save(self):
        self._saveNumpy('vocab.npz', list(self.vocab.values()))
        self._saveNumpy('sentences.npz', list(self.processedSentences))
        return


    def __loadVocab(self):
        vocabData = self._loadNumpy('vocab.npz')
        if vocabData is not None:
            for word in vocabData:
                stemmedWord = word['stemmed_word']
                self.vocab[stemmedWord] = word
        return


    def __loadSentences(self):
        print('--- loading sentence ---')
        sentenceData = self._loadNumpy('sentences.npz')
        self.processedSentences = []
        if sentenceData is not None:
            
            print(sentenceData)
            for sentence in sentenceData:
                print('-----------')
                print(sentence)
                self.processedSentences.append(sentence)
        return


    def __processText(self, details):
        text = details['text']
        lcProcessor = lc.Peripheral(text, 0)
        lcProcessor.setAllowedPosTypes(self.allowedPOSTypes)
        lcProcessor.setPositionContributingFactor(1)
        lcProcessor.setOccuranceContributingFactor(1)
        lcProcessor.setProperNounContributingFactor(1)
        lcProcessor.setTopScorePercentage(0.5)
        lcProcessor.setFilterWords(0.5)
        lcProcessor.train()
        lcProcessor.loadFilteredWords()

        localWords = lcProcessor.getWordInfo()
        self.__addToVocab(localWords, details)

        localSentences = lcProcessor.getSentences()
        self.__addToSentence(localSentences)
        return


    def __addToSentence(self, sentences):
        if len(sentences) == 0:
            return

        currentKeys = self.vocab.keys()
        for sentence in sentences:
            processedSentence = []
            for word in sentence:
                if word not in currentKeys:
                    continue
                
                processedSentence.append(self.vocab[word]['index'])

            if len(processedSentence) > 1:
                self.processedSentences.append(processedSentence)
        return


    def __addToVocab(self, words, details):
        if 'timestamp' in details.keys():
            self.sequenceIndex = details['timestamp']
        else:
            # Counting lines
            self.sequenceIndex += 1 
        totalWords = len(words)
        if not words:
            return

        currentVocabLength = len(self.vocab)
        currentKeys = self.vocab.keys()
        for word in words.keys():
            currentWordKeys = words[word].keys()
            if word not in currentKeys:
                wordDetails = {}
                wordDetails['number_of_blocks'] = 1
                wordDetails['total_count'] = words[word]['count']
                wordDetails['label'] = words[word]['pure_word']
                wordDetails['stemmed_word'] = words[word]['stemmed_word']
                if 'score' in currentWordKeys:
                    wordDetails['score'] = words[word]['score']
                else:
                    wordDetails['score'] = 0
                if 'appeared' not in currentWordKeys:
                    wordDetails['appeared'] = self.sequenceIndex
                wordDetails['index'] = currentVocabLength
                currentVocabLength += 1
            else:
                wordDetails = self.vocab[word]
                wordDetails['number_of_blocks'] += 1
                wordDetails['total_count'] += words[word]['count']
                if 'score' in currentWordKeys:
                    wordDetails['score'] += words[word]['score']

            self.vocab[word] = wordDetails
        return


    def __sort(self, attribute = 'number_of_blocks'):
        if len(self.vocab) == 0:
            return
        
        sortedVocab = {}
        for value in sorted(self.vocab.values(), key=operator.itemgetter(attribute), reverse=True):
            sortedVocab[value['stemmed_word']] = value

        self.vocab = sortedVocab
        return


        