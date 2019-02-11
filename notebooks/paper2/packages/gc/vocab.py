import os
import lc
import operator
import utility
import numpy

class Vocab():
    
    def __init__(self, datasetProcessor):
        self.datasetProcessor = datasetProcessor
        self.vocab = {}
        self.index = 0
        self.allowedPOSTypes = ['NN', 'NNP', 'NNS', 'NNPS']
        self.processedSentences = []
        return

    
    def setDatasetPath(self, path):
        self.datasetPath = path
        return


    '''
    allOptions = ['NN', 'NNP', 'NNS', 'NNPS', 'JJ', 'JJR', 'JJS' 'RB', 'RBR', 'RBS', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']
    '''
    def setAllowedPosTypes(self, allowedPOSTypes):
        self.allowedPOSTypes = allowedPOSTypes
        return


    def buildVocab(self):
        self.datasetProcessor.resetFileIndex()
        details = self.datasetProcessor.getNextTextBlockDetails()
        while details:
            self.__processText(details['text'])
            details = self.datasetProcessor.getNextTextBlockDetails()
        
        self.__sort()
        self._save()
        print('Total vocab: ', len(self.vocab))
        print('Total sentences: ', len(self.processedSentences))
        print('Finished processing')
        return


    def _load(self):
        path = self.datasetProcessor.getDatasetPath()
        filePath = utility.File.join(path, 'vocab.npz')
        file = utility.File(filePath)
        if not file.exists():
            return

        words = numpy.load(filePath)
        self.vocab = {}
        for fileRef in words:
            for word in words[fileRef]:
                stemmedWord = word['stemmed_word']
                self.vocab[stemmedWord] = word


        filePath = utility.File.join(path, 'sentences.npz')
        file = utility.File(filePath)
        if not file.exists():
            return
        loadedSentences = numpy.load(filePath)

        self.processedSentences = []
        for fileRef in loadedSentences:
            for sentence in loadedSentences[fileRef]:
                self.processedSentences.append(sentence)

        return


    def _save(self):
        path = self.datasetProcessor.getDatasetPath()
        filePath = utility.File.join(path, 'vocab.npz')
        file = utility.File(filePath)
        file.remove()
        numpy.savez(filePath, list(self.vocab.values()))

        filePath = utility.File.join(path, 'sentences.npz')
        file = utility.File(filePath)
        file.remove()
        numpy.savez(filePath, self.processedSentences)
        return


    def __processText(self, text):
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
        self.__addToVocab(localWords)

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


    def __addToVocab(self, words):
        totalWords = len(words)
        if not words:
            return

        currentVocabLength = len(self.vocab)
        currentKeys = self.vocab.keys()
        for word in words.keys():
            if word not in currentKeys:
                wordDetails = {}
                wordDetails['number_of_blocks'] = 1
                wordDetails['total_count'] = words[word]['count']
                wordDetails['label'] = words[word]['pure_word']
                wordDetails['stemmed_word'] = words[word]['stemmed_word']
                if 'score' in words[word].keys():
                    wordDetails['score'] = words[word]['score']
                else:
                    wordDetails['score'] = 0
                wordDetails['index'] = currentVocabLength
                currentVocabLength += 1
            else:
                wordDetails = self.vocab[word]
                wordDetails['number_of_blocks'] += 1
                wordDetails['total_count'] += words[word]['count']
                if 'score' in words[word].keys():
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


        