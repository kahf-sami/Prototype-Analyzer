import numpy as np
from .vocab import Vocab
from scipy.sparse import csr_matrix
import utility
from sklearn.decomposition import LatentDirichletAllocation

class LDA(Vocab):

    def __init__(self, datasetProcessor):
        super().__init__(datasetProcessor)
        self.iteration = 500
        self.verbose = 1
        self.perplexity = 10
        self.numberOfTopics = 10
        self.wordCoOccurenceVector = None
        self.topics = {}
        self._load()
        return

    def setNumberOfIterations(self, number):
        self.iteration = number
        return


    def setPerplexity(self, perplexity):
        self.perplexity = perplexity
        return


    def setVerbose(self, verbose):
        self.verbose = verbose
        return


    def loadWordCoOccurenceVectorsFromFile(self):
        self.wordCoOccurenceVector = self.__loadSparseCsr()
        return self.wordCoOccurenceVector

    def loadTopics(self):
        self.__loadLdaTopics()
        return self.topics


    def buildWordCoOccurenceVectors(self, saveInFile = False):
        if not self.dataSetPath:
            print('Failed to prepare word co-occurance matrix. Undefined dataset path.')
            return

        if self.wordCoOccurenceVector:
            return

        self.wordCoOccurenceVector = np.zeros((self.vocabSize, self.vocabSize))
        fileNames = self._getListOfTextFiles()
        if fileNames:
            for fileName in fileNames:
                self._processFileSentences(fileName)

            self.__convertToSparseMatrix()
            if saveInFile:
                self.__saveSparseCsr(self.wordCoOccurenceVector)

        return self.wordCoOccurenceVector


    def train(self):
        lda = LatentDirichletAllocation(n_components=self.numberOfTopics, max_iter=self.iteration, learning_method='online', learning_offset=1.0,random_state=0).fit(self.wordCoOccurenceVector)
        wordScores = {}
        
        self.topics = {}
        for topic_idx, topics in enumerate(lda.components_):
            for i in topics.argsort():
                word = self.vocabId2Word[i]
                if word in self.topics.keys():
                    if self.topics[word] < topics[i]:
                        self.topics[word] = topic_idx
                else:
                    self.topics[word] = topic_idx
                    
        print(self.topics)
        self.__saveLdaTopics()
        return


    def _processFileSentences(self, fileName = None):
        if not fileName:
            return

        text = self._getFileText(fileName)            
        self.sentences = []
        textWords = self._getFilteredWords(text)
        if self.sentences:
            wordKeys = list(self.vocabWord2Id.keys())
            for sentence in self.sentences:
                sentenceWords = sentence.split(" ")
                for word1 in sentenceWords:
                    if word1 not in wordKeys:
                        continue

                    word1Index = self.vocabWord2Id[word1]
                    for word2 in sentenceWords:
                        if word2 not in wordKeys:
                            continue

                        word2Index = self.vocabWord2Id[word2]
                        self.wordCoOccurenceVector[word1Index][word2Index] += 1

        return

    	
    def __saveSparseCsr(self, vectors):
        filePath = utility.File.join(self.dataSetPath, 'word_cooccurence.npz')
        file = utility.File(filePath)
        file.remove()
        np.savez(filePath, data=vectors.data, indices=vectors.indices, indptr=vectors.indptr, shape=vectors.shape)
        return

    def __saveLdaTopics(self):
        filePath = utility.File.join(self.dataSetPath, 'lda.npz')
        file = utility.File(filePath)
        file.remove()
        np.savez(filePath, self.topics)
        return

    def __loadLdaTopics(self):
        filePath = utility.File.join(self.dataSetPath, 'lda.npz')
        file = utility.File(filePath)
        if not file.exists():
            return
        self.topics = np.load(filePath)
        return
        
        
    def __loadSparseCsr(self):
        filePath = utility.File.join(self.dataSetPath, 'word_cooccurence.npz')
        file = utility.File(filePath)
        if(not file.exists()):
            return None
        
        loader = np.load(filePath)
        if ((loader['shape'][0] != loader['shape'][1])):
            return None

        return csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'])

        
    def __convertToSparseMatrix(self):
        data = []
        rows = []
        columns = []
        for i in range(0, self.vocabSize):
            for j in range(0, self.vocabSize):
                if self.wordCoOccurenceVector[i][j] > 0:
                    rows.append(i)
                    columns.append(j)
                    data.append(self.wordCoOccurenceVector[i][j])
        
        self.wordCoOccurenceVector = csr_matrix((data, (rows, columns)), shape=(self.vocabSize, self.vocabSize))
        return

    