from .lda import LDA
import tensorflow as tf
import utility
import numpy as np
from sklearn.manifold import TSNE as skTSNE


class Word2Vec(LDA):


    def __init__(self, dataProcessor):
        super().__init__(dataProcessor)
        self.indexSentence = {
            'train': 0,
            'valid': 0,
            'test': 0
            }
        self.skipWindow = 2
        self.batchSize = 128
        self.embeddingSize  = 128
        self.id2Word = {}
        self.__loadIdToWord()
        self.numberOfEpochs = 500
        self.learningRate = 0.5
        self.validSize = 8
        self.perplexity = 10
        self.numberOfComponents = 10
        self.numberOfIterations = 250
        self.learnedEmbeddings = None
        return
    

    def setSkipWindow(self, skipWindow):
        self.skipWindow = skipWindow
        return


    def setBatchSize(self, batchSize):
        self.batchSize = batchSize
        return


    def resetIndex(self):
        self.indexSentence = {
            'train': 0,
            'valid': 0,
            'test': 0
            }
        return
    

    def nextBatchSkipGram(self, type = 'train'):
        #print("\n----- nextBatchSkipGram -----\n")
        skip2 = self.skipWindow * 2
       
        totalLength = self.batchSize * skip2
        #print('total length ', totalLength)
        target = np.zeros(shape=[totalLength], dtype=np.int32)
        context = np.zeros(shape=[totalLength], dtype=np.int32)

        totalSentences = len(self.processedSentences)
        sampleIndex = 0

        for sentenceIndex in range(self.indexSentence[type], totalSentences):
            if sampleIndex == totalLength:
                return target, context
            
            #print("\n----- Sentence -----\n")
            #print('Sentence index: ', sentenceIndex)           
            sentence = self.processedSentences[sentenceIndex]
            totalInASentence = len(sentence)
            
            #print(sentence)
            #print('totalInASentence: ', totalInASentence)
            sentenceWordCounter = 0
            for word1index in sentence:
                start = 0
                if sentenceWordCounter >= self.skipWindow:
                    start = sentenceWordCounter - self.skipWindow
                
                end = sentenceWordCounter + self.skipWindow + 1
                if end > totalInASentence:
                    end = totalInASentence - 1
                slicedSectence = sentence[start:end]
                
                for word2index in slicedSectence:
                    if sampleIndex == totalLength:
                        return target, context
                
                    if word1index == word2index:
                        # Ignore same word
                        continue
     
                    target[sampleIndex] = word1index
                    context[sampleIndex] = word2index
                    sampleIndex += 1
                    
                sentenceWordCounter += 1
                
            self.indexSentence[type] += 1
            #print('return ----- 3')
        return target, context


    def train(self):
        tf.set_random_seed(123)
        skip = 2 * self.skipWindow
        batchSize = self.batchSize * skip
        embeddingSize = self.embeddingSize * skip
        nNegativeSamples = 64
        vocabSize = len(self.vocab)

        # clear the effects of previous sessions in the Jupyter Notebook
        tf.reset_default_graph()

        x_valid = np.random.choice(self.validSize * 10, self.validSize, replace=False)
        inputs = tf.placeholder(dtype=tf.int32, shape=[batchSize])
        outputs = tf.placeholder(dtype=tf.int32, shape=[batchSize,1])
        inputs_valid = tf.constant(x_valid, dtype=tf.int32)

        # define embeddings matrix with vocab_len rows and embedding_size columns
        # each row represents vectore representation or embedding of a word
        # in the vocbulary
        embedDist = tf.random_uniform(shape=[vocabSize, embeddingSize], minval=-1.0, maxval=1.0)

        embedMatrix = tf.Variable(embedDist, name='embed_matrix')

        # define the embedding lookup table
        # provides the embeddings of the word ids in the input tensor
        embedLookUpTable = tf.nn.embedding_lookup(embedMatrix, inputs)

        # define noise-contrastive estimation (NCE) loss layer

        nceDist = tf.truncated_normal(shape=[vocabSize, embeddingSize], stddev=1.0/tf.sqrt(embeddingSize * 1.0))

        nce_w = tf.Variable(nceDist)
        nce_b = tf.Variable(tf.zeros(shape=[vocabSize]))
        
        loss = tf.reduce_mean(
            tf.nn.nce_loss(weights=nce_w,
                biases=nce_b,
                inputs=embedLookUpTable,
                labels=outputs,
                num_sampled=nNegativeSamples,
                num_classes=vocabSize
                )
            )

        # Compute the cosine similarity between validation set samples
        # and all embeddings.
        norm = tf.sqrt(tf.reduce_sum(tf.square(embedMatrix), 1, 
                                    keepdims=True))
        normalizedEmbeddings = embedMatrix / norm
        embedValid = tf.nn.embedding_lookup(normalizedEmbeddings, 
                                            inputs_valid)
        similarity = tf.matmul(embedValid, normalizedEmbeddings, transpose_b=True)

        nBatches = self.getNumberOfBatched()
        print('Batches: ', nBatches)
        optimizer = tf.train.GradientDescentOptimizer(self.learningRate).minimize(loss)

        saver = tf.train.Saver()
        with tf.Session() as tfs:
            tf.global_variables_initializer().run()
            for epoch in range(self.numberOfEpochs):
                print("--------------------------")
                epochLoss = 0
                self.resetIndex()
                for step in range(nBatches):
                    x_batch, y_batch = self.nextBatchSkipGram()
                    #print(x_batch)
                    #print('----------')
                    #print(y_batch)
                    #print('----------')
                    y_batch = self.to2d(y_batch, unit_axis=1)
                    feedDict = {inputs: x_batch, outputs: y_batch}
                    _, batchLoss = tfs.run([optimizer, loss], feed_dict=feedDict)
                    
                    epochLoss += batchLoss
                    epochLoss = epochLoss / nBatches
                    print('\nAverage loss after epoch ', epoch, ': ', epochLoss)

                # print closest words to validation set at end of every epoch
                similarity_scores = tfs.run(similarity)
        
                top_k = 5
                for i in range(self.validSize):
                    similar_words = (-similarity_scores[i, :]).argsort()[1:top_k + 1]
                    stemmedWord = self.id2Word[x_valid[i]]
                    similar_str = 'Similar to {0:}:'.format(self.vocab[stemmedWord]['label'])
                    for k in range(top_k):
                        stemmedWord = self.id2Word[similar_words[k]]
                        similar_str = '{0:} {1:},'.format(similar_str, self.vocab[stemmedWord]['label'])
                    print(similar_str)
                    
            finalEmbeddings = tfs.run(normalizedEmbeddings)
            path = self.datasetProcessor.getDatasetPath()
            filePath = utility.File.join(path, 'model.ckpt')
            save_path = saver.save(tfs, filePath)
            print('Saving model in ', save_path)
            self.trainTSNE(finalEmbeddings)
        return


    def trainTSNE(self, embedding):
        tsne = skTSNE(perplexity=self.perplexity, 
            n_components=self.numberOfComponents, 
            init='pca', 
            n_iter=self.numberOfIterations, 
            method='exact')
        self.learnedEmbeddings = tsne.fit_transform(embedding)
        self.__saveTSNE()
        print('Trained for TSNE')
        return



    def to2d(self, x,unit_axis=1):
        if unit_axis==1: # one column
            col = 1
            row = -1
        else:
            col = -1
            row = 1
        return np.reshape(x,[row,col])


    def getNumberOfBatched(self):
        return len(self.vocab) // self.batchSize


    def getPoints(self, totalToDisplay = 100, attribute = ''):
        currentIndex = 0
        processedWordInfo = []
        for word in self.vocab:
            index = self.vocab[word]['index']
            self.vocab[word]['topic'] = self.topics[word]
            self.vocab[word]['x'] = self.learnedEmbeddings[index, 0]
            self.vocab[word]['y'] = self.learnedEmbeddings[index, 1]

            currentIndex += 1
            if currentIndex <= totalToDisplay:
                processedWordInfo.append(self.vocab[word])

        return processedWordInfo


    def __loadIdToWord(self):
        if len(self.vocab) == 0:
            return

        self.id2Word = {}
        for word in self.vocab:
            index = self.vocab[word]['index']
            self.id2Word[index] = word
        return


    def __saveTSNE(self):
        path = self.datasetProcessor.getDatasetPath()
        filePath = utility.File.join(path, 'gc-tsne.npz')
        file = utility.File(filePath)
        file.remove()
        np.savez(filePath, self.learnedEmbeddings)
        return


    def __saveEmbedding(self, embedding):
        '''
        path = self.datasetProcessor.getDatasetPath()
        filePath = utility.File.join(path, 'tsne.npz')
        file = utility.File(filePath)
        file.remove()
        np.savez(filePath, self.learnedEmbeddings)
        '''
        return


    def __loadEmbedding(self):
        path = self.datasetProcessor.getDatasetPath()
        filePath = utility.File.join(path, 'model.ckpt')
        saver = tf.train.Saver()
        with tf.Session() as tfs:
            saver.restore(tfs, filePath)
        return
