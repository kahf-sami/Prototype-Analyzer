import os, re, datetime
import utility

class Dataset():
    

    def __init__(self, path, name):
        self.path = path
        self.name = name
        self.datasetPath = None
        self.filePaths = {
            'all': None,
            'train': None,
            'valid': None,
            'test': None
        }
        self.fileIndex = 0

        if path and name:
            self.setDatasetPath(os.path.join(self.path, self.name))
            print('Dataset path:')
            print(self.datasetPath)
        else:
            print('Failed to identify dataset path and name')

        return


    def resetFileIndex(self):
        self.fileIndex = 0
        return

    
    def setDatasetPath(self, path):
        self.datasetPath = path
        return


    def getDatasetPath(self):
        return self.datasetPath


    def loadFilePaths(self, trainSetProportion = 0.8, validationSetProportion = 0.1, testSetProportion = 0.1):
        if not self.datasetPath:
            return

        totalFiles = 0
        textFiles = []
        for root, dirs, files in os.walk(self.datasetPath):
            for file in files:
                if file.endswith(".txt"):
                    fileName = file[:-4]
                    textFiles.append(fileName)
                    totalFiles += 1
                    
                    
        print('Total files', totalFiles)
        self.filePaths['all'] = textFiles
        print('-------------------------------')
        
        totalTrainingFiles = int(totalFiles * 0.8)
        self.filePaths['train'] = textFiles[:totalTrainingFiles]
        print('Training file count: ', totalTrainingFiles)
        print('Total files in list: ', len(self.filePaths['train']))
        print('-------------------------------')
        
        totalValidationFiles = int(totalFiles * 0.1)
        totalTrainValid = totalTrainingFiles + totalValidationFiles
        self.filePaths['valid'] = textFiles[totalTrainingFiles:totalTrainValid]
        print('Total validation file count', totalValidationFiles)
        print('Total files in list: ', len(self.filePaths['valid']))
        print('-------------------------------')
        
        totalTestFiles = totalFiles - (totalTrainingFiles + totalValidationFiles)
        self.filePaths['test'] = textFiles[totalTrainValid:]
        print('Total test file count', totalTestFiles)
        print('Total files in list: ', len(self.filePaths['test']))
        print('-------------------------------')
        return

    
    def getNextTextBlockDetails(self, type = 'train'):
        if type not in self.filePaths.keys():
            return None
        
        total = len(self.filePaths[type])
        if self.fileIndex >= total:
            return None

        details = self.read(self.filePaths[type][self.fileIndex])
        self.fileIndex += 1
        return details


    def read(self, fileName):
        filePath = os.path.join(self.datasetPath, (fileName + '.txt'))
        fileHandler = utility.File(filePath)
        details = {}
        details['filename'] = fileName
        details['text'] = self._clean(fileHandler.read())
        details['timestamp'] = int(datetime.datetime.now().strftime("%s"))
        return details


    def _clean(self, text):
        if not text:
            return ''
        text = re.sub('\s+', ' ', text)
        text = re.sub('\n+', '', text)
        text = re.sub('\'s+', '', text)
        text = re.sub('\'+', '', text)
        text = re.sub('"', '', text)
        return text
    
