import numpy as np
import utility

class Store():    
    

    def __init__(self, datasetProcessor):
        self.datasetProcessor = datasetProcessor
        self.prefix = ''
        return


    def setPrefix(self, prefix):
        self.prefix = prefix
        return


    def _loadNumpy(self, fileName):
        filePath = self._getFilePath(fileName)
        file = utility.File(filePath)
        if not file.exists():
            return None

        dataFromFile = np.load(filePath)

        for fileRef in dataFromFile:
            return dataFromFile[fileRef]

        return None


    def _saveNumpy(self, fileName, data):
        filePath = self._getFilePath(fileName)
        file = utility.File(filePath)
        file.remove()
        np.savez(filePath, data)
        return


    def _getFilePath(self, fileName):
        fileName = self.prefix + fileName
        path = self.datasetProcessor.getDatasetPath()
        return utility.File.join(path, fileName)