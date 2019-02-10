from . import Dataset
import utility
import os
import json
import datetime

class Brexit(Dataset):


    def __init__(self, path):
        super().__init__(path, 'brexit')
        return

    def read(self, fileName):
        identifier = fileName[:-8]
        #print('identifier:', identifier)
        details = super().read(fileName)
        if not details:
            return details

        filePath = os.path.join(self.datasetPath, (identifier + '.json'))
        fileHandler = utility.File(filePath)
        otherDetails = fileHandler.read()
        if otherDetails:
                otherDetails = json.loads(otherDetails)
                if 'Title' in otherDetails.keys():
                    details['text'] = self._clean(otherDetails['Title']) + '. ' + details['text']

                if 'Date' in otherDetails.keys():
                    details['timestamp'] = int(datetime.datetime.strptime(otherDetails['Date'], '%Y-%m-%d').strftime("%s"))

        return details

