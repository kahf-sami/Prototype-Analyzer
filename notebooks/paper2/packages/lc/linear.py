from . import K3S

class Linear(K3S):
    
    
    def __init__(self, text, filterRate = 0.2):
        super().__init__(text, filterRate)
        self.loadSentences(text)
        return
        
        
    def _getX(self, word):
        avg = 0
        for position in self.filteredWords[word]['positions']:
            avg += position

        avg = avg / len(self.filteredWords[word]['positions'])
        #return self.filteredWords[word]['positions'][0]
        return avg


    def _getY(self, word):
        return self.filteredWords[word]['count']

