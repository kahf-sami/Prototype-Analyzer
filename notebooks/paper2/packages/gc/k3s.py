from .lda import LDA
from .plotter import Plotter

class K3S(LDA):


    def __init__(self, datasetProcessor):
        super().__init__(datasetProcessor)
        self.max = 0
        self.setMaxRadius()
        return


    def setMaxRadius(self, attribute = 'number_of_blocks'):
        for word in self.vocab:
            if self.max < self.vocab[word][attribute]:
                self.max = self.vocab[word][attribute]

        return
