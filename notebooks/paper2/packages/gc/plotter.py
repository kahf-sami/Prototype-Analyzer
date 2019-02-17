import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams

class Plotter():    
    

    def __init__(self, wordInfo):
        self.wordInfo = wordInfo
        self.colors = ['crimson', 'fuchsia', 'pink', 'plum', 
            'violet', 'darkorchid', 'royalblue', 
            'dodgerblue', 'lightskyblue', 'aqua', 'aquamarine', 'green', 
            'yellowgreen', 'yellow', 'lightyellow', 'lightsalmon', 
            'coral', 'tomato', 'brown', 'maroon', 'gray']
        return


    def displayPlot(self):
        #rcParams['figure.figsize']=15,10
        points = self.getPoints()
        if not points:
            print('No points to display')
            return

        plt.figure(figsize=(20, 20))  # in inches(x, y, s=None, c=None, marker=None, cmap=None, norm=None, vmin=None, vmax=None, alpha=None, linewidths=None, verts=None, edgecolors=None, *, data=None, **kwargs)[source]
        for point in points:
            plt.scatter(point['x'], point['y'], c = point['color'])
            plt.annotate(point['label'], 
                xy=(point['x'], point['y']), 
                xytext=(5, 2), 
                textcoords='offset points', 
                ha='right', 
                va='bottom')
        
        plt.show()
        return


    def getPoints(self):
        if not len(self.wordInfo):
            return None

        points = []
        for word in self.wordInfo:
            point = {}
            point['x'] = word['x']
            point['y'] = word['y']
            point['color'] = self.colors[word['topic']]
            point['label'] = word['label']
            points.append(point)

        return points