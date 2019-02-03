from .lda import LDA
import store
import math, numpy, sys, decimal

class K3S(LDA):


	def __init__(self):
		super().__init__()
		self.graphTypes = ['count_avg', 'number_of_blocks', 'tf_idf_avg', 'signature', 'local_avg', 'zone', '*']
		self.radiusIncrementFactor = {}
		self.loadRadiusIncrementFactor()
		self.thetaIncrementFactorPerZone = {}
		self.loadThetaIncrementFactor()
		return


	def storePointsPerGraphType(self):
		for graphType in self.graphTypes:
			self.wordPointProcessor.deleteAllByGraphType(graphType)
			self.storeWordPointsforGraphType(graphType)
			#self.storeContentPointsforGraphType(graphType)
		return


	def storeContentPointsforGraphType(self, graphType):
		contentCursor = self.contentStorer.getContentsByBatch()

		for content in contentCursor:
			self.saveContentPoint(content[0], content[1], graphType)

		return


	def saveContentPoint(self, contentid, name, graphType):
		wordPoints = self.wordPointProcessor.getAssociatedPoints(contentid, graphType)
		numberOfWords = len(wordPoints)
		sumX = 0
		sumY = 0

		for wordPoint in wordPoints:
			sumX += wordPoint[1]
			sumY += wordPoint[2]
 
		data = {}
		data['contentid'] = contentid
		data['graph_type'] = graphType
		data['label'] = name
		data['y'] = sumY / numberOfWords
		data['x'] = sumX / numberOfWords
		data['r'] = math.sqrt(float(data['x']) * float(data['x']) + float(data['y']) * float(data['y']))

		zone = self.getZone(data['r'])
		data['color'] = self.colors[zone]
		self.contentPointProcessor.save(data)
		return


	def storeWordPointsforGraphType(self, graphType):
		column = 'wordid, word, ' + graphType + ', color'
		max = 0
		if graphType == '*':
			column = '*'
			for graphTypeItem in self.graphTypes:
				if graphTypeItem == '*':
					continue
				max += decimal.Decimal(self.wordStorer.getMaxValue(graphTypeItem)) * self.radiusIncrementFactor[graphTypeItem]
		else:
			max = decimal.Decimal(self.wordStorer.getMaxValue(graphType)) * self.radiusIncrementFactor[graphType]
		cursor = self.wordStorer.getWordsByBatch(column, True)
		max += 1


		thetaIncrement = 360 / self.totalWords
		theta = 0
		index = 0
		for word in cursor:
			index += 1
			print(index)
			data = {}
			data['wordid'] = word[0]
			data['graph_type'] = graphType
			data['label']	= word[1]
			if graphType == '*':
				data['r'] = self.radiusIncrementFactor['count_avg'] * word[3] 
				+ self.radiusIncrementFactor['number_of_blocks'] *  word[4] 
				+ self.radiusIncrementFactor['tf_idf_avg'] *  decimal.Decimal(word[5])
				+ self.radiusIncrementFactor['signature'] *  word[6] 
				+ self.radiusIncrementFactor['local_avg'] *  word[7]
				#+ self.radiusIncrementFactor['zone'] *  word[8] 
				score = self.radiusIncrementFactor['count_avg'] + self.radiusIncrementFactor['number_of_blocks']
				+ self.radiusIncrementFactor['tf_idf_avg'] + self.radiusIncrementFactor['signature'] + self.radiusIncrementFactor['local_avg']
				#+ self.radiusIncrementFactor['zone']
			else:
				data['r'] = max - (self.radiusIncrementFactor[graphType] * decimal.Decimal(word[2]))
				score = word[2]
			#zone = self.getZone(score)
			data['theta'] = theta 
			theta = theta + thetaIncrement
			data['color'] = word[3]
			#if zone in self.thetaIncrementFactorPerZone:
				#data['theta'] = theta + self.thetaIncrementFactorPerZone[zone]
				#data['color'] = word[3]
			#else:
				#data['theta'] = theta + 1
				#data['color'] = word[3]

			data['x'] = float(data['r']) * numpy.cos(numpy.deg2rad(data['theta']))
			data['y'] = float(data['r']) * numpy.sin(numpy.deg2rad(data['theta']))
			
			self.wordPointProcessor.save(data)
			print(data)
			#theta = data['theta']

		

		return


	def loadRadiusIncrementFactor(self):
		self.radiusIncrementFactor = {}
		self.radiusIncrementFactor['count_avg'] = decimal.Decimal(0.2)
		self.radiusIncrementFactor['number_of_blocks'] = decimal.Decimal(1) 
		self.radiusIncrementFactor['tf_idf_avg'] = decimal.Decimal(1)
		self.radiusIncrementFactor['signature'] = decimal.Decimal(0.01)
		self.radiusIncrementFactor['local_avg'] = decimal.Decimal(0.01)
		self.radiusIncrementFactor['zone'] = decimal.Decimal(0.1)
		return



	def loadThetaIncrementFactor(self):
		self.thetaIncrementFactorPerZone = {}

		zones = self.wordStorer.getZones()

		if not zones:
			return

		for zone in zones:
			if zone[0] == 0:
				self.thetaIncrementFactorPerZone[zone[1]] = 0
			else:
				self.thetaIncrementFactorPerZone[zone[1]] = 360 / zone[0]
		return



