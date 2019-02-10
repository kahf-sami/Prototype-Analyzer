import sys, math
import store

class Common():


	def __init__(self):
		self.contentStorer = store.Content()
		self.wordStorer = store.Word()
		self.lCStorer = store.LocalContext()
		self.contentPointProcessor = store.ContentPoint()
		self.wordPointProcessor = store.WordPoint()
		self.totalWords = self.wordStorer.getTotalWords()
		self.totalTextBlocks = self.contentStorer.getTotal()

		self.colors = ['crimson', 'fuchsia', 'pink', 'plum', 
			'violet', 'darkorchid', 'royalblue', 
			'dodgerblue', 'lightskyblue', 'aqua', 'aquamarine', 'green', 
			'yellowgreen', 'yellow', 'lightyellow', 'lightsalmon', 
			'coral', 'tomato', 'brown', 'maroon', 'gray']
		return


	def store(self):
		#cursor = self.wordStorer.getWordsByBatch('wordid,number_of_blocks')

		offset = 0
		while offset < self.totalWords:
			word = self.wordStorer.getNextWord('wordid,number_of_blocks', offset)
			#print(word)
			#for word in cursor:
			if not word:
				offset += 1
				continue
			localDetails = self.lCStorer.getAvgWeightDetails(word[0])
			localScoreAvgAll = 0;
			if localDetails:				
				totatContentsAppeared = len(localDetails)
				for row in localDetails:
					totalWordsInAContent = self.lCStorer.getTotalWordsInAContent(row[0])
					localScore = row[1] / totalWordsInAContent
					localScoreAvgAll += localScore

				localScoreAvgAll = localScoreAvgAll / totatContentsAppeared


			details = self.lCStorer.getAvgDetails(word[0])
			
			if not details[0]:
				offset += 1
				continue;

			data = {}
			data['wordid'] = word[0]
			data['count_avg'] = details[2]
			data['tf_idf_avg'] = details[1]
			data['local_avg'] = localScoreAvgAll
			#data['local_avg'] = details[0]
			data['zone'] = self.getZone(word[1])
			#print(data)
			self.wordStorer.update(data, data['wordid'])
			offset += 1

		return


	def updateAllLCTfIdf(self):
		contentCursor = self.contentStorer.getContentsByBatch()

		for content in contentCursor:
			self.saveLocalTfIdf(content[0])

		return


	def getWordPoints(self, graphType = '*'):
		return self.convert(self.wordPointProcessor.getPoints(graphType), 'word')


	def getContentPoints(self, graphType = '*'):
		return self.convert(self.contentPointProcessor.getPoints(graphType), 'content')


	def saveLocalTfIdf(self, contentid):
		lcWords = self.lCStorer.getLCs(contentid)
		totalWordsInContent = len(lcWords)
	

		for row in lcWords:
			wordData = {}
			wordData['wordid'] = row[2]

			wordDetails = self.wordStorer.read(wordData)
			numberOfDocHavingword = wordDetails[0][4]
			
			totalOccurred = row[6]
			
			data = {}
			data['local_contextid'] = row[0]
			# TF(t) = (Number of times term it appears in a document) / (Total number of terms in the document).
			tf = totalOccurred / totalWordsInContent
			
			# IDF(t) = log_e(Total number of documents / Number of documents with term t in it).
			idf = math.log(int(self.totalTextBlocks) / numberOfDocHavingword)
			
			data['tf_idf'] = "{0:.2f}".format(tf * idf * 10)
			'''
			print('Total text blocks: ' + str(self.totalTextBlocks))
			print('numberOfDocHavingword: ' + str(numberOfDocHavingword))
			print('totalOccurred: ' + str(totalOccurred))
			print('tf: ' + str(tf))
			print('idf: ' + str(idf))
			print('tf_idf: ' + str(data['tf_idf']))
			print(data)
			'''
			self.lCStorer.update(data, row[0])

		return


	def convert(self, data, type):
		if not data:
			return None

		totalColors = len(self.colors)
		processedData = []
		for item in data:
			processedItem = {}
			processedItem['label'] = item[2]
			processedItem['x'] = item[4]
			processedItem['y'] = item[5]


			if item[8].isdigit():
				colorIndex = int(item[8])
				if colorIndex < totalColors:
					processedItem['color'] = self.colors[colorIndex]
				else:
					processedItem['color'] = 'gray'
			else:
				processedItem['color'] = 'gray'
			processedData.append(processedItem)

		return processedData


	def getZone(self, numberOfBlocks):
		if not numberOfBlocks or not self.totalTextBlocks:
			return 0

		percentageOfNumberOfBlocks = (int(numberOfBlocks) * 100) / self.totalTextBlocks

		if percentageOfNumberOfBlocks >= 40:
			return 1

		if percentageOfNumberOfBlocks <= 0.10: 
			return 19

		if percentageOfNumberOfBlocks <= 0.20: 
			return 18

		if percentageOfNumberOfBlocks <= 0.30: 
			return 17

		if percentageOfNumberOfBlocks <= 0.40: 
			return 16

		if percentageOfNumberOfBlocks <= 0.50: 
			return 15

		if percentageOfNumberOfBlocks <= 0.60: 
			return 14

		if percentageOfNumberOfBlocks <= 0.70: 
			return 13

		if percentageOfNumberOfBlocks <= 0.80: 
			return 12

		if percentageOfNumberOfBlocks <= 0.90: 
			return 11

		if percentageOfNumberOfBlocks <= 1: 
			return 10

		if percentageOfNumberOfBlocks <= 2: 
			return 9

		if percentageOfNumberOfBlocks <= 3: 
			return 8

		if percentageOfNumberOfBlocks <= 4: 
			return 7
		
		if percentageOfNumberOfBlocks <= 5: 
			return 6
		
		if percentageOfNumberOfBlocks <= 10: 
			return 5
		
		if percentageOfNumberOfBlocks <= 15: 
			return 4
		
		if percentageOfNumberOfBlocks <= 20: 
			return 3
			
		return 2


