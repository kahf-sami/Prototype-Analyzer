from .config import Config
import math
import hashlib
import json
from .file import File
import os
import re

class Utility():


	def __init__(self):
		self.config = Config()
		return

	@staticmethod
	def printMatrix(matrixGenerator, rowNumber = None, columnNumber = None):
		if not matrixGenerator:
			return

		rowCount = 0
		
		stringToPrint = ''

		for row in matrixGenerator:
			if (rowNumber and rowNumber != rowCount):
				rowCount += 1
				continue

			columnCount = 0
			for column in row:
				if (columnNumber and (columnNumber == columnCount)) or (columnNumber == None):
					stringToPrint += str(column) + ' '
					columnCount += 1

			stringToPrint += "\n"

		print(stringToPrint)

	@staticmethod
	def stringToNumber(text):
		return int.from_bytes(text.encode(), 'little')


	@staticmethod
	def numberToString(number):
		return number.to_bytes(math.ceil(number.bit_length() / 8), 'little').decode()


	@staticmethod
	def getRowColumnOfScipySparseCsrCsrMatrix(matrix, rowNumber):
		print(matrix[rowNumber, :])


	""" return the list with duplicate elements removed """
	@staticmethod
	def unique(a):
		return list(set(a))


	""" return the intersection of two lists """
	@staticmethod
	def intersect(a, b):
		return list(set(a) & set(b))


	""" return the union of two lists """
	@staticmethod
	def union(a, b):
		return list(set(a) | set(b))


	@staticmethod
	def union(a, b):
		return list(set(a) | set(b))


	@staticmethod
	def getHash(data):
		m = hashlib.md5()
		
		if isinstance(data, list):
			data = ''.join(data)
		m.update(data.encode(encoding='UTF-8'))
		digest = m.hexdigest()

		asciiSum = 0
		for char in digest:
			asciiSum += ord(char)

		return asciiSum

	@staticmethod
	def getAsciiValue(data):
		asciiSum = 0
		for char in data:
			asciiSum += ord(char)

		return asciiSum

	@staticmethod
	def getStopWords():
		path = File.join(os.path.abspath(__file__ + "/../"), 'stopwords.txt')
		file = File(path)
		stopWords = re.split('[\n]', file.read())


		return stopWords


	@staticmethod
	def implode(terms, divider = ','):
		return divider.join(s for s in terms)

		



