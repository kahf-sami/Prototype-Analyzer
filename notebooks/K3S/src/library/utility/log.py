from .config import Config
import logging
import os

class Log():

	def __init__(self):
		self.config = Config()
		self.logPath = self.config.LOG_LOCATION
		self.fileName = 'general.log'
		self.setLogger()

	def setName(self, name):
		self.fileName = name;

	def setPath(self, path):
		self.logFilePath = path

	def setLogger(self):
		logFilePath = os.path.join(self.logPath, self.fileName)
		self.logger = logging
		self.logger.basicConfig(filename = logFilePath, level = logging.DEBUG, 
			format = '%(levelname)s:%(asctime)s:%(message)s', datefmt = '%m/%d/%Y %I:%M:%S %p')


	def getLogger(self):
		return self.logger

	def add(self, message, type = 'info'):
		if type == 'debug':
			self.logger.debug(message)
		elif type == 'warn':
			self.logger.warning(message)
		elif type == 'error':
			self.logger.error(message)
		elif type == 'critical':
			self.logger.critical(message)
		else:
			self.logger.info(message)

	def debug(self, message):
		self.add(message, 'debug')

	def error(self, message):
		self.add(message, 'error')

	def critical(self, message):
		self.add(message, 'critical')

	def warn(self, message):
		self.add(message, 'warn')