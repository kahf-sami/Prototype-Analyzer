import os


class Constants:


	def __init__(self):
		self.ROOT_PATH = os.path.abspath(__file__ + "/../../../")
		self.CONFIG_PATH = os.path.join(self.ROOT_PATH, 'config')
		self.LIBRARY_PATH = os.path.join(self.ROOT_PATH, 'library')
		self.DATA_PATH = os.path.join(self.ROOT_PATH, 'data')
		self.IMAGE_PATH = os.path.join(self.DATA_PATH, 'image')
		self.LOG_LOCATION = os.path.join(self.ROOT_PATH, 'logs')


