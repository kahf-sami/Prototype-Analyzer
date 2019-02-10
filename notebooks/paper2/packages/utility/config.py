import os
from .constants import Constants
import configparser


class Config:


	class __Config:

		def __init__(self):
			self.constants = Constants()
			self.config = None
			self.projectId = None
			self.environment = None
			self.contentId = None
			self.important = False
			self.loadFromIni()


		def loadFromIni(self):
			self.config = configparser.ConfigParser()
			for file in os.listdir(self.constants.CONFIG_PATH):
				if file.endswith(".ini"):
					self.config.read(os.path.join(self.constants.CONFIG_PATH, file))
		
			return


		def setProjectId(self, projectId):
			self.projectId = projectId
			return

		def getProjectId(self):
			return self.projectId


		def setImportant(self, important):
			self.important = important
			return

		def getImportant(self):
			return self.important
			

		def setEnvironment(self, environment):
			self.environment = environment
			if not self.environment:
				self.environment = 'user'
			return


		def setContentId(self, contentid):
			self.contentId = contentid
			return


		def getDatabaseName(self):
			return 'sp_' +  self.projectId + '_' + self.environment 
		
		def getSection(self, key):
			return self.config[key]


		def get(self, section, key):
			return self.config[section][key]


		def getContentId(self):
			return self.contentId



	instance = None


	def __new__(cls): # __new__ always a classmethod
		if not Config.instance:
			Config.instance = Config.__Config()

		return Config.instance


