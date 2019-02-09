import os
import shutil

class Directory():


	def __init__(self, path = None):
		self.path = path
		self.name = None

		return


	def setPath(self, path):
		self.path = path

		return

	def exists(self):
		return os.path.exists(self.path)


	def create(self, path = None):
		if(self.exists()):
			return

		if not path:
			path = self.path
		
		subPath = os.path.dirname(path)

		if not os.path.exists(subPath):
			self.create(subPath)

		if not os.path.exists(path):
			os.mkdir(path)

		return

	def remove(self, path = None):
		if(not self.exists()):
			return

		if not path:
			path = self.path

		for root, dirs, files in os.walk(self.path, topdown=False):
			if files:
				for name in files:
					os.remove(os.path.join(root, name))
			if dirs:
				for name in dirs:
					os.rmdir(os.path.join(root, name))

		return
					
		

	def scan(self):
		return next(os.walk(self.path))[2]