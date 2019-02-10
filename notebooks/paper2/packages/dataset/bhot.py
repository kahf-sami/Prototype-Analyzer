from . import Dataset

class BHOT(Dataset):


	def __init__(self, path):
		super().__init__(path, 'bhot')
		return