import datetime

class Timer():

	allTimer = []

	def __init__(self, identifier = 'default'):
		allTimer[identifier]['start'] = None
		allTimer[identifier]['end'] = None

	@staticmethod
	def start(self, identifier):
		allTimer[identifier]['start'] = datetime.datetime.now()


	@staticmethod
	def stop(self, identifier):
		allTimer[identifier]['end'] = datetime.datetime.now()

	@staticmethod
	def getRequiredTime(self, identifier):
		return self.getDifference(allTimer[identifier]['end'], allTimer[identifier]['start'])

	@staticmethod
	def getDifference(self, dateTime1, dateTime2):
		elapsedTime = dateTime1 - dateTime2
		datetime.timedelta(0, 125, 749430)
		return divmod(elapsedTime.total_seconds(), 60)

	@staticmethod
	def get(self):
		return datetime.datetime.now()


	@staticmethod
	def getFormatedDate(self, datetimeValue = None):
		if not datetimeValue:
			datetimeValue = self.get()
		return datetimeValue.strftime("%y-%m-%d-%H-%M-%s")