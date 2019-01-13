import sys, getopt, re

class Script():


   def __init__(self, args):
      self.args = args
      self.options = []
      self.params = {}
      return


   def addOption(self, key):
      if key:
         self.options = key
      return


   def getParams(self):
      return self.params


   def getParam(self, key):
      if key in self.params.keys():
         return self.params[key]
      return None


   def loadParams(self):
      lastKey = None
      for term in self.args:
         if re.search('^--', term):
            if lastKey and (lastKey not in self.params.keys()):
                self.params[lastKey] = 1
            lastKey = term[2:]
         else:
            self.params[lastKey] = term

      if lastKey not in self.params.keys():
         self.params[lastKey] = 1

      return self.params