from collections import OrderedDict
class HDict(OrderedDict):
    @property
    def hcontent(self):
        try:
            return self._hcontent
        except AttributeError:
            self._hcontent = frozenset(self.iteritems())
            return self._hcontent
    
    def __hash__(self):
        try:
            return self._hash
        except AttributeError:
            self._hash = hash(self.hcontent)
            return self._hash

    def __eq__(self,o):return hash(self) == hash(o)
    def __ne__(self,o):return not self.__eq__(o)
