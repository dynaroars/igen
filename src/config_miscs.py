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


class DTuple(tuple):
    @property
    def d(self):
        try:
            return self._d
        except AttributeError:
            self._d = dict(self)
            return self._d

    def __getitem__(self,x):
        return self.d[x]
