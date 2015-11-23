import abc
import itertools
import random
import os.path
import tempfile
from collections import OrderedDict, MutableMapping

from vu_common import HDict
import vu_common as CM

import z3
import z3util

logger_level = CM.VLog.DEBUG
allows_known_errors = False
show_cov = True
analyze_outps = False

#Data Structures
class CustDict(MutableMapping):
    """
    MuttableMapping ex: https://stackoverflow.com/questions/21361106/how-would-i-implement-a-dict-with-abstract-base-classes-in-python
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self): self.__dict__ = {}
    def __len__(self): return len(self.__dict__)
    def __getitem__(self, key): return self.__dict__[key]
    def __iter__(self): return iter(self.__dict__)    
    def __setitem__(self, key, val): raise NotImplementedError("setitem")
    def __delitem__(self, key): raise NotImplementedError("delitem")
    def add_set(self, key, val):
        """
        For mapping from key to set
        """
        if key not in self.__dict__:
            self.__dict__[key] = set()
        self.__dict__[key].add(val)
        
is_cov = lambda cov: (isinstance(cov, set) and
                      all(isinstance(s, str) for s in cov))
def str_of_cov(cov):
    """
    >>> assert str_of_cov(set("L2 L1 L3".split())) == '(3) L1,L2,L3'
    """
    if __debug__:
        assert is_cov(cov),cov

    s = "({})".format(len(cov))
    if show_cov:
        s = "{} {}".format(s, ','.join(sorted(cov)))
    return s


is_setting = lambda (k, v): isinstance(k, str) and isinstance(v, str)
def str_of_setting((k, v)):
    """
    >>> print str_of_setting(('x','1'))
    x=1
    """
    if __debug__:
        assert is_setting((k, v)), (k, v)
        
    return '{}={}'.format(k, v)


is_valset = lambda vs: (isinstance(vs, frozenset) and vs and
                        all(isinstance(v, str) for v in vs))

def str_of_valset(s):
    """
    >>> str_of_valset(frozenset(['1','2','3']))
    '1,2,3'
    """
    return ','.join(sorted(s))

is_csetting = lambda (k,vs): isinstance(k, str) and is_valset(vs)

def str_of_csetting((k,vs)):
    """
    >>> print str_of_csetting(('x',frozenset(['1'])))
    x=1
    >>> print str_of_csetting(('x',frozenset(['3','1'])))
    x=1,3
    """
    if __debug__:
        assert is_csetting((k, vs)), (k, vs)
    
    return '{}={}'.format(k, str_of_valset(vs))

class Dom(OrderedDict):
    """
    >>> dom = Dom([('x',frozenset(['1','2'])),\
    ('y',frozenset(['1'])),\
    ('z',frozenset(['0','1','2'])),\
    ('w',frozenset(['a','b','c']))\
    ])

    >>> print dom
    4 vars and 18 pos configs
    1. x: (2) 1,2
    2. y: (1) 1
    3. z: (3) 0,1,2
    4. w: (3) a,b,c

    >>> assert dom.siz == len(dom.gen_configs_full()) == 18
    >>> assert dom.max_fsiz ==  3

    >>> random.seed(0)
    >>> configs = dom.gen_configs_rand(5)
    >>> print "\\n".join(map(str,configs))
    x=2 y=1 z=2 w=a
    x=2 y=1 z=2 w=b
    x=2 y=1 z=0 w=a
    x=1 y=1 z=0 w=a
    x=1 y=1 z=2 w=c

    >>> random.seed(0)
    >>> configs = dom.gen_configs_tcover1()
    >>> print "\\n".join(map(str,configs))
    x=2 y=1 z=0 w=a
    x=1 y=1 z=2 w=c
    x=1 y=1 z=1 w=b


    >>> assert len(dom.z3db) == len(dom) and set(dom.z3db) == set(dom)

    """
    def __init__(self,dom):
        OrderedDict.__init__(self, dom)
        
        if __debug__:
            assert self and all(is_csetting(s) for s in self.iteritems()), self

    def __str__(self):
        """
        """
        s = "{} vars and {} pos configs".format(len(self),self.siz)
        s_detail = '\n'.join("{}. {}: ({}) {}"
                             .format(i+1, k, len(vs), str_of_valset(vs))
                             for i, (k, vs) in enumerate(self.iteritems()))
        s = "{}\n{}".format(s, s_detail)
        return s

    @property
    def siz(self): return CM.vmul(len(vs) for vs in self.itervalues())

    @property
    def max_fsiz(self):
        """
        Size of the largest finite domain
        """
        return max(len(vs) for vs in self.itervalues())
    
    @property
    def z3db(self):
        z3db = dict()
        for k, vs in self.iteritems():
            vs = sorted(list(vs))
            ttyp, tvals=z3.EnumSort(k,vs)
            rs = [vv for vv in zip(vs, tvals)]
            rs.append(('typ', ttyp))
            z3db[k] = (z3.Const(k, ttyp),dict(rs))
        return z3db
    
    @classmethod
    def get_dom(cls, dom_file):
        """
        Read domain info from a file. 
        Also read default configs (.default*) if given
        """
        if __debug__:
            assert os.path.isfile(dom_file), dom_file

        def get_lines(lines):
            rs = (line.split() for line in lines)
            rs = [(parts[0], frozenset(parts[1:])) for parts in rs]
            return rs

        dom = cls(get_lines(CM.iread_strip(dom_file)))

        #default configs
        dom_dir = os.path.dirname(dom_file)
        configs = [os.path.join(dom_dir, f) for f in os.listdir(dom_dir)
                   if dom_file in f and '.default' in f]

        configs = [dict(get_lines(CM.iread_strip(f))) for f in configs
                   if os.path.isfile(f)]
        configs = [[(k, list(c[k])[0]) for k in dom] for c in configs]
        return dom, configs

    #Methods to generate configurations
    def gen_configs_full(self, config_cls=None):
        if config_cls is None:
            config_cls = Config
        
        ns,vs = itertools.izip(*self.iteritems())
        configs = [config_cls(zip(ns, c)) for c in itertools.product(*vs)]
        return configs

    def gen_configs_rand(self, rand_n, config_cls=None):
        if __debug__:
            assert 0 < rand_n <= self.siz, (rand_n, self.siz)

        if config_cls is None:
            config_cls = Config
            
        rgen = lambda: [(k,random.choice(list(self[k]))) for k in self]
        configs = list(set(config_cls(rgen()) for _ in range(rand_n)))
        return configs

    def gen_configs_tcover1(self, config_cls=None):
        """
        Return a set of tcover array of stren 1
        """
        if config_cls is None:
            config_cls = Config
        dom_used = dict((k,set(self[k])) for k in self)

        def mk():
            config = []
            for k in self:
                if k in dom_used:
                    v = random.choice(list(dom_used[k]))
                    dom_used[k].remove(v)
                    if not dom_used[k]:
                        dom_used.pop(k)
                else:
                    v = random.choice(list(self[k]))

                config.append((k,v))
            return config_cls(config)

        configs = []
        while dom_used: configs.append(mk())
        return configs    


class Config(HDict):
    """
    >>> c = Config([('a', '1'), ('b', '0'), ('c', '1')])
    >>> print c
    a=1 b=0 c=1

    >>> dom = Dom([('a',frozenset(['1','2'])),\
    ('b',frozenset(['0','1'])),\
    ('c',frozenset(['0','1','2']))])
    >>> c.z3expr(dom.z3db)
    And(a == 1, b == 0, c == 1)
    """
    def __init__(self,config=HDict()):
        HDict.__init__(self, config)
        
        if __debug__:
            assert all(is_setting(s) for s in self.iteritems()), self

    def __str__(self,cov=None):
        if __debug__:
            assert cov is None or is_cov(cov), cov

        s =  ' '.join(map(str_of_setting,self.iteritems()))
        if cov:
            s = "{}: {}".format(s,str_of_cov(cov))
        return s

    def z3expr(self, z3db):
        if __debug__:
            
            #assert len(self) == len(z3db), (len(self), len(z3db))
            #not true when using partial config from Otter
            assert all(e in z3db for e in self), (self, z3db)

        f = []
        for vn,vv in self.iteritems():
            vn_,vs_ = z3db[vn]
            f.append(vn_==vs_[vv])

        return z3util.myAnd(f)    

class Covs_d(CustDict):
    """
    A mapping from sid -> {configs}

    >>> c1 = Config([('a', '0'), ('b', '0'), ('c', '0')])
    >>> c2 = Config([('a', '0'), ('b', '0'), ('c', '1')])
    >>> covs_d = Covs_d()
    >>> assert 'l1' not in covs_d
    >>> covs_d.add('l1',c1)
    >>> covs_d.add('l1',c2)
    >>> assert 'l1' in covs_d
    >>> assert covs_d['l1'] == set([c1,c2])
    """

    def add(self,sid,config):
        if __debug__:
            assert isinstance(sid, str),sid
            assert isinstance(config, Config),config
        super(Covs_d, self).add_set(sid, config)

class Configs_d(CustDict):
    def __setitem__(self, config, cov):
        if __debug__:
            assert isinstance(config, Config), config
            assert is_cov(cov), cov
        self.__dict__[config] = cov

    def __str__(self):
        ss = (c.__str__(self[c]) for c in self.__dict__)
        return '\n'.join("{}. {}".format(i+1, s) for i, s in enumerate(ss))


def eval_configs(configs, get_cov):
    if __debug__:
        assert (isinstance(configs, list) and
                all(isinstance(c, Config) for c in configs)
                and configs), configs
        assert callable(get_cov), get_cov

    cache = set()
    results = []
    for c in configs:
        if c in cache:
            continue
        cache.add(c)
        
        sids, outps = get_cov(c)
        rs = outps if analyze_outps else sids
        if not rs:
            logger.warn("config {} generates nothing".format(c))
        results.append((c,rs))
        
    return results

if __name__ == "__main__":
    import doctest
    doctest.testmod()

