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

logger = CM.VLog('alg_ds')
logger_level = CM.VLog.DEBUG
allows_known_errors = False
show_cov = True
analyze_outps = False

#Data Structures
is_cov = lambda cov: (isinstance(cov, (set, frozenset)) and
                      all(isinstance(c, str) for c in cov))
def str_of_cov(cov):
    """
    >>> assert str_of_cov(set("L2 L1 L3".split())) == '(3) L1,L2,L3'
    """
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
    >>> print str_of_csetting(('x', frozenset(['1'])))
    x=1
    >>> print str_of_csetting(('x', frozenset(['3','1'])))
    x=1,3
    """
    assert is_csetting((k, vs)), (k, vs)
    
    return '{}={}'.format(k, str_of_valset(vs))


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

   >>> z3db = Z3DB(dom)
    >>> assert len(z3db) == len(dom) and set(z3db) == set(dom)

    >>> random.seed(0)
    >>> configs = dom.gen_configs_rand_smt(5, z3db)
    >>> print "\\n".join(map(str,configs))
    x=2 y=1 z=0 w=a
    x=1 y=1 z=0 w=b
    x=2 y=1 z=1 w=a
    x=2 y=1 z=2 w=a
    x=1 y=1 z=0 w=a

    >>> random.seed(0)
    >>> configs = dom.gen_configs_rand_smt(5, z3db, configs+configs)
    >>> print "\\n".join(map(str, configs))
    x=1 y=1 z=0 w=c
    x=2 y=1 z=0 w=b
    x=2 y=1 z=0 w=c
    x=2 y=1 z=2 w=c
    x=2 y=1 z=2 w=b

    >>> new_configs = dom.gen_configs_rand_smt(dom.siz, z3db, configs)
    >>> assert len(new_configs) == dom.siz - len(configs), (len(new_configs), dom.siz, len(configs))

    >>> configs = dom.gen_configs_rand_smt(dom.siz, z3db)
    >>> assert len(configs) == dom.siz    

    >>> configs = dom.gen_configs_rand_smt(dom.siz, z3db, configs)
    >>> assert not configs

    """
    def __init__(self, dom):
        OrderedDict.__init__(self, dom)
        
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
    
    #Methods to generate configurations
    def gen_configs_full(self, config_cls=None, z3db=None, constraints=True):#TODO kconfig_contraint
        if config_cls is None:
            config_cls = Config
        
        ns,vs = itertools.izip(*self.iteritems())
        configs = [config_cls(zip(ns, c)) for c in itertools.product(*vs)]
        return configs

    def gen_configs_tcover1(self, z3db, constraints, config_cls=None):
        """
        Return a set of tcover array of stren 1
        """
        if config_cls is None:
            config_cls = Config
            
        dom_used = dict((k, set(self[k])) for k in self)
        
        def mk():
            config = []
            for k in self:
                while True:
                    if k in dom_used:
                        v = random.choice(list(dom_used[k]))
                        dom_used[k].remove(v)
                        if not dom_used[k]:
                            dom_used.pop(k)
                    else:
                        v = random.choice(list(self[k]))

                    cc = z3.And(constraints, z3db[k][0] == z3db[k][1][str(int(v))])
                    if z3util.get_models(cc, 1):
                        break
                        
                config.append((k,v))

            return config_cls(config)

        configs = []
        while dom_used: configs.append(mk())
        return configs    

    def gen_configs_rand(self, rand_n, config_cls=None, z3db=None, constraints=True):#TODO kconfig_contraint
        assert 0 < rand_n <= self.siz, (rand_n, self.siz)

        if config_cls is None:
            config_cls = Config
            
        rgen = lambda: [(k,random.choice(list(self[k]))) for k in self]
        configs = list(set(config_cls(rgen()) for _ in range(rand_n)))
        return configs

    #generate configs using an SMT solver
    def config_of_model(self, model, config_cls):
        """
        Ret a config from a model
        """
        assert isinstance(model, dict), model
        assert config_cls, config_cls
            
        _f = lambda k: (model[k] if k in model
                        else random.choice(list(self[k])))
        config = config_cls((k,_f(k)) for k in self)
        return config
    
    def gen_configs_expr(self, expr, k, config_cls):
        """
        Return at most k configs satisfying expr
        """
        assert z3.is_expr(expr), expr
        assert k > 0, k
        assert config_cls, config_cls
        
        def _f(m):
            m = dict((str(v), str(m[v])) for v in m)
            return None if not m else self.config_of_model(m, config_cls)

        models = z3util.get_models(expr, k)
        assert models is not None, models  #z3 cannot solve this
        if not models:  #not satisfy
            return []
        else:
            assert len(models) >= 1, models
            configs = [_f(m) for m in models]
            return configs
        
    def gen_configs_exprs(self, yexprs, nexprs, k, config_cls):
        """
        Return a config satisfying yexprs but not nexprs
        """
        assert all(Z3DB.maybe_expr(e) for e in yexprs),yexprs
        assert all(Z3DB.maybe_expr(e) for e in nexprs),nexprs
        assert k > 0, k
        assert config_cls, config_cls

        if z3.solve:
            pass
        yexprs = [e for e in yexprs if e is not None]
        nexprs = [z3.Not(e) for e in nexprs if e is not None]
        exprs = yexprs + nexprs
        assert exprs, 'empty exprs'

        expr = exprs[0] if len(exprs)==1 else z3util.myAnd(exprs)
        return self.gen_configs_expr(expr, k, config_cls)


    def gen_configs_rand_smt(self, rand_n, z3db, existing_configs=[], config_cls=None):
        """
        Create rand_n uniq configs
        """
        assert 0 < rand_n <= self.siz, (rand_n, self.siz)
        assert isinstance(z3db, Z3DB)
        assert isinstance(existing_configs, list), existing_configs
            
        if config_cls is None:
            config_cls = Config

        exprs = []
            
        existing_configs = set(existing_configs)
        if existing_configs:
            exprs = [c.z3expr(z3db) for c in existing_configs]
            nexpr = z3util.myOr(exprs)
            configs = self.gen_configs_exprs([], [nexpr], 1, config_cls)
            if not configs:
                return []
            else:
                config = configs[0]
        else:
            configs = self.gen_configs_rand(1, config_cls)
            assert len(configs) == 1, configs
            config = configs[0]

        for _ in range(rand_n - 1):
            exprs.append(config.z3expr(z3db))
            nexpr = z3util.myOr(exprs)
            configs_ = self.gen_configs_exprs([], [nexpr], 1, config_cls)
            if not configs_:
                break
            config = configs_[0]            
            configs.append(config)
            
        return configs    
        
class Z3DB(dict):
    def __init__(self, dom):
        assert isinstance(dom, Dom)
        db = {}
        for k, vs in dom.iteritems():
            vs = sorted(list(vs))
            ttyp, tvals=z3.EnumSort(k,vs)
            rs = [vv for vv in zip(vs, tvals)]
            rs.append(('typ', ttyp))
            db[k] = (z3.Const(k, ttyp), dict(rs))
            print k, db[k][0], db[k][1]
        dict.__init__(self, db)
        
        
    @property
    def cache(self):
        try:
            return self._exprs_cache
        except AttributeError:
            self._exprs_cache = {}
            return self._exprs_cache

    @property
    def solver(self):
        try:
            solver = self._solver
            return solver
        except AttributeError:
            self._solver = z3.Solver()
            return self._solver

    def add(self, k, v):
        assert self.maybe_expr(v), v 
        self.cache[k] = v

    def expr_of_dict(self, d):
        #s=1 t=1 u=1 v=1 x=0 y=0 z=4
        assert all(k in self for k in d), (d, self)
        
        if d in self.cache:
            #print "hitme2"
            return self.cache[d]
            
        rs = []
        for k,v in d.iteritems():
            k_,d_ = self[k] #var and dict 
            rs.append(k_ == d_[v])
            
        expr = z3util.myAnd(rs)
        self.add(d, expr)
        return expr
    
    def expr_of_dict_dict(self, d, is_and):
        #s=0 t=0 u=0 v=0 x=1 y=1 z=0,1,2  =>  ... (z=0 or z=1 or z=2)
        assert all(k in self for k in d), (d, self)
        
        key = (d, is_and)
        if key in self.cache:
            return self.cache[key]
        
        rs = []
        for k, vs in d.iteritems():
            k_, vs_ = self[k]
            rs.append(z3util.myOr([k_ == vs_[v] for v in vs]))
            
        myf =  z3util.myAnd if is_and else z3util.myOr
        expr = myf(rs)
        
        self.add(key, expr)
        return expr

    @staticmethod
    def maybe_expr(expr):
        #not None => z3expr
        return expr is None or z3.is_expr(expr)
    

class Config(HDict):
    """
    >>> c = Config([('a', '1'), ('b', '0'), ('c', '1')])
    >>> print c
    a=1 b=0 c=1

    >>> dom = Dom([('a',frozenset(['1','2'])),\
    ('b',frozenset(['0','1'])),\
    ('c',frozenset(['0','1','2']))])
    >>> c.z3expr(Z3DB(dom))
    And(a == 1, b == 0, c == 1)
    """
    def __init__(self,config=HDict()):
        HDict.__init__(self, config)
        
        assert all(is_setting(s) for s in self.iteritems()), self

    def __str__(self, cov=None):
        assert cov is None or is_cov(cov), cov

        s =  ' '.join(map(str_of_setting,self.iteritems()))
        if cov:
            s = "{}: {}".format(s,str_of_cov(cov))
        return s

    def z3expr(self, z3db):
        assert isinstance(z3db, Z3DB)
        return z3db.expr_of_dict(self)

    @staticmethod
    def mk(n, f):
        """
        Try to create at most n configs by calling f().
        """
        assert isinstance(n, int) and n > 0, n
        assert callable(f), f
            
        pop = OrderedDict()
        for _ in range(n):
            c = f()
            c_iter = 0
            while c in pop and c_iter < 5:
                c_iter += 1
                c = f()

            if c not in pop:
                pop[c]=None
                

        assert len(pop) <= n, (len(pop), n)
        pop = pop.keys()
        return pop
    

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

    def add(self,sid, config):
        assert isinstance(sid, str),sid
        assert isinstance(config, Config),config
        
        super(Covs_d, self).add_set(sid, config)

class Configs_d(CustDict):
    """
    A mapping from config -> {covs}
    """
    def __setitem__(self, config, cov):
        assert isinstance(config, Config), config
        assert is_cov(cov), cov
        
        self.__dict__[config] = cov

    def __str__(self):
        ss = (c.__str__(self[c]) for c in self.__dict__)
        return '\n'.join("{}. {}".format(i+1, s) for i, s in enumerate(ss))


if __name__ == "__main__":
    import doctest
    doctest.testmod()

