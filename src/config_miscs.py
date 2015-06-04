import abc

import itertools
import os.path
import random
from functools import total_ordering
from collections import OrderedDict

import z3
import z3util
import vu_common as CM

logger = CM.VLog('vconfig_miscs')
logger.level = CM.VLog.DEBUG
CM.VLog.PRINT_TIME = True

print_cov = True

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

    #todo: do I need these 2?
    def __eq__(self,o):return hash(self) == hash(o)
    def __ne__(self,o):return not self.__eq__(o)



class Miscs(object):
    @staticmethod
    def mk_tcover(vals,cover_siz,tseed,tmpdir):
            """
            Call external program to generate t-covering arrays

            >>> CM.VLog.PRINT_TIME = False
            >>> Miscs.mk_tcover([[1,2],[1,2,3],[4,5]], 2, 0,'/tmp/')
            vconfig_miscs:Debug:cmd: cover -r 0 -s 0 -o  /tmp/casaOutput.txt /tmp/casaInput.txt
            [[1, 1, 5], [2, 1, 4], [2, 2, 5], [1, 2, 4], [2, 3, 4], [1, 3, 5]]

            #linux gives this
            [[2, 3, 4], [1, 3, 5], [1, 1, 4], [2, 1, 5], [2, 2, 4], [1, 2, 5]]
            """
            if CM.__vdebug__:
                assert tseed >= 0 , tseed

            if cover_siz > len(vals):
                cover_siz = len(vals)

            infile = os.path.join(tmpdir,"casaInput.txt")
            outfile = os.path.join(tmpdir,"casaOutput.txt")

            #create input
            in_contents = "{}\n{}\n{}".format(
                cover_siz,len(vals),' '.join(map(str,map(len,vals))))
            CM.vwrite(infile,in_contents)

            #exec cover on file
            copt = "-r 0 -s {} -o ".format(tseed)
            cmd ="cover {} {} {}".format(copt,outfile,infile)
            logger.debug("cmd: {}".format(cmd))
            try:
                _,rs_err = CM.vcmd(cmd)
                assert len(rs_err) == 0, rs_err
            except:
                logger.error("cmd '{}' failed".format(cmd))


            #read output
            vals_ = CM.vflatten(vals)
            lines = [l.strip() for l in CM.iread(outfile)]
            vs = []
            for l in lines[1:]: #ignore size of covering array
                idxs = map(int,l.split())
                assert len(idxs) == len(vals)
                vs.append([vals_[i] for i in idxs])
            return vs


is_cov = lambda cov: (isinstance(cov,set) and
                      all(isinstance(sid,str) for sid in cov))

def str_of_cov(cov):
    if CM.__vdebug__:
        assert is_cov(cov),cov
    
    return ','.join(sorted(cov)) if print_cov else str(len(cov))

is_valset = lambda vs: (isinstance(vs,frozenset) and vs and
                        all(isinstance(v,str) for v in vs))
def str_of_valset(s):
    if CM.__vdebug__:
        assert is_valset(s),s

    return ','.join(sorted(s))


is_setting = lambda (vn,vv): isinstance(vn,str) and isinstance(vv,str)

def str_of_setting((vn,vv)):
    if CM.__vdebug__:
        assert is_setting((vn,vv)),(vn,vv)

    return '{}={}'.format(vn,vv)

is_csetting = lambda (vn,vs): isinstance(vn,str) and is_valset(vs)

def str_of_csetting((vn,vs)):
    if CM.__vdebug__:
        assert is_csetting((vn,vs)), (vn,vs)

    return '{}={}'.format(vn,str_of_valset(vs))

class Dom(OrderedDict):
    """
    >>> dom = Dom([('listen', frozenset(['1', '0'])), ('timeout', frozenset(['1', '0'])), ('ssl', frozenset(['1', '0'])), ('local', frozenset(['1', '0'])), ('anon', frozenset(['1', '0'])), ('log', frozenset(['1', '0'])), ('chunks', frozenset(['0', '65536', '4096', '2048']))])
    >>> print dom
    listen: 0,1
    timeout: 0,1
    ssl: 0,1
    local: 0,1
    anon: 0,1
    log: 0,1
    chunks: 0,2048,4096,65536

    >>> assert dom.siz == 256

    >>> dom.z3db
    OrderedDict([('listen', (listen, {'1': 1, '0': 0, 'typ': listen})), ('timeout', (timeout, {'1': 1, '0': 0, 'typ': timeout})), ('ssl', (ssl, {'1': 1, '0': 0, 'typ': ssl})), ('local', (local, {'1': 1, '0': 0, 'typ': local})), ('anon', (anon, {'1': 1, '0': 0, 'typ': anon})), ('log', (log, {'1': 1, '0': 0, 'typ': log})), ('chunks', (chunks, {'0': 0, '65536': 65536, 'typ': chunks, '4096': 4096, '2048': 2048}))])

    >>> c1 = Config([('listen','1'),('timeout','1'),('ssl','1'),('local','0'),('anon','0'),('log','1'),('chunks','2048')])
    >>> print c1
    listen=1 timeout=1 ssl=1 local=0 anon=0 log=1 chunks=2048
    >>> print c1.z3expr(dom)
    And(listen == 1,
        timeout == 1,
        ssl == 1,
        local == 0,
        anon == 0,
        log == 1,
        chunks == 2048)

    >>> c2 = Config([('listen','0'),('timeout','1'),('ssl','1'),('local','0'),('anon','0'),('log','0'),('chunks','0')])
    >>> configs = Configs([c1,c2])
    >>> print configs
    0. listen=1 timeout=1 ssl=1 local=0 anon=0 log=1 chunks=2048
    1. listen=0 timeout=1 ssl=1 local=0 anon=0 log=0 chunks=0

    >>> print configs.z3expr(dom)
    Or(And(listen == 1,
           timeout == 1,
           ssl == 1,
           local == 0,
           anon == 0,
           log == 1,
           chunks == 2048),
       And(listen == 0,
           timeout == 1,
           ssl == 1,
           local == 0,
           anon == 0,
           log == 0,
           chunks == 0))

    >>> assert len(dom.gen_configs_full()) == 256

    >>> random.seed(0)
    >>> assert len(dom.gen_configs_rand(256)) == 256

    >>> random.seed(0)
    >>> assert len(dom.gen_configs_rand(200)) == 140

    >>> random.seed(0)
    >>> assert len(dom.gen_configs_tcover1()) == 4
    
    >>> CM.VLog.PRINT_TIME = False
    >>> assert len(dom.gen_configs_tcover(3,1,'/tmp/')) == 24
    vconfig_miscs:Debug:cmd: cover -r 0 -s 1 -o  /tmp/casaOutput.txt /tmp/casaInput.txt


    """
    def __init__(self,d):
        OrderedDict.__init__(self,d)

        if CM.__vdebug__:
            assert (self and
                    all(isinstance(vn,str) and
                        is_valset(vs) for vn,vs in self.iteritems()))
        
    def __str__(self):
        return '\n'.join("{}: {}".format(k,str_of_valset(vs))
                         for k,vs in self.iteritems())
    @property
    def siz(self):
        return CM.vmul(len(vs) for vs in self.itervalues())

    @staticmethod
    def get_dom(dom_file):
        
        def get_lines(lines):
            rs = (line.split() for line in lines)
            rs = ((parts[0],frozenset(parts[1:])) for parts in rs)
            return rs

        dom_file = os.path.realpath(dom_file)
        dom = Dom(get_lines(CM.iread_strip(dom_file)))

        config_default = None
        dom_file_default = os.path.realpath(dom_file+'.default')
        if os.path.isfile(dom_file_default):
            rs = get_dom_lines(CM.iread_strip(dom_file_default))
            config_default = Config((k,list(v)[0]) for k,v in rs)

        return dom,config_default

    @property
    def z3db(self):
        z3db = OrderedDict()  #{'x':(x,{'true':True})}
        for k,vs in self.iteritems():
            vs = sorted(list(vs))
            ttyp,tvals=z3.EnumSort(k,vs)
            rs = []
            for v,v_ in zip(vs,tvals):
                rs.append((v,v_))
            rs.append(('typ',ttyp))

            z3db[k]=(z3.Const(k,ttyp),dict(rs))
        return z3db


    #generate configs
    def gen_configs_full(self):
        ns,vs = itertools.izip(*self.iteritems())
        configs = Configs(Config(zip(ns,c)) for c in itertools.product(*vs))
        return configs

    def gen_configs_rand(self,n):
        if CM.__vdebug__:
            assert n > 0, n

        if n >= self.siz:
            return self.gen_configs_full()
        
        rgen = lambda: [(k,random.choice(list(self[k]))) for k in self]
        configs =  Configs.uniq(Config(rgen()) for _ in range(n))
        return configs

    def gen_configs_tcover1(self):
        """
        Return a set of tcover array of strength 1
        """
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
            return Config(config)

        configs = Configs()
        while dom_used:
            configs.append(mk())

        return configs

    def gen_configs_tcover(self,cover_siz,tseed,tmpdir):
            
        vals = map(list,self.values())
        vs = Miscs.mk_tcover(vals,cover_siz,tseed,tmpdir)
        configs = Configs(Config(zip(self.keys(),vs_)) for vs_ in vs)

        return configs


class Config(HDict):
    """
    >>> dom = Dom([('listen', frozenset(['1', '0'])), ('timeout', frozenset(['1', '0'])), ('ssl', frozenset(['1', '0'])), ('local', frozenset(['1', '0'])), ('anon', frozenset(['1', '0'])), ('log', frozenset(['1', '0'])), ('chunks', frozenset(['0', '65536', '4096', '2048']))])
    >>> c1 = Config([('listen','1'),('timeout','1'),('ssl','1'),('local','0'),('anon','0'),('log','1'),('chunks','2048')])
    >>> print c1
    listen=1 timeout=1 ssl=1 local=0 anon=0 log=1 chunks=2048
    >>> print c1.z3expr(dom)
    And(listen == 1,
        timeout == 1,
        ssl == 1,
        local == 0,
        anon == 0,
        log == 1,
        chunks == 2048)

    >>> assert c1.c_implies({'listen':'1','chunks':frozenset(['0','2048'])})
    >>> assert not c1.c_implies({'listen':'1','chunks':frozenset(['0','4096'])})
    >>> assert not c1.c_implies({'listen':'3','chunks':frozenset(['0','2048'])})

    """
    def __init__(self,c=HDict()):
        HDict.__init__(self,c)

        if CM.__vdebug__:
            assert all(is_setting(s) for s in self.iteritems()), self

    def __str__(self):
        return ' '.join(map(str_of_setting,self.iteritems()))

    def z3expr(self,dom):
        if CM.__vdebug__:
            assert isinstance(dom,Dom), dom
            assert len(self) == len(dom), (len(self), len(dom))

        f = []
        for vn,vv in self.iteritems():
            vn_,vs_ = dom.z3db[vn]
            f.append(vn_==vs_[vv])

        z3expr = z3util.myAnd(f)
        return z3expr

    def c_implies(self,d): return all(self[k] in d[k] for k in d)
    def d_implies(self,d): return any(self[k] in d[k] for k in d)


class Configs(list):
    """
    >>> dom = Dom([('listen', frozenset(['1', '0'])), ('timeout', frozenset(['1', '0'])), ('ssl', frozenset(['1', '0'])), ('local', frozenset(['1', '0'])), ('anon', frozenset(['1', '0'])), ('log', frozenset(['1', '0'])), ('chunks', frozenset(['0', '65536', '4096', '2048']))])
    >>> c1 = Config([('listen','1'),('timeout','1'),('ssl','1'),('local','0'),('anon','0'),('log','1'),('chunks','2048')])
    >>> c2 = Config([('listen','0'),('timeout','1'),('ssl','1'),('local','0'),('anon','0'),('log','0'),('chunks','0')])
    >>> configs = Configs([c1,c2])

    >>> print configs
    0. listen=1 timeout=1 ssl=1 local=0 anon=0 log=1 chunks=2048
    1. listen=0 timeout=1 ssl=1 local=0 anon=0 log=0 chunks=0

    >> assert configs.c_implies({'timeout':frozenset(['1']),'chunks':frozenset(['0','2048'])})

    >>> print configs.z3expr(dom)
    Or(And(listen == 1,
           timeout == 1,
           ssl == 1,
           local == 0,
           anon == 0,
           log == 1,
           chunks == 2048),
       And(listen == 0,
           timeout == 1,
           ssl == 1,
           local == 0,
           anon == 0,
           log == 0,
           chunks == 0))

    >>> assert len(Configs.uniq([c1,c2,c2,c1])) == 2

    """
    def __init__(self,configs=[]):
        list.__init__(self,configs)

        if CM.__vdebug__:
            assert all(isinstance(c,Config) for c in self), self

    def __str__(self,covs=None):
        if CM.__vdebug__:
            assert (covs is None or 
                    (len(covs) == len(self) and
                     all(is_cov(cov) for cov in covs))), covs

        if  covs:
            return '\n'.join("{}. {}: {}"
                             .format(i+1,c,str_of_cov(cov))
                             for i,(c,cov) in enumerate(zip(self,covs)))
        else:
            return '\n'.join("{}. {}".format(i,c)
                             for i,c in enumerate(self))

    def implies(self,d):
        return all(c.c_implies(d) for c in self)
    
    def z3expr(self,dom):
        return z3util.myOr([c.z3expr(dom) for c in self])

    @staticmethod
    def uniq(configs):
        return Configs(set(configs))
    
class Core(HDict):
    """
    >>> c1 = Core()
    >>> print c1
    true

    >>> c3 = Core([('listen',frozenset(['1'])),('ssl',frozenset(['1'])),('anon',frozenset(['0'])),('chunks',frozenset(['2048','0']))])
    >>> print c3
    listen=1 ssl=1 anon=0 chunks=0,2048

    >>> assert not c1
    >>> assert c3

    >>> c3b = Core([('listen',frozenset(['1'])),('ssl',frozenset(['1'])),('anon',frozenset(['0'])),('chunks',frozenset(['2048','0']))])

    >>> dom = Dom([('listen', frozenset(['1', '0'])), ('timeout', frozenset(['1', '0'])), ('ssl', frozenset(['1', '0'])), ('local', frozenset(['1', '0'])), ('anon', frozenset(['1', '0'])), ('log', frozenset(['1', '0'])), ('chunks', frozenset(['0', '65536', '4096', '2048']))])

    >>> assert c1.z3expr(dom) is z3util.TRUE
    >>> print c3.z3expr(dom)
    And(listen == 1,
        ssl == 1,
        anon == 0,
        Or(chunks == 0, chunks == 2048))
    """
    def __init__(self,c=HDict()):
        HDict.__init__(self,c)
        
        if CM.__vdebug__:
            assert all(is_csetting(s) for s in self.iteritems()), self
        
    def __str__(self):
        if not self:
            return "true"  #no constraint
        

    
    def neg(self,dom): return Core((k,dom[k]-self[k]) for k in self)
        

    def z3expr(self,dom,myf=z3util.myAnd):
        if not self:
            return z3util.TRUE
        else:
            f = []
            for vn,vs in self.iteritems():
                vn_,vs_ = dom.z3db[vn]
                f.append(z3util.myOr([vn_ == vs_[v] for v in vs]))

            return myf(f)


class CDCore(tuple):
    """
    >>> dom = Dom([('listen', frozenset(['1', '0'])), ('timeout', frozenset(['1', '0'])), ('ssl', frozenset(['1', '0'])), ('local', frozenset(['1', '0'])), ('anon', frozenset(['1', '0'])), ('log', frozenset(['1', '0'])), ('chunks', frozenset(['0', '65536', '4096', '2048']))])

    >>> c = PCore.mk_default()
    >>> print c
    p: c(None), d(None)

    >>> assert c.z3expr(dom) is None

    >>> c = PCore.mk(Core([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))]), None)
    >>> print c
    p: c(listen=1 timeout=1), d(None)
    
    >>> print c.z3expr(dom)
    And(listen == 1, timeout == 1)

    >>> print CDCore.mk(Core(),None).z3expr(dom)
    True


    >>> c = PCore.mk(\
    Core([('listen', frozenset(['1']))]),\
    Core([('timeout', frozenset(['1'])), ('anon', frozenset(['1']))]))
    >>> print c.z3expr(dom)
    And(listen == 1, Or(timeout == 0, anon == 0))

    """
    __metaclass__ = abc.ABCMeta
    
    def __init__(self,(cc,dc)):
        if CM.__vdebug__:
            assert cc is None or isinstance(cc,Core), cc
            assert dc is None or isinstance(dc,Core), dc
        
        tuple.__init__(self,(cc,dc))
        
        
    @property
    def conj(self): return self[0]
    @property
    def disj(self): return self[1]

    @abc.abstractmethod
    def __str__(self):
        return "c({}), d({})".format(self.conj,self.disj)

    def z3expr(self,dom):
        conj = lambda: self.conj.z3expr(dom)
        disj = lambda: self.disj.neg(dom).z3expr(dom,myf=z3util.myOr)

        if self.conj is None: # no data
            return None
        elif self.conj:
            if self.disj:
                return z3.simplify(z3.And(conj(),disj()))
            else:
                return conj()
        else:
            if self.disj:
                return disj()
            else:
                return z3util.TRUE
            
    @classmethod
    def mk(cls,c,d): return cls((c,d))

    @classmethod
    def mk_default(cls): return cls.mk(None,None)

class PCore(CDCore):
    def __str__(self): return "p: {}".format(super(PCore,self).__str__())
    
class NCore(CDCore):
    def __str__(self): return "n: {}".format(super(NCore,self).__str__())


class PNCore(tuple):
    """
    >>> dom = Dom([('listen', frozenset(['1', '0'])), ('timeout', frozenset(['1', '0'])), ('ssl', frozenset(['1', '0'])), ('local', frozenset(['1', '0'])), ('anon', frozenset(['1', '0'])), ('log', frozenset(['1', '0'])), ('chunks', frozenset(['0', '65536', '4096', '2048']))])

    >>> print PNCore.mk_default()
    p: c(None), d(None), n: c(None), d(None)

    >>> pcc = Core([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))])
    >>> pdc = None
    >>> ncc = Core()
    >>> ndc = Core([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))])

    >>> c = PNCore.mk(pcc,pdc,ncc,ndc)
    >>> print c
    p: c(listen=1 timeout=1), d(None), n: c(true), d(listen=1 timeout=1)

    >>> print c.z3simplify(dom)
    And(listen == 1, timeout == 1)

    >>> assert len(c.settings) == 2


    >>> pcc = Core([('listen', frozenset(['0']))])
    >>> pdc = None
    >>> ncc = Core([('listen', frozenset(['1']))])
    >>> ndc = None
    >>> c = PNCore.mk(pcc,pdc,ncc,ndc)
    >>> print c.z3simplify(dom)
    listen == 0

    >>> assert len(c.settings) == 2

    >>> c = PNCore.mk(Core(),None,None,Core())
    >>> print c.z3simplify(dom)
    True

    >>> assert len(c.settings) == 0

    >>> dom = Dom([('a', frozenset(['1', '0'])), ('b', frozenset(['1', '0'])), ('c', frozenset(['1', '0'])), ('d', frozenset(['0', '1', '2', '3']))])    
    >>> L1_configs = [\
    Config([('a', '1'), ('b', '0'), ('c', '1'), ('d', '2')]), \
    Config([('a', '1'), ('b', '1'), ('c', '1'), ('d', '0')])]

    >>> L2_configs = [Config([('a', '1'), ('b', '0'), ('c', '0'), ('d', '1')])]

    >>> pcc = Core([('a', frozenset(['1'])),('c', frozenset(['1'])),('d', frozenset(['0','2']))])
    >>> pdc = None
    >>> ncc = Core([('a', frozenset(['1'])),('b', frozenset(['0'])),('c', frozenset(['0'])),('d', frozenset(['1']))])
    >>> ndc = None
    >>> c = PNCore.mk(pcc,pdc,ncc,ndc)



    >>> c = PNCore.mk(\
    Core([('a', frozenset(['1']))]),\
    Core([('b', frozenset(['1'])), ('c', frozenset(['1']))]),\
    Core(),\
    Core([('a', frozenset(['1']))]))
    >>> print c.z3simplify(dom)
    And(a == 1, Or(b == 0, c == 0))
    """
    def __init__(self,(pc,nc)):
        if CM.__vdebug__:
            assert isinstance(pc,PCore) and isinstance(nc,NCore), (pc,nc)

        tuple.__init__(self,(pc,nc))

    @property
    def pcore(self): return self[0]
    @property
    def ncore(self): return self[1]
        
    def __str__(self):
        return "{}, {}".format(self.pcore,self.ncore)

    def z3simplify(self,dom):
        """
        Compare between pcore and ncore and return the more precise one.
        """
        z3_pc = self.pcore.z3expr(dom)
        if z3_pc:
            z3_pc = z3.simplify(z3_pc)

        z3_nc = self.ncore.z3expr(dom)
        if z3_nc:
            z3_nc = z3.Not(z3_nc)
            z3_nc = z3.simplify(z3_nc)

        if z3_pc is None:
            return z3_nc
        elif z3_nc is None:
            return z3_pc
        else:
            if z3util.is_tautology(z3.Implies(z3_pc,z3_nc)):
                return z3_pc
            elif z3util.is_tautology(z3.Implies(z3_nc,z3_pc)):
                return z3_nc
            else:
                raise AssertionError("inconsistent ? {}".format(self))
            
    @property
    def settings(self):
        try:
            return self._settings
        except AttributeError:
            cores = [self.pcore.conj,self.pcore.disj,
                     self.ncore.conj,self.ncore.disj]
            cores = [c for c in cores if c]
            settings = []
            for c in cores:
                settings.extend(c.hcontent)
            self._settings = set(settings)
            return self._settings

    @property
    def strength(self): return len(self.settings)

    @staticmethod
    def mk(pc,pd,nc,nd):
        return PNCore((PCore.mk(pc,pd),NCore.mk(nc,nd)))
    @staticmethod
    def mk_default():
        return PNCore((PCore.mk_default(),NCore.mk_default()))

    
class CORES_D(OrderedDict):
    """
    >>> c1 = PNCore.mk(Core([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))]),None,Core(),Core([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))]))
    >>> c2 = PNCore.mk(Core([('listen', frozenset(['1'])), ('timeout', frozenset(['0']))]), None, Core(), Core([('listen', frozenset(['1'])), ('timeout', frozenset(['0']))]))
    >>> c3 = PNCore.mk(Core([('listen', frozenset(['0']))]), None, Core([('listen', frozenset(['1']))]), None)
    >>> c4 = PNCore.mk(Core(), None, None, Core())
    >>> c5 = PNCore.mk(Core(), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0']))]), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0']))]), None)
    >>> c6 = PNCore.mk(Core([('ssl', frozenset(['0'])), ('local', frozenset(['0']))]), None, Core(), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0']))]))
    >>> c7 = PNCore.mk(Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1']))]), None, Core(), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1']))]))
    >>> c8 = PNCore.mk(Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1'])), ('log', frozenset(['1'])), ('chunks', frozenset(['4096', '2048']))]), None, Core(), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1'])), ('log', frozenset(['1'])), ('chunks', frozenset(['4096', '2048']))]))
    >>> c9 = PNCore.mk(Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1'])), ('log', frozenset(['0'])), ('chunks', frozenset(['4096', '2048']))]), None, Core(), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1'])), ('log', frozenset(['0'])), ('chunks', frozenset(['4096', '2048']))]))
    >>> cores = [c1,c2,c3,c4,c5,c6,c7,c8,c9]
    >>> sids = ['L1','L2','L3','L4','L5','L6','L7','L8','L9']
    >>> cores_d = CORES_D(zip(sids,cores))
    >>> print cores_d
    1. L1: p: c(listen=1 timeout=1), d(None), n: c(true), d(listen=1 timeout=1)
    2. L2: p: c(listen=1 timeout=0), d(None), n: c(true), d(listen=1 timeout=0)
    3. L3: p: c(listen=0), d(None), n: c(listen=1), d(None)
    4. L4: p: c(true), d(None), n: c(None), d(true)
    5. L5: p: c(true), d(ssl=0 local=0), n: c(ssl=0 local=0), d(None)
    6. L6: p: c(ssl=0 local=0), d(None), n: c(true), d(ssl=0 local=0)
    7. L7: p: c(ssl=0 local=0 anon=1), d(None), n: c(true), d(ssl=0 local=0 anon=1)
    8. L8: p: c(ssl=0 local=0 anon=1 log=1 chunks=2048,4096), d(None), n: c(true), d(ssl=0 local=0 anon=1 log=1 chunks=2048,4096)
    9. L9: p: c(ssl=0 local=0 anon=1 log=0 chunks=2048,4096), d(None), n: c(true), d(ssl=0 local=0 anon=1 log=0 chunks=2048,4096)

    >>> mcores_d = cores_d.merge()
    >>> print mcores_d
    1. (0) p: c(true), d(None), n: c(None), d(true): L4
    2. (2) p: c(listen=1 timeout=1), d(None), n: c(true), d(listen=1 timeout=1): L1
    3. (2) p: c(listen=1 timeout=0), d(None), n: c(true), d(listen=1 timeout=0): L2
    4. (2) p: c(listen=0), d(None), n: c(listen=1), d(None): L3
    5. (2) p: c(true), d(ssl=0 local=0), n: c(ssl=0 local=0), d(None): L5
    6. (2) p: c(ssl=0 local=0), d(None), n: c(true), d(ssl=0 local=0): L6
    7. (3) p: c(ssl=0 local=0 anon=1), d(None), n: c(true), d(ssl=0 local=0 anon=1): L7
    8. (5) p: c(ssl=0 local=0 anon=1 log=1 chunks=2048,4096), d(None), n: c(true), d(ssl=0 local=0 anon=1 log=1 chunks=2048,4096): L8
    9. (5) p: c(ssl=0 local=0 anon=1 log=0 chunks=2048,4096), d(None), n: c(true), d(ssl=0 local=0 anon=1 log=0 chunks=2048,4096): L9


    #verification

    >>> dom = Dom([('a', frozenset(['1', '0'])), ('b', frozenset(['1', '0'])), ('c', frozenset(['1', '0'])), ('d', frozenset(['0', '1', '2', '3']))])
    >>> config_covs = [\
    (Config([('a', '1'), ('b', '1'), ('c', '1'), ('d', '1')]), 'L1'), \
    (Config([('a', '1'), ('b', '1'), ('c', '1'), ('d', '0')]), 'L1'), \
    (Config([('a', '1'), ('b', '1'), ('c', '1'), ('d', '3')]), 'L1'), \
    (Config([('a', '1'), ('b', '1'), ('c', '1'), ('d', '2')]), 'L1'), \
    (Config([('a', '1'), ('b', '1'), ('c', '0'), ('d', '1')]), 'L1'), \
    (Config([('a', '1'), ('b', '1'), ('c', '0'), ('d', '0')]), 'L1'), \
    (Config([('a', '1'), ('b', '1'), ('c', '0'), ('d', '3')]), 'L1'), \
    (Config([('a', '1'), ('b', '1'), ('c', '0'), ('d', '2')]), 'L1'), \
    (Config([('a', '1'), ('b', '0'), ('c', '1'), ('d', '1')]), 'L1'), \
    (Config([('a', '1'), ('b', '0'), ('c', '1'), ('d', '0')]), 'L1'), \
    (Config([('a', '1'), ('b', '0'), ('c', '1'), ('d', '3')]), 'L1'), \
    (Config([('a', '1'), ('b', '0'), ('c', '1'), ('d', '2')]), 'L1'), \
    (Config([('a', '1'), ('b', '0'), ('c', '0'), ('d', '1')]), 'L2'), \
    (Config([('a', '1'), ('b', '0'), ('c', '0'), ('d', '0')]), 'L2'), \
    (Config([('a', '1'), ('b', '0'), ('c', '0'), ('d', '3')]), 'L2'), \
    (Config([('a', '1'), ('b', '0'), ('c', '0'), ('d', '2')]), 'L2'), \
    (Config([('a', '0'), ('b', '1'), ('c', '1'), ('d', '1')]), 'L2'), \
    (Config([('a', '0'), ('b', '1'), ('c', '1'), ('d', '0')]), 'L2'), \
    (Config([('a', '0'), ('b', '1'), ('c', '1'), ('d', '3')]), 'L2'), \
    (Config([('a', '0'), ('b', '1'), ('c', '1'), ('d', '2')]), 'L2'), \
    (Config([('a', '0'), ('b', '1'), ('c', '0'), ('d', '1')]), 'L2'), \
    (Config([('a', '0'), ('b', '1'), ('c', '0'), ('d', '0')]), 'L2'), \
    (Config([('a', '0'), ('b', '1'), ('c', '0'), ('d', '3')]), 'L2'), \
    (Config([('a', '0'), ('b', '1'), ('c', '0'), ('d', '2')]), 'L2'), \
    (Config([('a', '0'), ('b', '0'), ('c', '1'), ('d', '1')]), 'L2'), \
    (Config([('a', '0'), ('b', '0'), ('c', '1'), ('d', '0')]), 'L2'), \
    (Config([('a', '0'), ('b', '0'), ('c', '1'), ('d', '3')]), 'L2'), \
    (Config([('a', '0'), ('b', '0'), ('c', '1'), ('d', '2')]), 'L2'), \
    (Config([('a', '0'), ('b', '0'), ('c', '0'), ('d', '1')]), 'L2'), \
    (Config([('a', '0'), ('b', '0'), ('c', '0'), ('d', '0')]), 'L2'), \
    (Config([('a', '0'), ('b', '0'), ('c', '0'), ('d', '3')]), 'L2'), \
    (Config([('a', '0'), ('b', '0'), ('c', '0'), ('d', '2')]), 'L2'), \
    ] 
    >>> configs,covs = zip(*config_covs)

    >>> c1 = PNCore((PCore((Core([('a', frozenset(['1']))]), Core([('b', frozenset(['0'])), ('c', frozenset(['0']))]))), NCore((Core(), Core([('a', frozenset(['1']))])))))
    >>> c2 = PNCore((PCore((Core(), Core([('a', frozenset(['1']))]))), NCore((Core([('a', frozenset(['1']))]), Core([('b', frozenset(['0'])), ('c', frozenset(['0']))])))))
    >>> cores_d = CORES_D([('L1',c1),('L2',c2)])
    >>> print cores_d
    1. L1: p: c(a=1), d(b=0 c=0), n: c(true), d(a=1)
    2. L2: p: c(true), d(a=1), n: c(a=1), d(b=0 c=0)


    # >>> new_cores_d = cores_d.check(configs,covs,dom)
    # >>> print new_cores_d
    # 1. L1: p: c(a=1), d(b=0 c=0), n: c(true), d(a=1)
    # 2. L2: p: c(true), d(a=1), n: c(a=1), d(b=0 c=0)
    """                                                       
    def __init__(self,d=OrderedDict()):
        OrderedDict.__init__(self,d)

        if CM.__vdebug__:
            assert all(isinstance(sid,str) and isinstance(c,PNCore)
                       for sid,c in self.iteritems())

    def __str__(self):
        return '\n'.join("{}. {}: {}"
                         .format(i+1,sid,self[sid])
                         for i,sid in enumerate(sorted(self)))
    

    def merge(self):
        mcores_d = OrderedDict()
        for sid,pncore in self.iteritems():
            if pncore in mcores_d:
                mcores_d[pncore].add(sid)
            else:
                mcores_d[pncore] = set([sid])

        for sid in mcores_d:
            mcores_d[sid]=frozenset(mcores_d[sid])

        return MCORES_D(mcores_d)









class MCORES_D(OrderedDict):
    """    
    >>> c1 = PNCore.mk(Core([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))]),None,Core(),Core([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))]))
    >>> c2 = PNCore.mk(Core([('listen', frozenset(['1'])), ('timeout', frozenset(['0']))]), None, Core(), Core([('listen', frozenset(['1'])), ('timeout', frozenset(['0']))]))
    >>> c3 = PNCore.mk(Core([('listen', frozenset(['0']))]), None, Core([('listen', frozenset(['1']))]), None)
    >>> c4 = PNCore.mk(Core(), None, None, Core())
    >>> c5 = PNCore.mk(Core(), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0']))]), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0']))]), None)
    >>> c6 = PNCore.mk(Core([('ssl', frozenset(['0'])), ('local', frozenset(['0']))]), None, Core(), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0']))]))
    >>> c7 = PNCore.mk(Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1']))]), None, Core(), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1']))]))
    >>> c8 = PNCore.mk(Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1'])), ('log', frozenset(['1'])), ('chunks', frozenset(['4096', '2048']))]), None, Core(), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1'])), ('log', frozenset(['1'])), ('chunks', frozenset(['4096', '2048']))]))
    >>> c9 = PNCore.mk(Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1'])), ('log', frozenset(['0'])), ('chunks', frozenset(['4096', '2048']))]), None, Core(), Core([('ssl', frozenset(['0'])), ('local', frozenset(['0'])), ('anon', frozenset(['1'])), ('log', frozenset(['0'])), ('chunks', frozenset(['4096', '2048']))]))
    >>> cores = [c1,c2,c3,c4,c5,c6,c7,c8,c9]
    >>> sids = ['L1','L2',['L3','L3b'],'L4','L5','L6','L7','L8','L9']
    >>> sids = [frozenset(sid if isinstance(sid,list) else [sid]) for sid in sids]

    >>> mcores_d = MCORES_D(zip(cores,sids))
    >>> print mcores_d
    1. (0) p: c(true), d(None), n: c(None), d(true): L4
    2. (2) p: c(listen=1 timeout=1), d(None), n: c(true), d(listen=1 timeout=1): L1
    3. (2) p: c(listen=1 timeout=0), d(None), n: c(true), d(listen=1 timeout=0): L2
    4. (2) p: c(true), d(ssl=0 local=0), n: c(ssl=0 local=0), d(None): L5
    5. (2) p: c(ssl=0 local=0), d(None), n: c(true), d(ssl=0 local=0): L6
    6. (2) p: c(listen=0), d(None), n: c(listen=1), d(None): L3,L3b
    7. (3) p: c(ssl=0 local=0 anon=1), d(None), n: c(true), d(ssl=0 local=0 anon=1): L7
    8. (5) p: c(ssl=0 local=0 anon=1 log=1 chunks=2048,4096), d(None), n: c(true), d(ssl=0 local=0 anon=1 log=1 chunks=2048,4096): L8
    9. (5) p: c(ssl=0 local=0 anon=1 log=0 chunks=2048,4096), d(None), n: c(true), d(ssl=0 local=0 anon=1 log=0 chunks=2048,4096): L9

    >>> print mcores_d.strength_cts
    [(0, 1, 1), (2, 5, 6), (3, 1, 1), (5, 2, 2)]

    >>> print mcores_d.strength_str
    (0,1,1), (2,5,6), (3,1,1), (5,2,2)
    """
    def __init__(self,d=OrderedDict()):
        #print d
        OrderedDict.__init__(self,d)

        if CM.__vdebug__:
            assert self and all(isinstance(c,PNCore) and is_cov(cov)
                                for c,cov in self.iteritems()), self

    def __str__(self):
        mc = sorted(self.iteritems(),
                    key=lambda (core,cov): (core.strength,len(cov)))

        return '\n'.join("{}. ({}) {}: {}"
                         .format(i+1,core.strength,core,str_of_cov(cov))
                         for i,(core,cov) in enumerate(mc))
    @property
    def strength_cts(self):
        """
        (strength,cores,sids)
        """
        try:
            return self._strength_cts
        except AttributeError:
            sizs = [core.strength for core in self]

            res = []
            for strength in sorted(set(sizs)):
                cores = [core for core in self if core.strength==strength]
                cov = set()
                for core in cores:
                    for sid in self[core]:
                        cov.add(sid)
                res.append((strength,len(cores),len(cov)))

            self._strength_cts = res
            return self._strength_cts

    @property
    def strength_str(self):
        return ', '.join("({},{},{})".format(siz,ncores,ncov)
                         for siz,ncores,ncov in self.strength_cts)
        

def doctestme():
    import doctest
    doctest.testmod()




