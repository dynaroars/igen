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


is_cov = lambda cov: (isinstance(cov,frozenset) and
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
    >>> print c1.z3expr(dom.z3db)
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

    >>> print configs.z3expr(dom.z3db)
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
        try:
            return self._z3db
        except AttributeError:
            _z3db = OrderedDict()  #{'x':(x,{'true':True})}
            for k,vs in self.iteritems():
                vs = sorted(list(vs))
                ttyp,tvals=z3.EnumSort(k,vs)
                rs = []
                for v,v_ in zip(vs,tvals):
                    rs.append((v,v_))
                rs.append(('typ',ttyp))

                _z3db[k]=(z3.Const(k,ttyp),dict(rs))
            self._z3db = _z3db
            return self._z3db


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
        
        rgen = lambda: [(k,random.choice(list(dom[k]))) for k in self]
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
    >>> print c1.z3expr(dom.z3db)
    And(listen == 1,
        timeout == 1,
        ssl == 1,
        local == 0,
        anon == 0,
        log == 1,
        chunks == 2048)

    >>> assert c1.implies({'listen':'1','chunks':frozenset(['0','2048'])})
    >>> assert not c1.implies({'listen':'1','chunks':frozenset(['0','4096'])})
    >>> assert not c1.implies({'boo':'1','chunks':frozenset(['0','2048'])})
    >>> assert not c1.implies({'listen':'3','chunks':frozenset(['0','2048'])})

    """
    def __init__(self,c=HDict()):
        HDict.__init__(self,c)

        if CM.__vdebug__:
            assert all(is_setting(s) for s in self.iteritems()), self

    def __str__(self):
        return ' '.join(map(str_of_setting,self.iteritems()))

    def z3expr(self,z3db):
        if CM.__vdebug__:
            assert len(self) == len(z3db), (len(self), len(z3db))
            
        f = []
        for vn,vv in self.iteritems():
            vn_,vs_ = z3db[vn]
            f.append(vn_==vs_[vv])

        return z3util.myAnd(f)

    def implies(self,d):
        try:
            return all(self[k] in d[k] for k in d)
        except KeyError:
            return False


class Configs(list):
    """
    >>> dom = Dom([('listen', frozenset(['1', '0'])), ('timeout', frozenset(['1', '0'])), ('ssl', frozenset(['1', '0'])), ('local', frozenset(['1', '0'])), ('anon', frozenset(['1', '0'])), ('log', frozenset(['1', '0'])), ('chunks', frozenset(['0', '65536', '4096', '2048']))])
    >>> c1 = Config([('listen','1'),('timeout','1'),('ssl','1'),('local','0'),('anon','0'),('log','1'),('chunks','2048')])
    >>> c2 = Config([('listen','0'),('timeout','1'),('ssl','1'),('local','0'),('anon','0'),('log','0'),('chunks','0')])
    >>> configs = Configs([c1,c2])

    >>> print configs
    0. listen=1 timeout=1 ssl=1 local=0 anon=0 log=1 chunks=2048
    1. listen=0 timeout=1 ssl=1 local=0 anon=0 log=0 chunks=0

    >> assert configs.implies({'timeout':frozenset(['1']),'chunks':frozenset(['0','2048'])})

    >>> print configs.z3expr(dom.z3db)
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
        return all(c.implies(d) for c in self)
    
    def z3expr(self,z3db):
        return z3util.myOr([c.z3expr(z3db) for c in self])

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

    >>> assert c1.z3expr(dom.z3db) is z3util.TRUE
    >>> print c3.z3expr(dom.z3db)
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
        return ' '.join(map(str_of_csetting,self.iteritems()))

    def z3expr(self,z3db):
        if not self:
            return z3util.TRUE
        else:
            f = []
            for vn,vs in self.iteritems():
                vn_,vs_ = z3db[vn]
                f.append(z3util.myOr([vn_ == vs_[v] for v in vs]))

            return z3util.myAnd(f)        
        

class CDCore(tuple):
    """
    >>> dom = Dom([('listen', frozenset(['1', '0'])), ('timeout', frozenset(['1', '0'])), ('ssl', frozenset(['1', '0'])), ('local', frozenset(['1', '0'])), ('anon', frozenset(['1', '0'])), ('log', frozenset(['1', '0'])), ('chunks', frozenset(['0', '65536', '4096', '2048']))])

    >>> c = CDCore.mk_default()
    >>> print c
    c(None), d(None)

    >>> assert c.z3expr(dom.z3db) is None

    >>> c = CDCore((Core([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))]), None))
    >>> print c
    c(listen=1 timeout=1), d(None)
    
    >>> print c.z3expr(dom.z3db)
    And(listen == 1, timeout == 1)

    >>> print CDCore((Core(),None)).z3expr(dom.z3db)
    True

    """    
    def __init__(self,(cc,dc)):
        if CM.__vdebug__:
            assert cc is None or isinstance(cc,Core), cc
            assert dc is None or isinstance(dc,Core), dc
        
        tuple.__init__(self,(cc,dc))

        
    @property
    def conj(self): return self[0]
    @property
    def disj(self): return self[1]

    def __str__(self):
        return "c({}), d({})".format(self.conj,self.disj)

    def z3expr(self,z3db):
        conj = lambda: self.conj.z3expr(z3db)
        disj = lambda: z3.Not(self.disj.z3expr(z3db))

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

    @staticmethod
    def mk_default():
        return CDCore((None,None))

class PNCore(tuple):
    """
    >>> dom = Dom([('listen', frozenset(['1', '0'])), ('timeout', frozenset(['1', '0'])), ('ssl', frozenset(['1', '0'])), ('local', frozenset(['1', '0'])), ('anon', frozenset(['1', '0'])), ('log', frozenset(['1', '0'])), ('chunks', frozenset(['0', '65536', '4096', '2048']))])

    >>> print PNCore.mk_default()
    p: c(None), d(None), n: c(None), d(None)

    >>> pcc = Core([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))])
    >>> pdc = None
    >>> ncc = Core()
    >>> ndc = Core([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))])

    >>> c = PNCore((CDCore((pcc,pdc)),CDCore((ncc,ndc))))
    >>> print c
    p: c(listen=1 timeout=1), d(None), n: c(true), d(listen=1 timeout=1)

    >>> print c.z3expr(dom.z3db)
    And(listen == 1, timeout == 1)

    >>> assert len(c.settings) == 2


    >>> pcc = Core([('listen', frozenset(['0']))])
    >>> pdc = None
    >>> ncc = Core([('listen', frozenset(['1']))])
    >>> ndc = None
    >>> c = PNCore((CDCore((pcc,pdc)),CDCore((ncc,ndc))))
    >>> print c.z3expr(dom.z3db)
    listen == 0

    >>> assert len(c.settings) == 2

    >>> c = PNCore((CDCore((Core(),None)),CDCore((None,Core()))))
    >>> print c.z3expr(dom.z3db)
    True

    >>> assert len(c.settings) == 0
    """
    def __init__(self,(pc,nc)):
        if CM.__vdebug__:
            assert isinstance(pc,CDCore) and isinstance(nc,CDCore), (pc,nc)

        tuple.__init__(self,(pc,nc))

    @property
    def pcore(self): return self[0]
    @property
    def ncore(self): return self[1]
        
    def __str__(self):
        return "p: {}, n: {}".format(self.pcore,self.ncore)
        
    # def repair(self,configs,dom):
    #     """
    #     Return a *real* inv/overapprox over configs
    #     inv = (conj,disj)
    #     """
    #     #pos traces => p1 & not(n2)
    #     if p1: #x & y 
    #         if not all(c_implies(c,p1) for c in configs):
    #             p1 = None

    #     if n2: # y & z => not(y) | not(z)
    #         n2 = neg_of_core(n2,core,dom)
    #         if not all(d_implies(c,n2) for c in configs):
    #             n2 = None

    #     #neg traces => n1 & not(p2)
    #     #not(n1) || p2
    #     if n1 is None and p2 is None:
    #         pass
    #     elif n1 is not None and p2 is None:
    #         n1 = neg_of_core(n1,core,dom)
    #         if not all(d_implies(c,n1) for c in configs):
    #             n1 = None
    #     elif n1 is None and p2 is not None:
    #         if not all(c_implies(c,p2) for c in configs):
    #             p2 = None
    #     else:
    #         n1 = neg_of_core(n1,core,dom)
    #         if not all(d_implies(c,n1) or c_implies(c,p2) 
    #                    for c in configs):
    #             n1 = None
    #             p2 = None
        
    def z3expr(self,z3db):
        z3_pc = self.pcore.z3expr(z3db)
        if z3_pc:
            z3_pc = z3.simplify(z3_pc)

        z3_nc = self.ncore.z3expr(z3db)
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
        cores = [self.pcore.conj,self.pcore.disj,
                 self.ncore.conj,self.ncore.disj]
        cores = [c for c in cores if c]
        settings = []
        for c in cores:
            settings.extend(c.hcontent)
        settings = set(settings)
        return settings

    @staticmethod
    def mk_default():
        return PNCore((CDCore.mk_default(),CDCore.mk_default()))


class CORES_D(OrderedDict):
    def __init__(self,d=OrderedDict()):
        OrderedDict.__init__(self,d)

        if CM.__vdebug__:
            assert all(isinstance(sid,str) and isinstance(c,PNCore)
                       for sid,c in self.iteritems())

    def __str__(self):
        return '\n'.join("{}. {}: {}"
                         .format(i+1,sid,self[sid])
                         for i,sid in enumerate(sorted(self)))
    





def doctestme():
    import doctest
    doctest.testmod()



