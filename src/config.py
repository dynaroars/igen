import abc 
import tempfile
from time import time
import os.path
import itertools
import random
import numpy

import z3
import z3util
from vu_common import HDict
import vu_common as CM

from collections import OrderedDict, MutableMapping

logger = CM.VLog('config')
logger.level = CM.VLog.DEBUG
CM.VLog.PRINT_TIME = True
CM.__vdebug__ = True  #IMPORTANT: TURN OFF WHEN DO REAL RUN!!

do_comb_conj_disj = True
show_cov = True
allows_known_errors = False
analyze_outps = False

getpath = lambda f: os.path.realpath(os.path.expanduser(f))

#Data Structures
class CustDict(MutableMapping):
    """
    MuttableMapping ex: https://stackoverflow.com/questions/21361106/how-would-i-implement-a-dict-with-abstract-base-classes-in-python
    """
    __metaclass__ = abc.ABCMeta
    def __init__(self): self.__dict__ = {}
    def __len__(self): return len(self.__dict__)
    def __getitem__(self,key): return self.__dict__[key]
    def __iter__(self): return iter(self.__dict__)    
    def __setitem__(self,key,val): raise NotImplementedError("no setitem")
    def __delitem__(self,key): raise NotImplementedError("no delitem")
    def add_set(self,key,val):
        """
        For mapping from key to set
        """
        if key not in self.__dict__:
            self.__dict__[key] = set()
        self.__dict__[key].add(val)
        
is_cov = lambda cov: (isinstance(cov,set) and
                      all(isinstance(s,str) for s in cov))
def str_of_cov(cov):
    """
    >>> assert str_of_cov(set("L2 L1 L3".split())) == '(3) L1,L2,L3'
    """
    if CM.__vdebug__:
        assert is_cov(cov),cov

    s = "({})".format(len(cov))
    if show_cov:
        s = "{} {}".format(s,','.join(sorted(cov)))
    return s
    
is_valset = lambda vs: (isinstance(vs,frozenset) and vs and
                        all(isinstance(v,str) for v in vs))

def str_of_valset(s):
    """
    >>> str_of_valset(frozenset(['1','2','3']))
    '1,2,3'
    """
    return ','.join(sorted(s))
    
is_setting = lambda (k,v): isinstance(k,str) and isinstance(v,str)
def str_of_setting((k,v)):
    """
    >>> print str_of_setting(('x','1'))
    x=1
    """
    if CM.__vdebug__:
        assert is_setting((k,v)), (k,v)
    return '{}={}'.format(k,v)

is_csetting = lambda (k,vs): isinstance(k,str) and is_valset(vs)

def str_of_csetting((k,vs)):
    """
    >>> print str_of_csetting(('x',frozenset(['1'])))
    x=1
    >>> print str_of_csetting(('x',frozenset(['3','1'])))
    x=1,3
    """
    if CM.__vdebug__:
        assert is_csetting((k,vs)), (k,vs)
    
    return '{}={}'.format(k,str_of_valset(vs))

class Dom(OrderedDict):
    def __init__(self,dom):
        OrderedDict.__init__(self,dom)
        
        if CM.__vdebug__:
            assert self and all(is_csetting(s) for s in self.iteritems()), self

    def __str__(self):
        """
        >>> print Dom([('x',frozenset(['1','2'])),('y',frozenset(['1']))])
        2 vars and 2 poss configs
        1. x: (2) 1,2
        2. y: (1) 1
        """
        s = "{} vars and {} poss configs".format(len(self),self.siz)
        s_detail = '\n'.join("{}. {}: ({}) {}"
                             .format(i+1,k,len(vs),str_of_valset(vs))
                             for i,(k,vs) in enumerate(self.iteritems()))
        s = "{}\n{}".format(s,s_detail)
        return s

    @property
    def siz(self): return CM.vmul(len(vs) for vs in self.itervalues())
    @property
    def z3db(self):
        z3db = dict()
        for k,vs in self.iteritems():
            vs = sorted(list(vs))
            ttyp,tvals=z3.EnumSort(k,vs)
            rs = [vv for vv in zip(vs,tvals)]
            rs.append(('typ',ttyp))
            z3db[k]=(z3.Const(k,ttyp),dict(rs))
        return z3db
    
    @staticmethod
    def get_dom(dom_file):
        """
        Read domain info from a file
        """
        if CM.__vdebug__:
            assert os.path.isfile(dom_file), dom_file

        def get_lines(lines):
            rs = (line.split() for line in lines)
            rs = [(parts[0],frozenset(parts[1:])) for parts in rs]
            return rs

        dom = Dom(get_lines(CM.iread_strip(dom_file)))

        config_default = None
        dom_file_default = dom_file+'.default'
        if os.path.isfile(dom_file_default):
            rs = dict(get_lines(CM.iread_strip(dom_file_default)))
            config_default = Config((k,list(rs[k])[0]) for k in dom)

        return dom,config_default

    #Methods to generate configurations
    def gen_configs_full(self):
        ns,vs = itertools.izip(*self.iteritems())
        configs = [Config(zip(ns,c)) for c in itertools.product(*vs)]
        return configs

    def gen_configs_rand(self,rand_n):
        if CM.__vdebug__:
            assert 0 < rand_n < self.siz, (rand_n,self.siz)

        rgen = lambda: [(k,random.choice(list(self[k]))) for k in self]
        configs = list(set(Config(rgen()) for _ in range(rand_n)))
        return configs

    def gen_configs_tcover1(self):
        """
        Return a set of tcover array of stren 1
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

        configs = []
        while dom_used: configs.append(mk())
        return configs
    
class Config(HDict):
    """
    >>> c1 = Config([('a', '0'), ('b', '0'), ('c', '0')])
    """
    def __init__(self,config=HDict()):
        HDict.__init__(self,config)
        
        if CM.__vdebug__:
            assert all(is_setting(s) for s in self.iteritems()), self

    def __str__(self,cov=None):
        if CM.__vdebug__:
            assert cov is None or is_cov(cov), cov

        s =  ' '.join(map(str_of_setting,self.iteritems()))
        if cov:
            s = "{}: {}".format(s,str_of_cov(cov))
        return s

    def c_implies(self,core):
        """
        x=0&y=1 => x=0,y=1
        not(x=0&z=1 => x=0,y=1)
        """
        if CM.__vdebug__:
            assert isinstance(core,Core),core

        return (not core or
                all(k in self and self[k] in core[k] for k in core))

    def d_implies(self,core): 
        if CM.__vdebug__:
            assert isinstance(core,Core),core

        return (not core or
                any(k in self and self[k] in core[k] for k in core))

    def z3expr(self,z3db):
        if CM.__vdebug__:
            assert len(self) == len(z3db), (len(self), len(z3db))

        f = []
        for vn,vv in self.iteritems():
            vn_,vs_ = z3db[vn]
            f.append(vn_==vs_[vv])

        return z3util.myAnd(f)

class Core(HDict):
    """
    >>> print Core()
    true

    >>> c = Core([('a',frozenset(['2'])),('c',frozenset(['0','1']))])
    >>> print c
    a=2 c=0,1
    """
    def __init__(self,core=HDict()):
        HDict.__init__(self,core)
        
        if CM.__vdebug__:
            assert all(is_csetting(s) for s in self.iteritems()), self

    def __str__(self,delim=' '):
        if self:
            return delim.join(map(str_of_csetting,self.iteritems()))
        else:
            return 'true'
        
    def neg(self,dom):
        try:
            return self._neg
        except AttributeError:
            if CM.__vdebug__:
                assert isinstance(dom,Dom),dom
            self._neg = Core((k,dom[k]-self[k]) for k in self)
            return self._neg
        
    @staticmethod
    def is_maybe_core(core): return core is None or isinstance(core,Core)

    def z3expr(self,z3db,myf):
        f = []
        for vn,vs in self.iteritems():
            vn_,vs_ = z3db[vn]
            f.append(z3util.myOr([vn_ == vs_[v] for v in vs]))

        return myf(f)
    
class MCore(tuple):
    """
    Multiple cores
    """
    def __init__(self,cores):
        tuple.__init__(self,cores)

        if CM.__vdebug__:
            assert len(self) == 2 or len(self) == 4, self
            assert all(Core.is_maybe_core(c) for c in self), self

    @property
    def settings(self):
        core = (c for c in self if c)
        return set(s for c in core for s in c.iteritems())

    @property
    def values(self):
        core = (c for c in self if c)
        return set(s for c in core for s in c.itervalues())

    @property
    def sstren(self): return len(self.settings)
    
    @property
    def vstren(self): return sum(map(len,self.values))


class SCore(MCore):
    def __init__(self,(mc,sc)):
        """
        mc: main core that will generate cex's
        sc (if not None): sat core that is satisfied by all generated cex'
        """
        super(SCore,self).__init__((mc,sc))
        #additional assertion
        if CM.__vdebug__:
            assert mc is None or isinstance(mc,Core) and mc, mc
            #sc is not None => ...
            assert not sc or all(k not in mc for k in sc), sc
        self.keep = False

    def set_keep(self):
        """
        keep: if true then generated cex's with diff settings than mc 
        and also those that have the settings of mc
        """
        self.keep = True
    
    @property
    def mc(self): return self[0]

    @property
    def sc(self): return self[1]

    def __str__(self):
        ss = []
        if self.mc:
            s = ""
            try:  #to support old format that doesn't have keep
                if self.keep:
                    s = "(keep)" 
            except AttributeError:
                pass

            ss.append("mc{}: {}".format(s,self.mc))
                                            
                      
        if self.sc:
            ss.append("sc: {}".format(self.sc))
        return '; '.join(ss)        
        
    @staticmethod
    def mk_default(): return SCore((None,None))
        
        

class PNCore(MCore):
    def __init__(self,(pc,pd,nc,nd)):
        super(PNCore,self).__init__((pc,pd,nc,nd))
        
    @property
    def pc(self): return self[0]
    @property
    def pd(self): return self[1]
    @property
    def nc(self): return self[2]
    @property
    def nd(self):return self[3]
    
    @staticmethod
    def mk_default(): return PNCore((None,None,None,None))

    def __str__(self,fstr_f=None):
        """
        >>> pc = Core([('a',frozenset(['2'])),('c',frozenset(['0','1']))])
        >>> pd = None
        >>> nc = Core([('b',frozenset(['2']))])
        >>> nd = None
        >>> print PNCore((pc,pd,nc,nd))
        pc: a=2 c=0,1; nc: b=2
        
        Important: only call fstr_f *after* being analyzed
        """
        if CM.__vdebug__:
            assert fstr_f is None or callable(fstr_f),fstr_f
        if fstr_f:
            return fstr_f(self)
        else:
            ss = ("{}: {}".format(s,c) for s,c in
                  zip('pc pd nc nd'.split(),self) if c is not None)
            return '; '.join(ss)

    def fstr(self,dom):
        """
        Assumption: all 4 cores are verified and simplified
        """
        if CM.__vdebug__:
            assert self.pc is None or (isinstance(self.pc,Core) and self.pc), self.pc
            assert self.pd is None or (isinstance(self.pd,Core) and self.pd), self.pd
            assert self.nc is None or (isinstance(self.nc,Core) and self.nc), self.nc
            assert self.nd is None or (isinstance(self.nd,Core) and self.nd), self.nd
            assert isinstance(dom,Dom),dom
            
        def _f(core,delim):
            s = core.__str__(delim)
            if len(core) > 1:
                s = '({})'.format(s)
            return s

        def _cd(ccore,dcore,delim):
            ss = []
            if ccore:
                ss.append(_f(ccore,' & '))
            if dcore:
                assert isinstance(dom,Dom),dom
                dcore_n = dcore.neg(dom)
                ss.append(_f(dcore_n, ' | '))
            return delim.join(ss) if ss else 'true'

        if self.is_simplified():
            if (self.nc is None and self.nd is None):
                #pc & not(pd)
                ss = _cd(self.pc,self.pd,' & ')
            else:
                #not(nc & not(nd))  =  not(nc) | nd
                ss = _cd(self.nd,self.nc,' | ')
        else:
            p_ss = _cd(self.pc,self.pd,' & ')
            n_ss = _cd(self.nd,self.nc,' | ')
            ss = ','.join([p_ss,n_ss]) + '***'  #inconcistent?

        return ss
    @property
    def typ(self):
        ss = set()
        if self.pc and self.pd:
            ss.add('mixed1')
        elif self.pc:
            ss.add('conj')
        elif self.pd:
            ss.add('disj')

        if self.nc and self.nd:
            ss.add('mixed2')
        elif self.nc:
            ss.add('disj')
        elif self.nd:
            ss.add('conj')

        if any(s.startswith('mixed') for s in ss):
            return 'mix'
        elif 'conj' in ss and 'disj' in ss:
            return 'mix'
        elif len(ss)==1 and 'conj' in ss:
            return 'conj'
        elif len(ss)==1 and 'disj' in ss:
            return 'disj'
        elif len(ss)==0:
            return 'conj'  #true
        else:
            logger.warn("W: unexpected case of typ, ss {}".format(ss))
            return 'conj'
            
            
            

    def verify(self,configs,dom):
        if CM.__vdebug__:
            assert self.pc is not None, self.pc #this never could happen
            #nc is None => pd is None
            assert self.nc is not None or self.pd is None, (self.nc,self.nd)
            assert all(isinstance(c,Config) for c in configs) and configs, configs
            assert isinstance(dom,Dom),dom

        pc,pd,nc,nd = self

        #traces => pc & neg(pd)
        if CM.__vdebug__:
            if pc:
                assert all(c.c_implies(pc) for c in configs), pc

        if pd:
            pd_n = pd.neg(dom)
            if not all(c.d_implies(pd_n) for c in configs):
                logger.debug('pd {} invalid'.format(pd))
                pd = None

        #neg traces => nc & neg(nd)
        #pos traces => neg(nc & neg(nd))
        #post traces => nd | neg(nc) 
        if nc and not nd:
            nc_n = nc.neg(dom)
            if not all(c.d_implies(nc_n) for c in configs):
                logger.debug('nc {} invalid'.format(nc))
                nc = None
        elif not nc and nd:
            if not all(c.c_implies(nd) for c in configs):
                logger.debug('nd {} invalid'.format(nd))
                nd = None
        elif nc and nd:
            nc_n = nc.neg(dom)        
            if not all(c.c_implies(nd) or
                       c.d_implies(nc_n) for c in configs):
                logger.debug('nc {} & nd {} invalid').format(nc,nd)
                nc = None
                nd = None

        return PNCore((pc,pd,nc,nd))
    
    def simplify(self,dom):
        """
        Compare between (pc,pd) and (nc,nd) and return the stronger one.
        This will set either (pc,pd) or (nc,nd) to (None,None)

        Assumption: all 4 cores are verified
        """
        if CM.__vdebug__:
            assert self.pc is not None, self.pc #this never could happen
            #nc is None => pd is None
            assert (self.nc is not None or self.pd is None), (self.nc,self.nd)
            assert isinstance(dom,Dom),dom

        #pf = pc & neg(pd)
        #nf = neg(nc & neg(nd)) = nd | neg(nc)
        pc,pd,nc,nd = self

        #remove empty ones
        if not pc: pc = None
        if not pd: pd = None
        if not nc: nc = None
        if not nd: nd = None

        if (pc is None and pd is None) or (nc is None and nd is None):
            return PNCore((pc,pd,nc,nd))

        #convert to z3
        z3db = dom.z3db

        def _f(cc,cd):
            fs = []
            if cc:
                f = cc.z3expr(z3db,myf=z3util.myAnd)
                fs.append(f)
            if cd:
                cd_n = cd.neg(dom)
                f = cd_n.z3expr(z3db,myf=z3util.myOr)
                fs.append(f)
            return fs

        pf = z3util.myAnd(_f(pc,pd))
        nf = z3util.myOr(_f(nd,nc))

        if z3util.is_tautology(z3.Implies(pf,nf)):
            nc = None
            nd = None
        elif z3util.is_tautology(z3.Implies(nf,pf)):
            pc = None
            pd = None
        else:
            #could occur when using incomplete traces
            logger.warn("inconsistent ? {}\npf: {} ?? nf: {}"
                        .format(PNCore((pc,pd,nc,nd)),
                        pf,nf))

        return PNCore((pc,pd,nc,nd))

    def is_simplified(self):
        return ((self.pc is None and self.pd is None) or
                (self.nc is None and self.nd is None))



class Cores_d(CustDict):
    def __setitem__(self,sid,pncore):
        if CM.__vdebug__:
            assert isinstance(sid,str),sid
            assert isinstance(pncore,PNCore),pncore
        self.__dict__[sid]=pncore

    def __str__(self):
        return '\n'.join("{}. {}: {}"
                         .format(i+1,sid,self[sid].__str__(fstr_f=None))
                         for i,sid in enumerate(sorted(self)))
    def merge(self):
        mcores_d = Mcores_d()
        for sid,core in self.iteritems():
            mcores_d.add(core,sid)
        return mcores_d

    def analyze(self,covs_d,dom):
        if CM.__vdebug__:
            assert isinstance(covs_d,Covs_d) and covs_d, covs_d
            assert len(self) == len(covs_d), (len(self),len(covs_d))
            assert isinstance(dom,Dom), dom

        def show_compare(sid,old_c,new_c):
            if old_c != new_c:
                logger.debug("sid {}: {} ~~> {}".
                             format(sid,old_c,new_c))

        logger.info("analyze interactions for {} sids".format(len(self)))
        logger.debug("verify ...")
        vcache = {}
        rs_verify = []    
        for sid,core in self.iteritems():
            configs = frozenset(covs_d[sid])
            key = (core,configs)
            if key not in vcache:
                core_ = core.verify(configs,dom)
                vcache[key]=core_
                show_compare(sid,core,core_)
            else:
                core_ = vcache[key]

            rs_verify.append((sid,core_))        

        logger.debug("simplify ...")
        scache = {}
        rs_simplify = Cores_d()
        for sid,core in rs_verify:
            if core not in scache:
                core_ = core.simplify(dom)
                scache[core]=core_
                show_compare(sid,core,core_)
            else:
                core_ = scache[core]
            rs_simplify[sid]=core_

        return rs_simplify

    def show_analysis(self,dom):
        if CM.__vdebug__:
            assert isinstance(dom,Dom),dom
            
        mcores_d = self.merge()
        fstr_f=lambda c: c.fstr(dom)
        strs = mcores_d.strs(fstr_f)
        
        logger.debug("mcores_d has {} items\n".format(len(mcores_d)) +
                     '\n'.join(strs))
        
        logger.info("mcores_d strens: {}".format(mcores_d.strens_str))

        
        return mcores_d
    
class Mcores_d(CustDict):
    """
    A mapping from core -> {sids}
    """
    def add(self,core,sid):
        if CM.__vdebug__:
            assert isinstance(core,PNCore),core
            assert isinstance(sid,str),str
        super(Mcores_d,self).add_set(core,sid)

    def strs(self,fstr_f=None):
        if CM.__vdebug__:
            assert fstr_f is None or callable(fstr_f),fstr_f

        mc = sorted(self.iteritems(),
                    key=lambda (core,cov): (core.sstren,core.vstren,len(cov)))

        ss = ("{}. ({}) {}: {}"
              .format(i+1,core.sstren,core.__str__(fstr_f),str_of_cov(cov))
              for i,(core,cov) in enumerate(mc))
        return ss
        
    def __str__(self,fstr_f=None):
        return '\n'.join(self.strs(fstr_f))

    @property
    def typs(self):
        typs = [core.typ for core in self]
        return typs
    @property
    def ntyps(self):
        d = {'conj':0,'disj':0,'mix':0}
        for t in self.typs:
            d[t] = d[t] + 1

        return (d['conj'],d['disj'],d['mix'])
    
    @property
    def strens(self):
        """
        (strength,cores,sids)
        """
        strens = set(core.sstren for core in self)

        rs = []
        for stren in sorted(strens):
            cores = [c for c in self if c.sstren == stren]
            cov = set(sid for core in cores for sid in self[core])
            rs.append((stren,len(cores),len(cov)))
        return rs

    @property
    def strens_str(self): return self.str_of_strens(self.strens)

    @staticmethod
    def str_of_strens(strens):
        return ', '.join("({}, {}, {})".format(siz,ncores,ncov)
                         for siz,ncores,ncov in strens)

    

    # @property
    # def shared_graph(self):
    #     g = {}
    #     for core in self:
    #         if not core.settings:  #skip true
    #             continue
    #         for core_ in self:
    #             if core_ is not core and core_.settings:
    #                 if core.settings.is_superset(core_.settings):
    #                     if core not in d:
    #                         g[core] = set()
    #                     g[core].add(core_)
    #     return g

        
    
#Inference algorithm
class Infer(object):
    @staticmethod
    def infer(configs,core,dom):
        """
        Approximation in *conjunctive* form
        """
        if CM.__vdebug__:
            assert all(isinstance(c,Config) for c in configs) and configs, configs
            assert Core.is_maybe_core(core),core
            assert isinstance(dom,Dom),dom

        if core is None:  #not yet set
            core = min(configs,key=lambda c:len(c))
            core = Core((k,frozenset([v])) for k,v in core.iteritems())

        def f(k,s,ldx):
            s_ = set(s)
            for config in configs:
                if k in config:
                    s_.add(config[k])
                    if len(s_) == ldx:
                        return None
                else:
                    return None
            return s_

        vss = [f(k,vs,len(dom[k])) for k,vs in core.iteritems()]
        core = Core((k,frozenset(vs)) for k,vs in zip(core,vss) if vs)
        return core  

    @staticmethod
    def infer_cache(core,configs,dom,cache):
        if CM.__vdebug__:
            assert core is None or isinstance(core,Core),core
            assert all(isinstance(c,Config) for c in configs) and configs, configs
            assert isinstance(dom,Dom),dom
            assert isinstance(cache,dict),cache

        configs = frozenset(configs)
        key = (core,configs)
        if key not in cache:
            cache[key] = Infer.infer(configs,core,dom)
        return cache[key]

    @staticmethod
    def infer_sid(sid,core,cconfigs_d,configs_d,covs_d,dom,cache):
        if CM.__vdebug__:
            assert isinstance(sid,str),sid
            assert isinstance(core,PNCore),core
            assert isinstance(cconfigs_d,Configs_d) and cconfigs_d,cconfigs_d
            assert isinstance(configs_d,Configs_d),configs_d
            assert isinstance(covs_d,Covs_d),covs_d
            assert isinstance(dom,Dom),dom        
            assert isinstance(cache,dict),cache

        def _f(configs,cc,cd,_b):
            new_cc,new_cd = cc,cd
            if configs:
                new_cc = Infer.infer_cache(cc,configs,dom,cache)
                
            if do_comb_conj_disj and new_cc:
                configs_ = [c for c in _b() if c.c_implies(new_cc)]
                if configs_:
                    new_cd = Infer.infer_cache(cd,configs_,dom,cache)
                    if new_cd:
                        new_cd = Core((k,v) for (k,v) in new_cd.iteritems()
                                      if k not in new_cc)

            return new_cc,new_cd

        pc,pd,nc,nd = core
        
        pconfigs = [c for c in cconfigs_d if sid in cconfigs_d[c]]
 
        if nc is None:
            #never done nc, so has to consider all traces
            nconfigs = [c for c in configs_d if sid not in configs_d[c]]
        else:
            #done nc, so can do incremental traces
            nconfigs = [c for c in cconfigs_d if sid not in cconfigs_d[c]]
            
        _b = lambda: [c for c in configs_d if sid not in configs_d[c]]
        pc_,pd_ = _f(pconfigs,pc,pd,_b)
        
        _b = lambda: covs_d[sid]
        nc_,nd_ = _f(nconfigs,nc,nd,_b)
        return PNCore((pc_,pd_,nc_,nd_))

    @staticmethod
    def infer_covs(cores_d,cconfigs_d,configs_d,covs_d,dom):
        if CM.__vdebug__:
            assert isinstance(cores_d,Cores_d),cores_d
            assert isinstance(cconfigs_d,Configs_d) and cconfigs_d,cconfigs_d
            assert isinstance(configs_d,Configs_d),configs_d        
            assert all(c not in configs_d for c in cconfigs_d),cconfigs_d
            assert isinstance(covs_d,Covs_d),covs_d
            assert isinstance(dom,Dom),dom

        sids = set(cores_d.keys())
        #update configs_d and covs_d
        for config in cconfigs_d:
            for sid in cconfigs_d[config]:
                sids.add(sid)
                covs_d.add(sid,config)

            assert config not in configs_d, config
            configs_d[config]=cconfigs_d[config]

        cache = {}
        new_covs,new_cores = set(),set()  #updated stuff    
        for i,sid in enumerate(sorted(sids)):
            if sid in cores_d:
                core = cores_d[sid]
            else:
                core = PNCore.mk_default()
                new_covs.add(sid)

            core_ = Infer.infer_sid(sid,core,cconfigs_d,
                                         configs_d,covs_d,dom,cache)
            if not core_ == core: #progress
                new_cores.add(sid)
                cores_d[sid] = core_

        return new_covs,new_cores

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
        if CM.__vdebug__:
            assert isinstance(sid,str),sid
            assert isinstance(config,Config),config
        super(Covs_d,self).add_set(sid,config)

class Configs_d(CustDict):
    def __setitem__(self,config,cov):
        if CM.__vdebug__:
            assert isinstance(config,Config),config
            assert is_cov(cov),cov
        self.__dict__[config] = cov

    def __str__(self):
        ss = (c.__str__(self[c]) for c in self.__dict__)
        return '\n'.join("{}. {}".format(i+1,s) for i,s in enumerate(ss))
    
class IGen(object):
    """
    Main algorithm
    """
    def __init__(self,dom,get_cov,config_default=None):
        if CM.__vdebug__:
            assert isinstance(dom,Dom),dom
            assert callable(get_cov),get_cov
            assert (config_default is None or
                    isinstance(config_default,Config)), config_default
            
        self.dom = dom
        self.z3db = self.dom.z3db
        self.get_cov = get_cov
        self.config_default = config_default

    def go(self,seed=None,rand_n=None,tmpdir=None):
        """
        rand_n = None: use default interative mode
        rand_n = 0  : use init configs
        rand_n > 0  : use rand_n configs
        rand_n < 0  : use all possible configs
        """
        if CM.__vdebug__:
            assert isinstance(tmpdir,str) and os.path.isdir(tmpdir), tmpdir
            
        seed = seed if seed is not None else round(time(),2)
        random.seed(seed)
        logger.info("seed: {}, tmpdir: {}".format(seed,tmpdir))
        analysis = Analysis(tmpdir)
        analysis.save_pre(seed,self.dom)

        #some settings
        cur_iter = 1
        min_stren = 1
        cur_min_stren = min_stren
        cur_stuck = 0
        max_stuck = 3
        do_perturb = True #set this to True will not do any perturbing
        cores_d,configs_d,covs_d = Cores_d(),Configs_d(),Covs_d()
        sel_core = SCore.mk_default()
        ignore_sel_cores = set()

        #begin
        st = time()
        ct = st
        xtime_total = 0.0
        
        configs = self.gen_configs_init(rand_n,seed)
        if self.config_default:
            logger.debug("add default config")
            configs.append(self.config_default)
            
        cconfigs_d,xtime = self.eval_configs(configs)
        xtime_total += xtime
        new_covs,new_cores = Infer.infer_covs(cores_d,cconfigs_d,
                                              configs_d,covs_d,self.dom)
        while True:
            ct_ = time();itime = ct_ - ct;ct = ct_
            dtrace = DTrace(cur_iter,itime,xtime,
                            len(configs_d),len(covs_d),len(cores_d),
                            cconfigs_d,
                            new_covs,new_cores,
                            sel_core,
                            cores_d)
            dtrace.show()
            analysis.save_iter(cur_iter,dtrace)

            if rand_n is not None:
                break

            cur_iter += 1
            sel_core,configs = self.gen_configs_iter(
                set(cores_d.values()),ignore_sel_cores,
                cur_min_stren,configs_d)

            stop = False
            if sel_core is None:
                if do_perturb: #if already perturb
                    stop = True
                else:
                    do_perturb = True
                    logger.debug("perturb !")
                    sel_core,configs = self.gen_configs_iter(
                        set(cores_d.values()),set(),cur_min_stren,configs_d)
                    if sel_core is None:
                        stop = True
                        
                if stop:
                    cur_iter -= 1
                    logger.info('done after iter {}'.format(cur_iter))
                    break

            assert configs,configs
            cconfigs_d,xtime = self.eval_configs(configs)
            xtime_total += xtime
            new_covs,new_cores = Infer.infer_covs(
                cores_d,cconfigs_d,configs_d,covs_d,self.dom)

            if new_covs or new_cores: #progress
                cur_stuck = 0
                cur_min_stren = min_stren
            else: #no progress
                cur_stuck += 1
                if cur_stuck > max_stuck:
                    cur_stuck = 0
                    cur_min_stren += 1
                    print('cur_min_stren is now {}'.format(cur_min_stren))

        #postprocess
        pp_cores_d = cores_d.analyze(covs_d,self.dom)
        pp_cores_d.show_analysis(self.dom)
        itime_total = time() - st
        
        logger.info(Analysis.str_of_summary(
            seed,cur_iter,itime_total,xtime_total,
            len(configs_d),len(covs_d),tmpdir))
        logger.info("Done (seed {}, test {})".format(seed,random.randrange(100)))
        analysis.save_post(pp_cores_d,itime_total)
        
        return pp_cores_d,cores_d,configs_d,covs_d,self.dom

    #Shortcuts
    def go_full(self,tmpdir=None):
        return self.go(seed=None,rand_n=-1,tmpdir=tmpdir)
    def go_rand(self,rand_n,seed=None,tmpdir=None):
        return self.go(seed=seed,rand_n=rand_n,tmpdir=tmpdir)

    #Helper functions
    def eval_configs(self,configs):
        if CM.__vdebug__:
            assert (isinstance(configs,list) and
                    all(isinstance(c,Config) for c in configs)
                    and configs), configs
        st = time()
        cconfigs_d = Configs_d()
        for c in configs:
            if c in cconfigs_d: #skip
                continue

            sids,outps = self.get_cov(c)
            if analyze_outps:
                cconfigs_d[c]=outps
            else:
                cconfigs_d[c]=sids
        return cconfigs_d,time() - st

    def gen_configs_init(self,rand_n,seed):
        
        if not rand_n: #None or 0
            configs = self.dom.gen_configs_tcover1()
            logger.info("gen {} configs using tcover 1".format(len(configs)))
        elif rand_n > 0 and rand_n < self.dom.siz:        
            configs = self.dom.gen_configs_rand(rand_n)
            logger.info("gen {} rand configs".format(len(configs)))
        else:
            configs = self.dom.gen_configs_full()
            logger.info("gen all {} configs".format(len(configs)))

        configs = list(set(configs))
        assert configs, 'no initial configs created'
        return configs
        
    def gen_configs_iter(self,cores,ignore_sel_cores,min_stren,configs_d):
        if CM.__vdebug__:
            assert (isinstance(cores,set) and 
                    all(isinstance(c,PNCore) for c in cores)), cores
            assert (isinstance(ignore_sel_cores,set) and 
                    all(isinstance(c,SCore) for c in ignore_sel_cores)),\
                    ignore_sel_cores
            assert isinstance(configs_d,Configs_d),configs_d

        configs = []
        while True:
            sel_core = self.select_core(cores,ignore_sel_cores,min_stren)
            if sel_core is None:
                break

            configs = self.gen_configs_sel_core(sel_core,configs_d)
            configs = list(set(configs)) 
            if configs:
                break
            else:
                logger.debug("no cex's created for sel_core {}, try new core"
                             .format(sel_core))

        #self_core -> configs
        if CM.__vdebug__:
            assert not sel_core or configs, (sel_core,configs)
            assert all(c not in configs_d for c in configs), configs

        return sel_core, configs

    #Generate new configurations using z3
    def gen_configs_sel_core(self,sel_core,configs_d):
        """
        create configs by changing settings in core
        Also, these configs satisfy sat_core
        x=0,y=1  =>  [x=0,y=0,z=rand;x=0,y=2,z=rand;x=1,y=1;z=rand]
        """
        if CM.__vdebug__:
            assert isinstance(sel_core,SCore),sel_core

        configs = []            
        core,sat_core = sel_core

        #keep
        changes = []        
        if sel_core.keep and (len(self.dom) - len(core)):
            changes.append(core)

        #change
        _new = lambda : Core((k,core[k]) for k in core)
        for k in core:
            vs = self.dom[k]-core[k]
            for v in vs:
                new_core = _new()
                new_core[k] = frozenset([v])
                if sat_core:
                    for sk,sv in sat_core.iteritems():
                        assert sk not in new_core, sk
                        new_core[sk] = sv
                changes.append(new_core)

        existing_configs = [c.z3expr(self.z3db) for c in configs_d]                
        for changed_core in changes:
            core_expr = changed_core.z3expr(self.z3db,z3util.myAnd)
            model = self.get_sat_core(core_expr,z3util.myOr(existing_configs))
            if not model:
                continue
            config = self.config_of_model(model)
            configs.append(config)
            existing_configs.append(config.z3expr(self.z3db))
            if CM.__vdebug__:
                assert config.c_implies(changed_core)

        return configs

    def get_sat_core(self,core_expr,configs_expr):
        """
        Return a config satisfying core and not already in existing_configs

        # >>> dom = HDict([('a', frozenset(['1', '0'])), \
        # ('b', frozenset(['1', '0'])), ('c', frozenset(['1', '0', '2']))])
        # >>> z3db = z3db_of_dom(dom)

        # >>> c1 = HDict([('a', '0'), ('b', '0'), ('c', '0')])
        # >>> c2 = HDict([('a', '0'), ('b', '0'), ('c', '1')])
        # >>> c3 = HDict([('a', '0'), ('b', '0'), ('c', '2')])

        # >>> c4 = HDict([('a', '0'), ('b', '1'), ('c', '0')])
        # >>> c5 = HDict([('a', '0'), ('b', '1'), ('c', '1')])
        # >>> c6 = HDict([('a', '0'), ('b', '1'), ('c', '2')])

        # >>> c7 = HDict([('a', '1'), ('b', '0'), ('c', '0')])
        # >>> c8 = HDict([('a', '1'), ('b', '0'), ('c', '1')])
        # >>> c9 = HDict([('a', '1'), ('b', '0'), ('c', '2')])

        # >>> c10 = HDict([('a', '1'), ('b', '1'), ('c', '0')])
        # >>> c11 = HDict([('a', '1'), ('b', '1'), ('c', '1')])
        # >>> c12 = HDict([('a', '1'), ('b', '1'), ('c', '2')])

        # >>> configs = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11]
        # >>> configs_expr = z3util.myOr([z3expr_of_config(c,z3db) for c in configs])

        # >>> model = z3_get_sat_core(None,configs_expr,z3db)
        # >>> assert config_of_model(model,dom) == c12

        # >>> core = HDict([('a',frozenset(['1']))])
        # >>> core_expr = z3expr_of_core(core,z3db,z3util.myAnd)
        # >>> model = z3_get_sat_core(core_expr,configs_expr,z3db)
        # >>> assert config_of_model(model,dom) == c12


        # >>> core = HDict([('a',frozenset(['0']))])
        # >>> core_expr = z3expr_of_core(core,z3db,z3util.myAnd)
        # >>> model = z3_get_sat_core(core_expr,configs_expr,z3db)
        # >>> assert model is None

        # >>> core = HDict([('c',frozenset(['0','1']))])
        # >>> core_expr = z3expr_of_core(core,z3db,z3util.myAnd)
        # >>> model = z3_get_sat_core(core_expr,configs_expr,z3db)
        # >>> assert model is None

        # >>> core = HDict([('c',frozenset(['0','2']))])
        # >>> core_expr = z3expr_of_core(core,z3db,z3util.myAnd)
        # >>> model = z3_get_sat_core(core_expr,configs_expr,z3db)
        # >>> assert model == {'a': '1', 'c': '2', 'b': '1'}
        """
        if CM.__vdebug__:
            assert core_expr is None or z3.is_expr(core_expr),core_expr
            assert z3.is_expr(configs_expr),configs_expr

        not_configs_expr = z3.Not(configs_expr)
        if core_expr:
            f = z3.And(core_expr,not_configs_expr)
        else:
            f = not_configs_expr

        models = z3util.get_models(f,k=1)
        assert models is not None, models  #z3 cannot solve this
        if not models:  #not satisfy
            return None
        assert len(models)==1,models
        model = models[0]
        model = dict((str(v),str(model[v])) for v in model)
        return model

    def config_of_model(self,model):
        """
        Obtain a config from a model
        """
        if CM.__vdebug__:
            assert isinstance(model,dict),model

        _f = lambda k: (model[k] if k in model
                        else random.choice(list(self.dom[k])))
        config = Config((k,_f(k)) for k in self.dom)
        return config
    

    @staticmethod
    def select_core(pncores,ignore_sel_cores,min_stren):
        """
        Returns either None or SCore
        """
        if CM.__vdebug__:
            assert all(isinstance(c,PNCore) for c in pncores) and pncores,pncores
            assert (isinstance(ignore_sel_cores,set) and
                    all(isinstance(c,SCore) for c in ignore_sel_cores)),\
                    ignore_sel_cores

        sel_cores = []
        for (pc,pd,nc,nd) in pncores:
            #if can add pc then don't cosider pd (i.e., refine pc first)
            if pc and (pc,None) not in ignore_sel_cores:
                sc = SCore((pc,None))
                if pd is None: sc.set_keep()
                sel_cores.append(sc)
                    
            elif pd and (pd,pc) not in ignore_sel_cores:
                sc = SCore((pd,pc))
                sel_cores.append(sc)

            if nc and (nc,None) not in ignore_sel_cores:
                sc = SCore((nc,None))
                if nd is None: sc.set_keep()
                sel_cores.append(sc)

            elif nd and (nd,nc) not in ignore_sel_cores:
                sc = SCore((nd,nc))
                sel_cores.append(sc)
                
        sel_cores = [c for c in sel_cores if c.sstren >= min_stren]

        if sel_cores:
            sel_core = max(sel_cores,key=lambda c: (c.sstren,c.vstren))
            ignore_sel_cores.add(sel_core)
        else:
            sel_core = None

        return sel_core


### Analyzing results    
class DTrace(object):
    def __init__(self,citer,itime,xtime,
                 nconfigs,ncovs,ncores,
                 cconfigs_d,new_covs,new_cores,
                 sel_core,cores_d):

        self.citer = citer
        self.itime = itime
        self.xtime = xtime
        self.nconfigs = nconfigs
        self.ncovs = ncovs
        self.ncores = ncores
        self.cconfigs_d = cconfigs_d
        self.new_covs = new_covs
        self.new_cores = new_cores
        self.sel_core = sel_core
        self.cores_d = cores_d
        
    def show(self):
        logger.info("ITER {}, ".format(self.citer) +
                    "{}s, ".format(self.itime) +
                    "{}s eval, ".format(self.xtime) +
                    "total: {} configs, {} covs, {} cores, "
                    .format(self.nconfigs,self.ncovs,self.ncores) +
                    "new: {} configs, {} covs, {} updated cores, "
                    .format(len(self.cconfigs_d),
                            len(self.new_covs),len(self.new_cores)) +
                    "{}".format("** progress **"
                                if self.new_covs or self.new_cores else ""))

        logger.debug('select core: ({}) {}'.format(self.sel_core.sstren,
                                                   self.sel_core))
        logger.debug('create {} configs'.format(len(self.cconfigs_d)))
        logger.detail("\n"+str(self.cconfigs_d))

        mcores_d = self.cores_d.merge()
        logger.debug("infer {} interactions".format(len(mcores_d)))
        logger.detail('\n{}'.format(mcores_d))
        logger.info("strens: {}".format(mcores_d.strens_str))

class Analysis(object):
    def __init__(self,tmpdir):
        self.tmpdir = tmpdir

    def save_pre(self,seed,dom):
        CM.vsave(os.path.join(self.tmpdir,'pre'),(seed,dom))
    def save_post(self,pp_cores_d,itime_total):
        CM.vsave(os.path.join(self.tmpdir,'post'),(pp_cores_d,itime_total))
    def save_iter(self,cur_iter,dtrace):
        CM.vsave(os.path.join(self.tmpdir,'{}.tvn'.format(cur_iter)),dtrace)

    @staticmethod
    def load_pre(dir_):
        seed,dom = CM.vload(os.path.join(dir_,'pre'))
        return seed,dom
    @staticmethod
    def load_post(dir_):
        pp_cores_d,itime_total = CM.vload(os.path.join(dir_,'post'))
        return pp_cores_d,itime_total
    @staticmethod
    def load_iter(dir_,f):
        dtrace =CM.vload(os.path.join(dir_,f))
        return dtrace
    
    @staticmethod
    def str_of_summary(seed,iters,itime,xtime,nconfigs,ncovs,tmpdir):
        ss = ["Seed {}".format(seed),
              "Iters {}".format(iters),
              "Time ({}s, {}s)".format(itime,xtime),
              "Configs {}".format(nconfigs),
              "Covs {}".format(ncovs),
              "Tmpdir {}".format(tmpdir)]
        return "Summary: " + ', '.join(ss)
    
    @staticmethod
    def replay(dir_):
        """
        Replay execution info from saved info in dir_
        """

        def load_dir(dir_):
            seed,dom = Analysis.load_pre(dir_)
            dts = [Analysis.load_iter(dir_,f)
                   for f in os.listdir(dir_) if f.endswith('.tvn')]
            try:
                pp_cores_d,itime_total = Analysis.load_post(dir_)
            except IOError:
                logger.error("post info not avail")
                pp_cores_d,itime_total = None,None
            return seed,dom,dts,pp_cores_d,itime_total

        seed,dom,dts,pp_cores_d,itime_total = load_dir(dir_)

        #print info
        logger.info("replay dir: '{}'".format(dir_))
        logger.info('seed: {}'.format(seed))
        logger.debug(dom.__str__())
        
        #print iterations
        for dt in sorted(dts,key=lambda dt: dt.citer):
            dt.show()

        #print postprocess results
        mcores_d = pp_cores_d.show_analysis(dom)
        
        #print summary
        xtime_total = itime_total - sum(dt.xtime for dt in dts)
        logger.info(Analysis.str_of_summary(
            seed,len(dts),itime_total,xtime_total,dt.nconfigs,dt.ncovs,dir_))
            
        return (len(dts),len(mcores_d),
                itime_total,xtime_total,dt.nconfigs,dt.ncovs,
                mcores_d.strens,mcores_d.ntyps)

    @staticmethod
    def replay_dirs(dir_):
        dir_ = getpath(dir_)
        logger.info("replay_dirs '{}'".format(dir_))
        
        niters_total = 0
        nresults_total = 0        
        nitime_total = 0
        nxtime_total = 0    
        nconfigs_total = 0
        ncovs_total = 0
        strens_s = []
        ntyps_s = []

        #modified by ugur
        niters_arr = []
        nresults_arr = []
        nitime_arr = []
        nxtime_arr = [] 
        nconfigs_arr = []
        ncovs_arr = []
        counter = 0
        csv_arr = []

        for rdir in sorted(os.listdir(dir_)):
            rdir = os.path.join(dir_,rdir)
            (niters,nresults,itime,xtime,nconfigs,ncovs,strens,ntyps) = Analysis.replay(rdir)
            niters_total += niters
            nresults_total += nresults
            nitime_total += itime
            nxtime_total += xtime
            nconfigs_total += nconfigs
            ncovs_total += ncovs
            strens_s.append(strens)
            ntyps_s.append(ntyps)

            niters_arr.append(niters)
            nresults_arr.append(nresults)
            nitime_arr.append(itime)
            nxtime_arr.append(xtime)
            nconfigs_arr.append(nconfigs)
            ncovs_arr.append(ncovs)
            csv_arr.append("{},{},{},{},{},{},{},{},{}".format(counter,niters,nresults,itime,xtime,nconfigs,ncovs,','.join(map(str, ntyps)),','.join(map(str, strens))))
            counter += 1

        nruns_total = float(len(strens_s))

        ss = ["iter {}".format(niters_total/nruns_total),
              "results {}".format(nresults_total/nruns_total),
              "time {}".format(nitime_total/nruns_total),
              "xtime {}".format(nxtime_total/nruns_total),
              "configs {}".format(nconfigs_total/nruns_total),
              "covs {}".format(ncovs_total/nruns_total)]
        logger.info("STATS of {} runs (averages): {}".format(nruns_total,', '.join(ss)))
        
        ssMed = ["iter {}".format(numpy.median(niters_arr)),
              "results {}".format(numpy.median(nresults_arr)),
              "time {}".format(numpy.median(nitime_arr)),
              "xtime {}".format(numpy.median(nxtime_arr)),
              "configs {}".format(numpy.median(nconfigs_arr)),
              "covs {}".format(numpy.median(ncovs_arr))]
        logger.info("STATS of {} runs (medians) : {}".format(nruns_total,', '.join(ssMed)))
        
        ssSIQR = ["iter {}".format(Analysis.siqr(niters_arr)),
              "results {}".format(Analysis.siqr(nresults_arr)),
              "time {}".format(Analysis.siqr(nitime_arr)),
              "xtime {}".format(Analysis.siqr(nxtime_arr)),
              "configs {}".format(Analysis.siqr(nconfigs_arr)),
              "covs {}".format(Analysis.siqr(ncovs_arr))]
        logger.info("STATS of {} runs (SIQR)   : {}".format(nruns_total,', '.join(ssSIQR)))

        sres = {}
        for i,strens in enumerate(strens_s):
            logger.debug("run {}: {}".format(i+1,Mcores_d.str_of_strens(strens)))
            for strength,ninters,ncov in strens:
                if strength not in sres:
                    sres[strength] = ([ninters],[ncov])
                else:
                    inters,covs = sres[strength]
                    inters.append(ninters)
                    covs.append(ncov)

        ss = []
        medians = []
        siqrs = []
        tmp = []
        tex_table4=[]
        tex_table5=[]
        for strength in sorted(sres):
            inters,covs = sres[strength]
            length=len(inters)
            for num in range(length,int(nruns_total)):
                inters.append(0)
                covs.append(0)
            ss.append("({}, {}, {})".format(strength,sum(inters)/nruns_total,sum(covs)/nruns_total))
            medians.append("({}, {}, {})".format(strength, numpy.median(inters), numpy.median(covs)))
            siqrs.append("({}, {}, {})".format(strength, Analysis.siqr(inters), Analysis.siqr(covs)))
            tmp.append("{},{})".format(strength,','.join(map(str, inters))))
            tex_table4.append("{} \\mso{{{}}}{{{}}}".format(strength,numpy.median(inters),Analysis.siqr(inters)))
            tex_table5.append("{} \\mso{{{}}}{{{}}}".format(strength,numpy.median(covs),Analysis.siqr(covs)))
        
        logger.info("interaction strens averages: {}".format(', '.join(ss)))
        logger.info("interaction strens medians : {}".format(', '.join(medians)))
        logger.info("interaction strens SIQRs   : {}".format(', '.join(siqrs)))
        #logger.info("interactions arrays   : {}".format('\n'.join(tmp)))
        
        conjs = [c for c,_,_ in ntyps_s]
        disjs = [d for _,d,_ in ntyps_s]
        mixs = [m for _,_,m in ntyps_s]
        
        length=len(conjs)
        for num in range(length,int(nruns_total)):
            conjs.append(0)
        
        length=len(disjs)
        for num in range(length,int(nruns_total)):
            disjs.append(0)
        
        length=len(mixs)
        for num in range(length,int(nruns_total)):
            mixs.append(0)
        
        #logger.info("conjs array: {}".format(', '.join(map(str, conjs))))
        #logger.info("disjs array: {}".format(', '.join(map(str, disjs))))
        #logger.info("mixs  array: {}".format(', '.join(map(str, mixs))))

        nconjs = sum(conjs)/nruns_total
        ndisjs = sum(disjs)/nruns_total
        nmixs  = sum(mixs)/nruns_total
        
        logger.info("interaction typs (averages): conjs {}, disjs {}, mixeds {}"
                    .format(nconjs,ndisjs,nmixs))            
        
        logger.info("interaction typs (medians) : conjs {}, disjs {}, mixeds {}"
                    .format(numpy.median(conjs),numpy.median(disjs),numpy.median(mixs)))
        
        logger.info("interaction typs (SIQRs)   : conjs {}, disjs {}, mixeds {}"
                    .format(Analysis.siqr(conjs),Analysis.siqr(disjs),Analysis.siqr(mixs)))

        logger.info("tex_table4:{}".format(' & '.join(tex_table4)))
        logger.info("tex_table5:{}".format(' & '.join(tex_table5)))

        logger.info("CVSs\n{}".format('\n'.join(csv_arr)))
        #end of modification

    @staticmethod
    def siqr(arr):
        return (numpy.percentile(arr, 75, interpolation='higher') - 
                numpy.percentile(arr, 25, interpolation='lower'))/2
    
    @staticmethod
    def debug_find_configs(sid,configs_d,find_in):
        if find_in:
            cconfigs_d = dict((c,cov) for c,cov in configs_d.iteritems()
                           if sid in cov)
        else:
            cconfigs_d = dict((c,cov) for c,cov in configs_d.iteritems()
                              if sid not in cov)

        logger.info(cconfigs_d)

