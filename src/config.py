import tempfile
from time import time
import os.path
import itertools
import random

import z3
import z3util
from vu_common import HDict
import vu_common as CM

from collections import namedtuple

logger = CM.VLog('config')
logger.level = CM.VLog.DEBUG
CM.VLog.PRINT_TIME = True
CM.__vdebug__ = True  #IMPORTANT: TURN OFF WHEN DO REAL RUN!!
do_comb_conj_disj = True
show_cov = True

getpath = lambda f: os.path.realpath(os.path.expanduser(f))

#Data Structures
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

class Dom(HDict):
    def __init__(self,dom):
        HDict.__init__(self,dom)

        if CM.__vdebug__:
            assert self and all(is_csetting(s) for s in self.iteritems()), self
            
    def __str__(self):
        """
        >>> print Dom([('x',frozenset(['1','2'])),('y',frozenset(['1']))])
        2 vars and 2 poss configs
        x: (2) 1,2
        y: (1) 1
        """
        s = "{} vars and {} poss configs".format(len(self),self.siz)
        s_detail = '\n'.join("{}: ({}) {}".format(k,len(vs),str_of_valset(vs))
                             for k,vs in self.iteritems())

        s = "{}\n{}".format(s,s_detail)
        return s

    @property
    def siz(self): return CM.vmul(len(vs) for vs in self.itervalues())
        
    @staticmethod
    def get_dom(dom_file):
        """
        Read domain info from a file
        """
        if CM.__vdebug__:
            assert os.path.isfile(dom_file), domfile

        def get_lines(lines):
            rs = (line.split() for line in lines)
            rs = ((parts[0],frozenset(parts[1:])) for parts in rs)
            return rs

        dom = Dom(get_lines(CM.iread_strip(dom_file)))

        config_default = None
        dom_file_default = dom_file+'.default'
        if os.path.isfile(dom_file_default):
            rs = get_dom_lines(CM.iread_strip(dom_file_default))
            config_default = Config((k,list(rs[k])[0]) for k in rs)

        return dom,config_default

    #Methods to generate configurations
    def gen_configs_full(self):
        ns,vs = itertools.izip(*self.iteritems())
        configs = [Config(zip(ns,c)) for c in itertools.product(*vs)]
        return configs

    def gen_configs_rand(self,rand_n):
        if CM.__vdebug__:
            assert 0 < rand_n < dom.siz, (rand_n,self.siz)

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
        while dom_used:
            configs.append(mk())

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

def str_of_configs(configs):
    if CM.__vdebug__:
        assert all(isinstance(c,Config) for c in configs), configs

    return '\n'.join("{}. {}".format(i+1,c) for i,c in enumerate(configs))

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
    
class MCore(tuple):
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
            assert self.pc is None or isinstance(self.pc,Core) and self.pc, self.pc
            assert self.pd is None or isinstance(self.pd,Core) and self.pd, self.pd
            assert self.nc is None or isinstance(self.nc,Core) and self.nc, self.nc
            assert self.nd is None or isinstance(self.nd,Core) and self.nd, self.nd
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
            ss = ','.join([p_ss,n_ss]) + '***'

        return ss

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
        z3db = z3db_of_dom(dom)

        def _f(cc,cd):
            fs = []
            if cc:
                f = z3expr_of_core(cc,z3db,myf=z3util.myAnd)
                fs.append(f)
            if cd:
                cd_n = cd.neg(dom)
                f = z3expr_of_core(cd_n,z3db,myf=z3util.myOr)
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

class SCore(MCore):
    def __init__(self,(mc,sc)):
        super(SCore,self).__init__((mc,sc))

    @property
    def mc(self): return self[0]

    @property
    def sc(self): return self[1]

    def __str__(self):
        ss = []
        if self.mc: ss.append("mc: {}".format(self.mc))
        if self.sc: ss.append("sc: {}".format(self.sc))
        return ';'.join(ss)        
        
    @staticmethod
    def mk_default(): return MCore((None,None))
    
class Cores_d(dict):
    def __init__(self,cores_d={}):
        dict.__init__(self,cores_d)

        if CM.__vdebug__:
            assert all(isinstance(sid,str) and isinstance(c,PNCore)
                       for sid,c in self.iteritems()), self

    def __str__(self):
        return '\n'.join("{}. {}: {}"
                         .format(i+1,sid,self[sid].__str__(fstr_f=None))
                         for i,sid in enumerate(sorted(self)))

    def merge(self):
        mcores_d = {}
        for sid,core in self.iteritems():
            if core in mcores_d:
                mcores_d[core].add(sid)
            else:
                mcores_d[core] = set([sid])

        return Mcores_d(mcores_d)

    def analyze(self,covs_d,dom):
        if CM.__vdebug__:
            assert is_covs_d(covs_d),covs_d
            assert len(self) == len(covs_d), (len(self),len(covs_d))
            assert isinstance(dom,Dom),dom

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
        rs_simplify = []
        for sid,core in rs_verify:
            if core not in scache:
                core_ = core.simplify(dom)
                scache[core]=core_
                show_compare(sid,core,core_)
            else:
                core_ = scache[core]
            rs_simplify.append((sid,core_))

        return Cores_d(rs_simplify)    

class Mcores_d(dict):
    def __init__(self,mcores_d):
        dict.__init__(self,mcores_d)

        if CM.__vdebug__:
            assert self and all(isinstance(c,PNCore) and is_cov(cov)
                                for c,cov in mcores_d.iteritems()), self

    def __str__(self,fstr_f=None):
        if CM.__vdebug__:
            assert fstr_f is None or callable(fstr_f),fstr_f

        mc = sorted(self.iteritems(),
                    key=lambda (core,cov): (core.sstren,core.vstren,len(cov)))

        ss = ("{}. ({}) {}: {}"
              .format(i+1,core.sstren,core.__str__(fstr_f),str_of_cov(cov))
              for i,(core,cov) in enumerate(mc))

        return '\n'.join(ss)

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
    def strens_str(self):
        return ', '.join("({},{},{})".format(siz,ncores,ncov)
                         for siz,ncores,ncov in self.strens)

DTrace = namedtuple("DTrace",
                    "citer etime ctime "
                    "nconfigs ncovs ncores "
                    "cconfigs_d "
                    "new_covs new_cores "
                    "sel_core cores_d")

#Inference algorithm

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

def infer_cache(core,configs,dom,cache):
    if CM.__vdebug__:
        assert core is None or isinstance(core,Core),core
        assert all(isinstance(c,Config) for c in configs) and configs, configs
        assert isinstance(dom,Dom),dom
        assert isinstance(cache,dict),cache

    configs = frozenset(configs)
    key = (core,configs)
    if key not in cache:
        cache[key] = infer(configs,core,dom)
    return cache[key]

def infer_sid(sid,core,cconfigs_d,configs_d,covs_d,dom,cache):
    if CM.__vdebug__:
        assert isinstance(sid,str),sid
        assert isinstance(core,PNCore),core
        assert is_configs_d(cconfigs_d) and cconfigs_d,cconfigs_d
        assert is_configs_d(configs_d),configs_d
        assert is_covs_d(covs_d),covs_d
        assert isinstance(dom,Dom),dom        
        assert isinstance(cache,dict),cache

    def _f(configs,cc,cd,_b):
        new_cc,new_cd = cc,cd
        if configs:
            new_cc = infer_cache(cc,configs,dom,cache)
        if do_comb_conj_disj and new_cc:
            configs_ = [c for c in _b() if c.c_implies(new_cc)]
            if configs_:
                new_cd = infer_cache(cd,configs_,dom,cache)
                if new_cd:
                    new_cd = Core((k,v) for (k,v) in new_cd.iteritems()
                                  if k not in new_cc)
            
        return new_cc,new_cd

    pconfigs,nconfigs = [],[]
    for config in cconfigs_d:
        if sid in cconfigs_d[config]:
            pconfigs.append(config)
        else:
            nconfigs.append(config)
            
    pc,pd,nc,nd = core
    _b = lambda: [c for c in configs_d if sid not in configs_d[c]]
    pc_,pd_ = _f(pconfigs,pc,pd,_b)
    _b = lambda: covs_d[sid]
    nc_,nd_ = _f(nconfigs,nc,nd,_b)    
    return PNCore((pc_,pd_,nc_,nd_))

def is_covs_d(covs_d):
    return (isinstance(covs_d,dict) and
            all(isinstance(sid,str) and isinstance(configs,set)
                and all(isinstance(c,Config) for c in configs)
                for sid,configs in covs_d.iteritems()))

def is_configs_d(configs_d):
    return (isinstance(configs_d,dict) and
            all(isinstance(config,Config) and is_cov(cov)
                for config,cov in configs_d.iteritems()))

def str_of_configs_d(configs_d):
    ss = (c.__str__(configs_d[c]) for c in configs_d)
    return '\n'.join("{}. {}".format(i+1,s) for i,s in enumerate(ss))
    
def infer_covs(cores_d,cconfigs_d,configs_d,covs_d,dom):
    if CM.__vdebug__:
        assert isinstance(cores_d,Cores_d),cores_d
        assert is_configs_d(cconfigs_d) and cconfigs_d,cconfigs_d
        assert is_configs_d(configs_d),configs_d        
        assert all(c not in configs_d for c in cconfigs_d),cconfigs_d
        assert is_covs_d(covs_d),covs_d
        assert isinstance(dom,Dom),dom
        
    sids = set(cores_d.keys())
    #update configs_d and covs_d
    for config in cconfigs_d:
        for sid in cconfigs_d[config]:
            sids.add(sid)
            if sid not in covs_d:
                covs_d[sid]=set()
            covs_d[sid].add(config)

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
        
        core_ = infer_sid(sid,core,cconfigs_d,
                          configs_d,covs_d,dom,cache)
                          
        if not core_ == core: #progress
            new_cores.add(sid)
            cores_d[sid] = core_

    return new_covs,new_cores


#Interpretation algorithm


#Interative algorithm
def gen_configs_core_z3(core,sat_core,configs_d,z3db,dom):
    """
    create configs by changing settings in core
    Also, these configs satisfy sat_core
    x=0,y=1  =>  [x=0,y=0,z=rand;x=0,y=2,z=rand;x=1,y=1;z=rand]
    """
    if CM.__vdebug__:
        assert isinstance(core,Core) and core, core
        assert (sat_core is None or isinstance(sat_core,Core) and
                all(k not in core for k in sat_core)),sat_core
        assert isinstance(dom,Dom),dom

    _new = lambda : Core((k,core[k]) for k in core)
    changes = []
    for k in core:
        vs = dom[k]-core[k]
        for v in vs:
            new_core = _new()
            new_core[k] = frozenset([v])
            if sat_core:
                for sk,sv in sat_core.iteritems():
                    assert sk not in new_core, sk
                    new_core[sk] = sv
            changes.append(new_core)

    existing_configs_expr = z3util.myOr([z3expr_of_config(c,z3db)
                                         for c in configs_d])

    configs = []
    for changed_core in changes:
        core_expr = z3expr_of_core(changed_core,z3db,z3util.myAnd)
        model = z3_get_sat_core(core_expr,existing_configs_expr,z3db)
        if not model:
            continue
        config = config_of_model(model,dom)
        configs.append(config)
        if CM.__vdebug__:
            assert config.c_implies(changed_core)
                
    return configs

def select_core(pncores,ignore_sel_cores):
    """
    Returns either None or SCore
    """
    if CM.__vdebug__:
        assert all(isinstance(c,PNCore) for c in pncores) and pncores,pncores
        assert (isinstance(ignore_sel_cores,set) and
                all(isinstace(c,SCore) for c in ignore_sel_cores)),\
                ignore_sel_cores
                    
    sel_cores = []
    for (pc,pd,nc,nd) in pncores:
        if pc and (pc,None) not in ignore_sel_cores:
            sel_cores.append(SCore((pc,None)))
        elif pd and (pd,pc) not in ignore_sel_cores:
            sel_cores.append(SCore((pd,pc)))

        if nc and (nc,None) not in ignore_sel_cores:
            sel_cores.append(SCore((nc,None)))
        elif nd and (nd,nc) not in ignore_sel_cores:
            sel_cores.append(SCore((nd,nc)))

    if sel_cores:
        sel_core = max(sel_cores,key=lambda c: (c.sstren,c.vstren))
        ignore_sel_cores.add(sel_core)
    else:
        sel_core = None

    return sel_core

def eval_configs(configs,get_cov):
    if CM.__vdebug__:
        assert (isinstance(configs,list) and
                all(isinstance(c,Config) for c in configs)
                and configs), configs
        assert callable(get_cov),get_cov
    st = time()
    cconfigs_d = {}
    for c in configs:
        cconfigs_d[c] = get_cov(c)
    return cconfigs_d,time() - st

def gen_configs_init(rand_n,seed,dom):
    if not rand_n: #None or 0
        configs = dom.gen_configs_tcover1()
        logger.info("gen {} configs using tcover 1".format(len(configs)))
    elif rand_n > 0 and rand_n < dom.siz:        
        configs = dom.gen_configs_rand(rand_n)
        logger.info("gen {} rand configs".format(len(configs)))
    else:
        configs = dom.gen_configs_full()
        logger.info("gen all {} configs".format(len(configs)))

    configs = list(set(configs))
    assert configs, 'no initial configs created'
    return configs

def gen_configs_iter(cores,ignore_sel_cores,configs_d,z3db,dom):
    if CM.__vdebug__:
        assert (isinstance(cores,set) and 
                all(isinstance(c,PNCore) for c in cores)), cores
        assert (isinstance(ignore_sel_cores,set) and 
                all(isinstance(c,SCore) for c in ignore_sel_cores)),\
                ignore_sel_cores
        assert is_configs_d(configs_d),configs_d
        assert isinstance(z3db,dict),z3db
        assert isinstance(dom,Dom),dom

    configs = []
    while True:
        sel_core = select_core(cores,ignore_sel_cores)
        if sel_core is None:
            break
        mc,sc = sel_core
        configs = gen_configs_core_z3(mc,sc,configs_d,z3db,dom)
        configs = list(set(configs))        
        if configs:
            break
        else:
            logger.debug("no sample created for sel_core {}".format(sel_core))
        
    #self_core -> configs
    if CM.__vdebug__:
        assert not sel_core or configs, (sel_core,configs)
        assert all(c not in configs_d for c in configs), configs
        
    return sel_core, configs

#Iterative Algorithm
def igen(dom,get_cov,seed=None,rand_n=None,config_default=None):
    """
    cover_siz=(0,n):  generates n random configs
    cover_siz=(0,-1):  generates full random configs

    rand_n = None or 0: use default interative mode
    rand_n > 0   : use rand_n configs
    rand_n < 0  : use all possible configs
    """
    if CM.__vdebug__:
        assert isinstance(dom,Dom),dom
        assert callable(get_cov),get_cov
        assert (config_default is None or
                isinstance(config_default,Config)), config_default

    seed = round(time(),2) if seed is None else float(seed)
    random.seed(seed)
    tmpdir = tempfile.mkdtemp(dir='/var/tmp',prefix="vu")
        
    logger.info("seed: {}, tmpdir: {}".format(seed,tmpdir))
    
    #save info
    iobj = (seed,dom); CM.vsave(os.path.join(tmpdir,'info'),iobj)
    
    #some settings
    cur_iter = 1
    cur_stuck= 0
    cores_d,configs_d,covs_d = Cores_d(),{},{}
    sel_core = SCore.mk_default()
    ignore_sel_cores = set()
    
    #begin
    st = time()
    ct = st
    cov_time = 0.0
    configs = gen_configs_init(rand_n,seed,dom)
    if config_default: configs.append(config_default)
    cconfigs_d,ctime = eval_configs(configs,get_cov)
    cov_time += ctime

    new_covs,new_cores = infer_covs(cores_d,cconfigs_d,
                                    configs_d,covs_d,dom)
    z3db = z3db_of_dom(dom)
    while True:

        #save info
        ct_ = time();etime = ct_ - ct;ct = ct_
        dtrace = DTrace(cur_iter,etime,cov_time,
                        len(configs_d),len(covs_d),len(cores_d),
                        cconfigs_d,
                        new_covs,new_cores,
                        sel_core,
                        cores_d)
        dfile = os.path.join(tmpdir,"{}.tvn".format(cur_iter))
        CM.vsave(dfile,dtrace)
        show_dtrace(dtrace)
        
        if rand_n:
            break
        
        cur_iter += 1
        sel_core,configs = gen_configs_iter(set(cores_d.values()),
                                            ignore_sel_cores,
                                            configs_d,z3db,dom)
        if sel_core is None:
            cur_iter -= 1
            logger.info('done after iter {}'.format(cur_iter))
            break

        assert configs,configs
        cconfigs_d,ctime = eval_configs(configs,get_cov)
        cov_time += ctime
        new_covs,new_cores = infer_covs(cores_d,cconfigs_d,configs_d,covs_d,dom)

        if new_covs or new_cores: #progress
            cur_stuck = 0
        else: #no progress
            cur_stuck += 1

    pp_cores_d = postprocess(cores_d,covs_d,dom)
    show_analyzed_cores_d(pp_cores_d,dom)
    CM.vsave(os.path.join(tmpdir,'postprocess'),pp_cores_d)
    
    logger.info(str_of_summary(seed,cur_iter,time()-st,cov_time,
                               len(configs_d),len(covs_d),
                               tmpdir))
    logger.info("Done (seed {}, test {})".format(seed,random.randrange(100)))
    return pp_cores_d,cores_d,configs_d,covs_d,dom

#Shortcuts
def igen_full(dom,get_cov):
    return igen(dom,get_cov,seed=None,rand_n=-1,config_default=None)
                
def igen_rand(dom,get_cov,rand_n,seed=None):
    return igen(dom,get_cov,seed=seed,rand_n=rand_n,config_default=None)
                

#postprocess
def show_analyzed_cores_d(cores_d,dom):
    if CM.__vdebug__:
        assert isinstance(cores_d,Cores_d),cores_d

    mcores_d = cores_d.merge()
    fstr_f=lambda c: c.fstr(dom)
    logger.debug("mcores_d has {} items\n".format(len(mcores_d)) +
                 mcores_d.__str__(fstr_f)) 
    logger.info("strens: {}".format(mcores_d.strens_str))
                
def postprocess_rs(rs): return postprocess(rs[1],rs[3],rs[4])

def postprocess(cores_d,covs_d,dom):
    if CM.__vdebug__:
        assert isinstance(cores_d,Cores_d),cores_d
        assert is_covs_d(covs_d),covs_d
        assert isinstance(dom,Dom),dom

    cores_d = cores_d.analyze(covs_d,dom)
    return cores_d
        
def reformat((pc,pd,nc,nd),dom):
    if pd: pd.neg(dom) 
    if nc: nc.neg(dom) 
    return (pc,pd,nc,nd)


#Miscs
def debug_find_configs(sid,configs_d,find_in):
    if find_in:
        cconfigs_d = dict((c,cov) for c,cov in configs_d.iteritems()
                       if sid in cov)
    else:
        cconfigs_d = dict((c,cov) for c,cov in configs_d.iteritems()
                          if sid not in cov)

    print str_of_configs_d(cconfigs_d)
        

#generate configs


def show_dtrace(dt):
    if CM.__vdebug__:
        assert isinstance(dt,DTrace)
        
    logger.info("ITER {}, ".format(dt.citer) +
                "{}s, ".format(dt.etime) +
                "{}s eval, ".format(dt.ctime) +
                "total: {} configs, {} covs, {} cores, "
                .format(dt.nconfigs,dt.ncovs,dt.ncores) +
                "new: {} configs, {} covs, {} updated cores, "
                .format(len(dt.cconfigs_d),len(dt.new_covs),len(dt.new_cores)) +
                "{}".format("** progress **"
                            if dt.new_covs or dt.new_cores else ""))

    logger.debug('select core: ({}) {}'.format(dt.sel_core.sstren, dt.sel_core))
    logger.debug('create {} configs'.format(len(dt.cconfigs_d)))
    logger.detail('\n' + str_of_configs_d(dt.cconfigs_d))

    mcores_d = dt.cores_d.merge()
    logger.debug("infer {} interactions".format(len(mcores_d)))
    logger.detail('\n{}'.format(mcores_d))
    logger.info("strens: {}".format(mcores_d.strens_str))


def str_of_summary(seed,iters,ntime,ctime,nconfigs,ncovs,tmpdir):
    s = ("Summary: " +
         "Seed {}, ".format(seed) +
         "Iters {}, ".format(iters) +
         "Time ({}s, {}s), ".format(ntime,ctime) +
         "Configs {}, ".format(nconfigs) +
         "Covs {}, ".format(ncovs) +
         "Tmpdir {}".format(tmpdir))
    return s

def replay(dir_):
    """
    Replay execution info from saved info in dir_
    """
    def load_dir(dir_):
        info = CM.vload(os.path.join(dir_,'info'))
        dt_files = [os.path.join(dir_,f) for f in os.listdir(dir_)
                    if f.endswith('.tvn')]
        dts = [CM.vload(dt) for dt in dt_files]
        pp_cores_d = CM.vload(os.path.join(dir_,'postprocess'))
        
        return info, dts, pp_cores_d
    
    info,dts,pp_cores_d = load_dir(dir_)
    seed,dom = info

    #print info
    logger.info('seed: {}'.format(seed))
    logger.debug(dom)

    #print iterations
    for dt in sorted(dts,key=lambda dt: dt.citer):
        show_dtrace(dt)

    #print postprocess results
    show_analyzed_cores_d(pp_cores_d,dom)

    #print summary
    ntime = sum(dt.etime for dt in dts)
    logger.info(str_of_summary(seed,len(dts),ntime,dt.ctime,
                               dt.nconfigs,dt.ncovs,dir_))
    return len(dts),ntime,dt.ctime,dt.nconfigs,dt.ncovs,dt.cores_d,pp_cores_d


#Z3 STUFF
def z3db_of_dom(dom):
    if CM.__vdebug__:
        assert isinstance(dom,Dom), dom
        
    z3db = dict()
    for k,vs in dom.iteritems():
        vs = sorted(list(vs))
        ttyp,tvals=z3.EnumSort(k,vs)
        rs = [vv for vv in zip(vs,tvals)]
        rs.append(('typ',ttyp))
        z3db[k]=(z3.Const(k,ttyp),dict(rs))
    return z3db

def z3expr_of_core(core,z3db,myf):
    if CM.__vdebug__:
        assert isinstance(core,Core) and core, core

    f = []
    for vn,vs in core.iteritems():
        vn_,vs_ = z3db[vn]
        f.append(z3util.myOr([vn_ == vs_[v] for v in vs]))

    return myf(f)

def z3expr_of_config(config,z3db):
    if CM.__vdebug__:
        assert isinstance(config,Config),config
        assert len(config) == len(z3db), (len(config), len(z3db))

    f = []
    for vn,vv in config.iteritems():
        vn_,vs_ = z3db[vn]
        f.append(vn_==vs_[vv])

    return z3util.myAnd(f)

def z3_get_sat_core(core_expr,configs_expr,z3db):
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
        assert isinstance(z3db,dict),z3db

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

def config_of_model(model,dom):
    if CM.__vdebug__:
        assert isinstance(model,dict),model
        assert isinstance(dom,Dom),dom
        
    _f = lambda k: (model[k] if k in model
                    else random.choice(list(dom[k])))
    config = Config((k,_f(k)) for k in dom)
    return config


#Tests
#Otter
def prepare_otter(prog):

    #dir_ = getpath('~/Src/Devel/iTree_stuff/expData/{}'.format(prog))
    dir_ = getpath('~/Dropbox/git/config/otter_exps/{}'.format(prog))
    dom_file = os.path.join(dir_,'possibleValues.txt')
    pathconds_d_file = os.path.join(dir_,'{}.tvn'.format('pathconds_d'))
    assert os.path.isfile(dom_file),dom_file
    assert os.path.isfile(pathconds_d_file),pathconds_d_file
    
    dom,_ = Dom.get_dom(dom_file)
    logger.info("dom_file '{}': {}".format(dom_file,dom))
    
    st = time()
    pathconds_d = CM.vload(pathconds_d_file)
    logger.info("'{}': {} path conds ({}s)"
                .format(pathconds_d_file,len(pathconds_d),time()-st))

    args={'pathconds_d':pathconds_d}
    get_cov = lambda config: get_cov_otter(config,args)
    return dom,get_cov,pathconds_d

def get_cov_otter(config,args):
    if CM.__vdebug__:
        assert isinstance(config,Config),config
        assert isinstance(args,dict) and 'pathconds_d' in args, args
    sids = set()        
    for cov,configs in args['pathconds_d'].itervalues():
        if any(config.hcontent.issuperset(c) for c in configs):
            for sid in cov:
                sids.add(sid)
    return sids

def do_gt(dom,pathconds_d,n=None):
    """
    Obtain interactions using Otter's pathconds
    """
    if CM.__vdebug__:
        assert n is None or 0 <= n <= len(pathconds_d), n
        logger.warn("DEBUG MODE ON. Can be slow !")

    if n:
        print 'selectin {} rand'.format(n)
        rs = random.sample(pathconds_d.values(),n)
    else:
        rs = pathconds_d.itervalues()

    cconfigs_d = {}
    for covs,configs in rs:
        for c in configs:
            c = Config(c)
            if c not in cconfigs_d:
                cconfigs_d[c] = set(covs)
            else:
                covs_ = cconfigs_d[c]
                for sid in covs:
                    covs_.add(sid)
            
    logger.info("infer interactions using {} configs"
                .format(len(cconfigs_d)))
    st = time()
    cores_d,configs_d,covs_d = Cores_d(),{},{}
    _ = infer_covs(cores_d,cconfigs_d,configs_d,covs_d,dom)
    pp_cores_d = postprocess(cores_d,covs_d,dom)
    show_analyzed_cores_d(pp_cores_d,dom)
    
    logger.info(str_of_summary(0,0,time()-st,0,
                               len(configs_d),len(pp_cores_d),
                               ''))
    
    return pp_cores_d,cores_d,configs_d,covs_d,dom

# Real executions
def void_run_single(cmd):
    logger.detail(cmd)
    try:
        rs_outp,rs_err = CM.vcmd(cmd)
        if rs_outp:
            logger.detail("outp: {}".format(rs_outp))
        
        #IMPORTANT, command out the below allows
        #erroneous test runs, which can be helpful
        #to detect incorrect configs
        #assert len(rs_err) == 0, rs_err

        serious_errors = ["invalid",
                          "-c: line",
                          "assuming not executed"]
        if rs_err:
            logger.detail("error: {}".format(rs_err))
            if any(serr in rs_err for serr in serious_errors): 
                raise AssertionError("Check this error !")
            
    except Exception as e:
        raise AssertionError("cmd '{}' fails, raise error: {}".format(cmd,e))
    
def void_run(cmds):
    "just exec command, does not return anything"
    assert cmds, cmds
    
    if not CM.is_iterable(cmds): cmds = [cmds]
    logger.detail('run {} cmds'.format(len(cmds)))
    for cmd in cmds:
        void_run_single(cmd)
    
    
from gcovparse import gcovparse
def parse_gcov(gcov_file):
    if CM.__vdebug__:
        assert os.path.isfile(gcov_file)

    gcov_obj = gcovparse(CM.vread(gcov_file))
    assert len(gcov_obj) == 1, gcov_obj
    gcov_obj = gcov_obj[0]
    sids = (d['line'] for d in gcov_obj['lines'] if d['hit'] > 0)
    sids = set("{}:{}".format(gcov_obj['file'],line) for line in sids)
    return sids

def get_cov_wrapper(config,args):
    cur_dir = os.getcwd()
    try:
        dir_ = args["dir_"]
        os.chdir(dir_)
        cov = args['get_cov'](config,args)
        os.chdir(cur_dir)
        return cov
    except:
        os.chdir(cur_dir)
        raise

        
#Motivation/Simple examples
def prepare_motiv(dom_file,prog_name):
    if CM.__vdebug__:
        assert isinstance(dom_file,str),dom_file
        assert isinstance(prog_name,str),prog_name
    import platform
    
    dir_ = getpath('~/Dropbox/git/config/benchmarks/examples')
    dom_file = getpath(os.path.join(dir_,"{}.dom".format(dom_file)))
    prog_exe = getpath(os.path.join(dir_,"{}.{}.exe"
                                    .format(prog_name,platform.system())))
    assert os.path.isfile(dom_file),dom_file
    assert os.path.isfile(prog_exe),prog_exe
    
    dom,_ = Dom.get_dom(dom_file)
    logger.info("dom_file '{}': {}"
                .format(dom_file,dom))
    logger.info("prog_exe: '{}'".format(prog_exe))

    args = {'var_names':dom.keys(),
            'prog_name': prog_name,
            'prog_exe': prog_exe,
            'get_cov': get_cov_motiv,  #_gcov
            'dir_': dir_}
    get_cov = lambda config: get_cov_wrapper(config,args)
    return dom,get_cov

def get_cov_motiv(config,args):
    """
    Traces read from stdin
    """
    if CM.__vdebug__:
        assert isinstance(config,Config),config
        
    tmpdir = '/var/tmp/'
    prog_exe = args['prog_exe']
    var_names = args['var_names']
    opts = ' '.join(config[vname] for vname in var_names)
    traces = os.path.join(tmpdir,'t.out')
    cmd = "{} {} > {}".format(prog_exe,opts,traces)
    void_run(cmd)
    sids = set(CM.iread_strip(traces))
    return sids

def get_cov_motiv_gcov(config,args):
    """
    Traces ared from gcov info
    """
    prog_name = args['prog_name']
    prog_exe = args['prog_exe']
    var_names = args['var_names']    
    opts = ' '.join(config[vname] for vname in var_names)
    
    #cleanup
    cmd = "rm -rf *.gcov *.gcda"
    void_run(cmd)
    #CM.pause()
    
    #run testsuite
    cmd = "{} {}".format(prog_exe,opts)
    void_run(cmd)
    #CM.pause()

    #read traces from gcov
    #/path/prog.Linux.exe -> prog
    cmd = "gcov {}".format(prog_name)
    void_run(cmd)
    gcov_dir = os.getcwd()
    sids = (parse_gcov(os.path.join(gcov_dir,f))
            for f in os.listdir(gcov_dir) if f.endswith(".gcov"))
    sids = set(CM.iflatten(sids))
    return sids


def doctestme():
    import doctest
    doctest.testmod()


"""
gt
ngircd:
vsftpd: (0,1,336), (1,4,101), (2,6,170), (3,5,1385), (4,24,410), (5,5,102), (6,6,35), (7,2,10)


(0,1,56), (1,3,11), (2,3,15), (3,3,5)
"""    
