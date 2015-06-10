import tempfile
from time import time
import os.path
import itertools
import random

from config_miscs import HDict
import z3
import z3util
import vu_common as CM

from collections import namedtuple, OrderedDict

logger = CM.VLog('config')
logger.level = CM.VLog.DETAIL
CM.VLog.PRINT_TIME = True
CM.__vdebug__ = False
do_comb_conj_disj = True
print_cov = False

#Data Structures
is_cov = lambda cov: (isinstance(cov,set) and
                      all(isinstance(s,str) for s in cov))
def str_of_cov(cov):
    if CM.__vdebug__:
        assert is_cov(cov),cov

    s = "({})".format(len(cov))
    if print_cov:
        s = "{} {}".format(s,','.join(sorted(cov)))
    return s
    
is_valset = lambda vs: (isinstance(vs,frozenset) and vs and
                        all(isinstance(v,str) for v in vs))

def str_of_valset(s): return ','.join(sorted(s))
    
is_setting = lambda (k,v): isinstance(k,str) and isinstance(k,str)
def str_of_setting((k,v)):
    if CM.__vdebug__:
        assert is_setting((k,v)), (k,v)
    return '{}={}'.format(k,v)

is_csetting = lambda (k,vs): isinstance(k,str) and is_valset(vs)

def str_of_csetting((k,vs)):
    if CM.__vdebug__:
        assert is_setting((k,vs)), (k,vs)
    
    return '{}={}'.format(k,str_of_valset(vs))

is_dom = lambda dom: (isinstance(dom,HDict) and dom and
                      all(is_csetting(s) for s in dom.iteritems()))

def str_of_dom(dom,print_simple=False):
    if CM.__vdebug__:
        assert is_dom(dom),dom
        
    if print_simple:
        return("{} vars and {} poss configs"
               .format(len(dom),siz_of_dom(dom)))
    else:
        return '\n'.join("{}: {}".format(k,str_of_valset(dom[k]))
                         for k in dom)

is_config = lambda c: (isinstance(c,HDict) and c and
                       all(is_setting(s) for s in c.iteritems()))

def str_of_config(config,cov=None):
    if CM.__vdebug__:
        assert is_config(config), config
        assert cov is None or is_cov(cov), cov
        
    s =  ' '.join(map(str_of_setting,config.iteritems()))
    if cov:
        s = "{}: {}".format(s,str_of_cov(cov))
    return s

def str_of_configs(configs,covs=None):
    if CM.__vdebug__:
        assert all(is_config(c) for c in configs), configs
        assert covs is None or len(covs) == len(configs), \
            (len(covs),len(configs))
    if covs:
        ss = (str_of_config(c,cov) for c,cov in zip(configs,covs))
    else:
        ss = (str_of_config(c,None) for c in configs)
    return '\n'.join("{}. {}".format(i+1,s) for i,s in enumerate(ss))

is_core = lambda core: (isinstance(core,HDict) and
                        all(is_csetting(s) for s in core.iteritems()))

maybe_core = lambda c: c is None or is_core(c)

def str_of_core(core,delim=' '):
    """
    >>> print str_of_core(HDict([('a',frozenset(['2'])),('c',frozenset(['0','1']))]))
    a=2 c=0,1
    >>> print str_of_core(HDict())
    true
    """
    if CM.__vdebug__:
        assert is_core(core), core
    if core:
        return delim.join(map(str_of_csetting,core.iteritems()))
    else:
        return 'true'

def is_pncore((pc,pd,nc,nd)):
    return all(maybe_core(c) for c in (pc,pd,nc,nd))
        
def sstr_of_pncore(pncore):
    """
    >>> pc = HDict([('a',frozenset(['2'])),('c',frozenset(['0','1']))])
    >>> pd = None
    >>> nc = HDict([('b',frozenset(['2']))])
    >>> nd = None
    >>> print sstr_of_pncore((pc,pd,nc,nd))
    pc: a=2 c=0,1; nc: b=2
    """
    if CM.__vdebug__:
        assert is_pncore(pncore),pncore
        
    ss = ("{}: {}".format(s,str_of_core(c)) for s,c in
          zip('pc pd nc nd'.split(),pncore) if c is not None)
    return '; '.join(ss)

def str_of_pncore(pncore,fstr_f=None):
    """
    Important: only call fstr_of_pncore *after* being analyzed
    
    """
    return fstr_f(pncore) if fstr_f else sstr_of_pncore(pncore)

is_cores_d = lambda cores_d: (isinstance(cores_d,dict) and
                              all(isinstance(sid,str) and is_pncore(c)
                                  for sid,c in cores_d.iteritems()))
def str_of_cores_d(cores_d,fstr_f=None):
    if CM.__vdebug__:
        assert is_cores_d(cores_d),cores_d
        assert dom is None or is_dom(dom),dom
        
    return '\n'.join("{}. {}: {}"
                     .format(i+1,sid,str_of_pncore(cores_d[sid],fstr_f))
                     for i,sid in enumerate(sorted(cores_d)))

is_mcores_d = lambda mcores_d: (isinstance(mcores_d,dict) and
                                all(is_pncore(c) and is_cov(cov)
                                    for c,cov in mcores_d.iteritems()))

def str_of_mcores_d(mcores_d,fstr_f=None):
    if CM.__vdebug__:
        assert is_mcores_d(mcores_d),mcores_d
        assert dom is None or is_dom(dom),dom

    mc = sorted(mcores_d.iteritems(),
                key=lambda (core,cov): (
                    stren_of_core(core),
                    vstren_of_core(core),
                    len(cov)))

    ss = ("{}. ({}) {}: {}"
          .format(i+1,
                  stren_of_core(core),str_of_pncore(core,fstr_f),
                  str_of_cov(cov))
          for i,(core,cov) in enumerate(mc))

    return '\n'.join(ss)

def strens_of_mcores_d(mcores_d):
    """
    (strength,cores,sids)
    """
    if CM.__vdebug__:
        assert is_mcores_d(mcores_d),mcores_d
    
    strens = set(stren_of_core(core) for core in mcores_d)

    rs = []
    for stren in sorted(strens):
        cores = [c for c in mcores_d if stren_of_core(c) == stren]
        cov = set(sid for core in cores for sid in mcores_d[core])
        rs.append((stren,len(cores),len(cov)))
    return rs


def strens_str_of_mcores_d(mcores_d):
    if CM.__vdebug__:
        assert is_mcores_d(mcores_d),mcores_d
    
    return ', '.join("({},{},{})".format(siz,ncores,ncov)
                     for siz,ncores,ncov in strens_of_mcores_d(mcores_d))

# Functions on Data Structures
def config_c_implies(config,core):
    """
    x=0&y=1 => x=0,y=1
    not(x=0&z=1 => x=0,y=1)
    """
    if CM.__vdebug__:
        assert is_config(config),config
        assert is_core(core),core

    return (not core or
            all(k in config and config[k] in core[k] for k in core))

def config_d_implies(config,core): 
    if CM.__vdebug__:
        assert is_config(config),config
        assert is_core(core),core

    return (not core or
            any(k in config and config[k] in core[k] for k in core))
   
def neg_of_core(core,dom):
    if CM.__vdebug__:
        assert is_core(core),core
        assert is_dom(dom),dom

    return HDict((k,dom[k]-core[k]) for k in core)

pncore_mk_default = lambda: (None,None,None,None)

def settings_of_core(core):
    if CM.__vdebug__:
        assert (isinstance(core,tuple) and
                (len(core)==2 or len(core)==4) and 
                all(c is None or is_core(c) for c in core)),core
    core = (c for c in core if c)
    return set(s for c in core for s in c.iteritems())
    
def values_of_core(core):
    if CM.__vdebug__:
        assert all(maybe_core(c) for c in core), core
        
    core = (c for c in core if c)
    return set(s for c in core for s in c.itervalues())

stren_of_core = lambda core: len(settings_of_core(core))
vstren_of_core = lambda core: sum(map(len,values_of_core(core)))

def merge_cores_d(cores_d):
    if CM.__vdebug__:
        assert is_cores_d(cores_d),cores_d

    mcores_d = {}
    for sid,core in cores_d.iteritems():
        if core in mcores_d:
            mcores_d[core].add(sid)
        else:
            mcores_d[core] = set([sid])

    return mcores_d

#Inference algorithm
DTrace = namedtuple("DTrace",
                    "citer etime ctime "
                    "configs covs "
                    "new_covs new_cores "
                    "sel_core cores_d")

def infer(configs,core,dom):
    """
    Approximation in *conjunctive* form
    """
    if CM.__vdebug__:
        assert all(is_config(c) for c in configs) and configs, configs
        assert core is None or is_core(core),core        
        assert is_dom(dom),dom

    if core is None:  #not yet set
        core = min(configs,key=lambda c:len(c))
        core = HDict((k,frozenset([v])) for k,v in core.iteritems())
        
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
    core = HDict((k,frozenset(vs)) for k,vs in zip(core,vss) if vs)
    return core  

def infer_cache(core,configs,dom,cache):
    if CM.__vdebug__:
        assert core is None or is_core(core),core
        assert all(is_config(c) for c in configs) and configs, configs
        assert is_dom(dom),dom
        assert isinstance(cache,dict),cache

    configs = frozenset(configs)
    key = (core,configs)
    if key not in cache:
        cache[key] = infer(configs,core,dom)
    return cache[key]

def infer_sid(sid,core,configs,configs_d,covs_d,dom,cache):
    if CM.__vdebug__:
        assert isinstance(sid,str),sid
        assert is_pncore(core), core
        assert is_dom(dom),dom
        assert is_configs_d(configs_d),configs_d
        assert is_covs_d(covs_d),covs_d
        assert isinstance(cache,dict),cache

    def _f(configs,cc,cd,_b):
        new_cc,new_cd = cc,cd
        if configs:
            new_cc = infer_cache(cc,configs,dom,cache)
        if do_comb_conj_disj and new_cc:
            configs_ = [c for c in _b() if config_c_implies(c,new_cc)]
            if configs_:
                new_cd = infer_cache(cd,configs_,dom,cache)
                if new_cd:
                    new_cd = HDict((k,v) for (k,v) in new_cd.iteritems()
                                   if k not in new_cc)
            
        return new_cc,new_cd

    pconfigs,nconfigs = [],[]
    for config in configs:
        if sid in configs_d[config]:
            pconfigs.append(config)
        else:
            nconfigs.append(config)
            
    # print 'pos'
    # print str_of_configs(pconfigs)
    # print 'neg'
    # print str_of_configs(pconfigs)    
    pc,pd,nc,nd = core
    _b = lambda: [c for c in configs_d if sid not in configs_d[c]]
    pc_,pd_ = _f(pconfigs,pc,pd,_b)
    _b = lambda: covs_d[sid]
    nc_,nd_ = _f(nconfigs,nc,nd,_b)    
    return pc_,pd_,nc_,nd_

is_covs_d = lambda covs_d: (isinstance(covs_d,dict) and
                            all(isinstance(sid,str) and isinstance(configs,set)
                                and all(is_config(c) for c in configs)
                                for sid,configs in covs_d.iteritems()))
is_configs_d = lambda configs_d: (isinstance(configs_d,dict) and
                                  all(is_config(config) and isinstance(cov,set)
                                      and all(isinstance(c,str) for c in cov)
                                      for config,cov in configs_d.iteritems()))


def infer_covs(cores_d,configs,covs,configs_d,covs_d,dom):
    if CM.__vdebug__:
        assert is_cores_d(cores_d),cores_d
        assert all(is_config(c) for c in configs) and configs, configs
        assert all(is_cov(c) for c in covs), covs
        assert len(configs) == len(covs), (len(configs),len(covs))
        assert all(c not in configs_d for c in configs), configs
        assert is_configs_d(configs_d),configs_d
        assert is_covs_d(covs_d),covs_d
        assert is_dom(dom),dom
        
    new_covs,new_cores = set(),set()  #updated stuff

    sids = set(cores_d.keys())
    assert all(c not in configs_d for c in configs), configs    
    #update configs_d and covs_d
    for config,cov in zip(configs,covs):
        for sid in cov:
            sids.add(sid)
            if sid not in covs_d:
                covs_d[sid] = set()
            covs_d[sid].add(config)

        assert config not in configs_d, str_of_config(config)
        configs_d[config] = cov

    # logger.detail("infer invs for {} cov using {} configs"
    #               .format(len(sids),len(configs)))
    
    cache = {}
    for i,sid in enumerate(sorted(sids)):
        if sid in cores_d:
            core = cores_d[sid]
        else:
            core = pncore_mk_default()
            new_covs.add(sid)
        
        core_ = infer_sid(sid,core,configs,configs_d,covs_d,dom,cache)
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
        assert is_core(core) and core, core
        assert (sat_core is None or
                is_core(sat_core) and
                all(k not in core for k in sat_core)),sat_core
        assert is_dom(dom),dom

    _new = lambda : HDict((k,core[k]) for k in core)
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
        sat_d = z3_get_sat_core(changed_core,existing_configs_expr,z3db)
        if not sat_d:
            continue

        settings = []
        for k in dom:
            if k in sat_d:
                v = str(sat_d[k])
            else:
                v = random.choice(list(dom[k]))
            settings.append((k,v))

        config = HDict(settings)
        configs.append(config)
        if CM.__vdebug__:
            assert config_c_implies(config,changed_core)
                
    return configs

def select_core(pncores,ignore_cores):
    if CM.__vdebug__:
        assert all(is_pncore(c) for c in pncores) and pncores,pncores
        assert (isinstance(ignore_cores,set) and
                all(is_pncore(c) for c in ignore_cores)),ignore_cores

    cores = [(core,stren_of_core(core)) for core in pncores
             if core not in ignore_cores]
    cores = [(core,stren) for core,stren in cores if stren]
    if cores:
        core = max(cores,key=lambda (c,stren):stren)[0]
    else:
        core = None
    ignore_cores.add(core)
    return core

def eval_configs(configs,get_cov):
    if CM.__vdebug__:
        assert (isinstance(configs,list) and
                all(is_config(c) for c in configs)
                and configs), configs
        assert callable(get_cov),get_cov
    st = time()
    covs = [get_cov(c) for c in configs]
    return covs,time() - st

def gen_configs_init(cover_siz,seed,dom):
    #initial configs
    if cover_siz:
        cstren,max_confs = cover_siz
        if cstren == 0:
            if max_confs < 0:
                configs = gen_configs_full(dom)
                logger.info("gen all {} configs".format(len(configs)))
            else:
                configs = gen_configs_rand(max_confs,dom)
                logger.info("gen {} rand configs".format(len(configs)))
        else:
            configs = gen_configs_tcover(cstren,seed,tmpdir)
            configs_rand_n = max_confs - len(configs)

            if configs_rand_n:
                configs_ = gen_configs_rand(configs_rand_n,dom)
                configs.extend(configs_)

            configs = list(set(configs))
            logger.info("gen {} {}-cover configs"
                        .format(len(configs),cstren))
    else:
        configs = gen_configs_tcover1(dom)

    assert configs, "empty set of configs"
    return configs

def gen_configs_iter(cores,ignore_cores,configs_d,z3db,dom):
    if CM.__vdebug__:
        assert (isinstance(cores,set) and 
                all(is_pncore(c) for c in cores)), cores
        assert (isinstance(ignore_cores,set) and 
                all(is_pncore(c) for c in ignore_cores)), ignore_cores
        assert is_configs_d(configs_d),configs_d
        assert is_dom(dom),dom

    configs = []
    while True:
        sel_core = select_core(cores,ignore_cores)
        if sel_core is None:
            break
        pc,pd,nc,nd = sel_core
        if pc:
            configs.extend(gen_configs_core_z3(pc,None,configs_d,z3db,dom))
        if pd:
            configs.extend(gen_configs_core_z3(pd,pc,configs_d,z3db,dom))
        if nc:
            configs.extend(gen_configs_core_z3(nc,None,configs_d,z3db,dom))
        if nd:
            configs.extend(gen_configs_core_z3(nd,nc,configs_d,z3db,dom))
            
        configs = list(set(configs))        
        if configs:
            break
        else:
            logger.debug("no sample created for sel_core {}"
                         .format(str_of_pncore(sel_core)))
        
    #self_core -> configs
    if CM.__vdebug__:
        assert not sel_core or configs, (sel_core,configs)
        assert all(c not in configs_d for c in configs), configs
        
    return sel_core, configs

    
def intgen(dom,get_cov,seed=None,tmpdir=None,cover_siz=None,
           config_default=None,prefix='vu',do_postprocess=False):
    """
    cover_siz=(0,n):  generates n random configs
    cover_siz=(0,-1):  generates full random configs
    """
    if CM.__vdebug__:
        assert is_dom(dom),dom
        assert callable(get_cov),get_cov
        assert (config_default is None or
                is_config(config_default)), config_default

    seed = round(time(),2) if seed is None else float(seed)
    random.seed(seed)

    if tmpdir is None:
        tmpdir = tempfile.mkdtemp(dir='/var/tmp',prefix=prefix)
        
    logger.info("seed: {}, tmpdir: {}".format(seed,tmpdir))
    
    #save info
    ifile = os.path.join(tmpdir,'info')
    iobj = (seed,dom)
    CM.vsave(ifile,iobj)
    
    #some settings
    cores_d = {} 
    cur_iter = 1
    max_stuck = 5
    cur_stuck = 0
    ignore_cores = set()
    configs_d = {}
    covs_d = {}
    sel_core = pncore_mk_default()
  
    #begin
    st = time()
    ct = st
    cov_time = 0.0
    
    configs = gen_configs_init(cover_siz,seed,dom)
    if config_default: configs.append(config_default)
    covs,ctime = eval_configs(configs,get_cov)
    cov_time += ctime
    new_covs,new_cores = infer_covs(cores_d,configs,covs,
                                    configs_d,covs_d,dom)

    z3db = z3db_of_dom(dom)
    while cur_stuck <= max_stuck:
        #save info
        ct_ = time();etime = ct_ - ct;ct = ct_
        dtrace = DTrace(cur_iter,etime,cov_time,
                        configs,covs,
                        new_covs,new_cores,
                        sel_core,
                        cores_d)
        dfile = os.path.join(tmpdir,"{}.tvn".format(cur_iter))
        CM.vsave(dfile,dtrace)
        print_dtrace(dtrace)
        
        if cover_siz:
            break
        
        cur_iter += 1

        # print str_of_cores_d(cores_d)
        # print '\n'.join(map(str_of_pncore,set(cores_d.values())))
        #gen new configs
        sel_core,configs = gen_configs_iter(set(cores_d.values()),
                                            ignore_cores,configs_d,z3db,dom)
        if sel_core is None:
            logger.info('select no core for refinement, '
                        'done at iter {}'.format(cur_iter))
            break

        assert configs,configs
        covs,ctime = eval_configs(configs,get_cov)
        cov_time += ctime
        new_covs,new_cores = infer_covs(cores_d,configs,covs,
                                        configs_d,covs_d,dom)

        if new_covs or new_cores: #progress
            cur_stuck = 0
        else: #no progress
            cur_stuck += 1

    logger.info(str_of_summary(seed,cur_iter,time()-st,cov_time,
                               len(configs_d),len(cores_d),
                               tmpdir))

    pp_cores_d = None
    if do_postprocess:
        logger.info("*** postprocess ***")
        pp_cores_d = postprocess(cores_d,covs_d,dom)

    return pp_cores_d,cores_d,configs_d,covs_d,dom

#Shortcuts
def intgen_full(dom,get_cov,tmpdir=None,prefix='vu',do_postprocess=True):
    return intgen(dom,get_cov,seed=None,tmpdir=tmpdir,
                 cover_siz=(0,-1),config_default=None,
                  prefix=prefix,do_postprocess=do_postprocess)

def intgen_rand(dom,get_cov,rand_n,seed=None,tmpdir=None,
                prefix='vu',do_postprocess=True):
    return intgen(dom,get_cov,seed=seed,tmpdir=tmpdir,
                 cover_siz=(0,rand_n),config_default=None,
                  prefix=prefix,do_postprocess=do_postprocess)

#postprocess
def postprocess_rs(rs): return postprocess(rs[1],rs[3],rs[4])

def postprocess(cores_d,covs_d,dom):
    if CM.__vdebug__:
        assert is_covs_d(covs_d),covs_d
        assert is_dom(dom),dom

    cores_d = analyze(cores_d,covs_d,dom)
    mcores_d = merge_cores_d(cores_d)

    fstr_f=lambda c: fstr_of_pncore(c,dom)
    logger.debug("mcores_d has {} items\n".format(len(mcores_d)) +
                 str_of_mcores_d(mcores_d,fstr_f)) 
    logger.info(strens_str_of_mcores_d(mcores_d))

    return cores_d

def analyze(cores_d,covs_d,dom):
    if CM.__vdebug__:
        assert is_cores_d(cores_d),cores_d
        assert is_covs_d(covs_d),covs_d
        assert is_dom(dom),dom
        
    logger.info("analyze interactions for {} sids".format(len(cores_d)))
    logger.debug("verify ...")
    verify_cache = {}
    rs_verify = [(sid,verify_pncore_cache(core,covs_d[sid],dom,verify_cache))
          for sid,core in cores_d.iteritems()]
    
    logger.debug("simplify ...")
    simplify_cache = {}
    rs_simplify = []
    for sid,core in rs_verify:
        pncore = simplify_pncore_cache(core,dom,simplify_cache)
        rs_simplify.append((sid,pncore))
        
    return dict(rs_simplify)

def reformat((pc,pd,nc,nd),dom):
    if pd: pd = neg_of_core(pd,dom)
    if nc: nc = neg_of_core(nc,dom)
    return (pc,pd,nc,nd)

def verify_pncore_cache(pncore,configs,dom,cache):
    if CM.__vdebug__:
        assert is_pncore(pncore),pncore
        assert all(is_config(c) for c in configs) and configs,configs
        assert is_dom(dom),dom
        assert isinstance(cache,dict),cache
    configs = frozenset(configs)
    key = (pncore,configs)
    if key not in cache:
        cache[key] = verify_pncore(pncore,configs,dom)
    return cache[key]
    
def verify_pncore((pc,pd,nc,nd),configs,dom):
    if CM.__vdebug__:
        assert is_pncore((pc,pd,nc,nd)), (pc,pd,nc,nd)
        assert pc is not None, pc #this never could happen
        #nc is None => pd is None
        assert (nc is not None or pd is None), (nc,nd)
        assert all(is_config(c) for c in configs) and configs, configs
        assert is_dom(dom),dom

    old_pncore = (pc,pd,nc,nd)        
    #traces => pc & neg(pd)
    if pc:
        assert all(config_c_implies(c,pc) for c in configs), pc
        
    if pd:
        pd_n = neg_of_core(pd,dom)
        if not all(config_d_implies(c,pd_n) for c in configs):
            logger.debug('pd {} invalid'.format(str_of_core(pd)))
            pd = None

    #neg traces => nc & neg(nd)
    #pos traces => neg(nc & neg(nd))
    #post traces => nd | neg(nc) 
    if nc is not None and nd is None:
        nc_n = neg_of_core(nc,dom)
        if not all(config_d_implies(c,nc_n) for c in configs):
            logger.debug('nc {} invalid'.format(str_of_core(nc)))
            nc = None
    elif nc is None and nd is not None:
        if not all(config_c_implies(c,nd) for c in configs):
            logger.debug('nd {} invalid'.format(str_of_core(nd)))
            nd = None
    elif nc is not None and nd is not None:
        nc_n = neg_of_core(nc,dom)
        if not all(config_c_implies(c,nd) or config_d_implies(c,nc_n)
                   for c in configs):
            logger.debug('nc & nd invalid')
            nc = None
            nd = None

    pncore = (pc,pd,nc,nd)
    if old_pncore != pncore:
        logger.debug("{} -> {}".
                     format(str_of_pncore(old_pncore),
                            str_of_pncore(pncore)))
        
    return pncore

def simplify_pncore_cache(pncore,dom,cache):
    if CM.__vdebug__:
        assert is_pncore(pncore),pncore
        assert is_dom(dom),dom
        assert isinstance(cache,dict),cache

    if pncore not in cache:
        cache[pncore]=simplify_pncore(pncore,dom)
    return cache[pncore]
        
def simplify_pncore((pc,pd,nc,nd),dom):
    """
    Compare between (pc,pd) and (nc,nd) and return the stronger one.
    This will set either (pc,pd) or (nc,nd) to (None,None)
    
    Assumption: all 4 cores are verified
    """
    if CM.__vdebug__:
        assert is_pncore((pc,pd,nc,nd)), (pc,pd,nc,nd)
        assert pc is not None, pc #this never could happen
        #nc is None => pd is None
        assert (nc is not None or pd is None), (nc,nd)
        assert is_dom(dom),dom
        
    #pf = pc & neg(pd)
    #nf = neg(nc & neg(nd)) = nd | neg(nc)
    old_pncore = (pc,pd,nc,nd)
    
    #remove empty ones
    if not pc: pc = None
    if not pd: pd = None
    if not nc: nc = None
    if not nd: nd = None

    # print pc,pd
    # print nc,nd
    if (pc is None and pd is None) or (nc is None and nd is None):
        return (pc,pd,nc,nd)

    #convert to z3
    z3db = z3db_of_dom(dom)
    
    def _f(cc,cd):
        fs = []
        if cc:
            f = z3expr_of_core(cc,z3db,myf=z3util.myAnd)
            fs.append(f)
        if cd:
            cd_n = neg_of_core(cd,dom)
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
                    .format(str_of_pncore((pc,pd,nc,nd)),
                    pf,nf))
        
    pncore = (pc,pd,nc,nd)
    if old_pncore != pncore:
        logger.debug("{} -> {}".
                     format(str_of_pncore(old_pncore),
                            str_of_pncore(pncore)))        
    return pncore

def is_simplified((pc,pd,nc,nd)):
    return (pc is None and pd is None) or (nc is None and nd is None)

def fstr_of_pncore((pc,pd,nc,nd),dom):
    """
    Assumption: all 4 cores are verified and simplified
    """

    if CM.__vdebug__:
        assert pc is None or is_core(pc) and pc, pc
        assert pd is None or is_core(pd) and pd, pd
        assert nc is None or is_core(nc) and nc, nc
        assert nd is None or is_core(nd) and nd, nd
        assert is_dom(dom),dom
        
    def _f(core,delim):
        s = str_of_core(core,delim)
        if len(core) > 1:
            s = '({})'.format(s)
        return s

    def _cd(ccore,dcore,delim):
        ss = []
        if ccore:
            ss.append(_f(ccore,' & '))
        if dcore:
            assert is_dom(dom),dom
            dcore_n = neg_of_core(dcore,dom)
            ss.append(_f(dcore_n, ' | '))
        return delim.join(ss) if ss else 'true'

    if is_simplified((pc,pd,nc,nd)):
        if (nc is None and nd is None):
            #pc & not(pd)
            ss = _cd(pc,pd,' & ')
        else:
            #not(nc & not(nd))  =  not(nc) | nd
            ss = _cd(nd,nc,' | ')
    else:
        p_ss = _cd(pc,pd,' & ')
        n_ss = _cd(nd,nc,' | ')
        ss = ','.join([p_ss,n_ss]) + '***'
        
    return ss

#Miscs
def debug_find_configs(sid,configs_d,find_in):
    if find_in:
        configs = [(c,cov) for c,cov in configs_d.iteritems() if sid in cov]
    else:
        configs = [(c,cov) for c,cov in configs_d.iteritems() if sid not in cov]

    if configs:
        configs,covs = zip(*configs)
        print str_of_configs(configs,covs)
    
        
def get_dom(dom_file):
    if CM.__vdebug__:
        assert os.path.isfile(dom_file), domfile
        
    def get_lines(lines):
        rs = (line.split() for line in lines)
        rs = ((parts[0],frozenset(parts[1:])) for parts in rs)
        return rs

    dom = HDict(get_lines(CM.iread_strip(dom_file)))

    config_default = None
    dom_file_default = dom_file+'.default'
    if os.path.isfile(dom_file_default):
        rs = get_dom_lines(CM.iread_strip(dom_file_default))
        config_default = HDict((k,list(rs[k])[0]) for k in rs)

    return dom,config_default

#generate configs
siz_of_dom = lambda d: CM.vmul(len(vs) for vs in d.itervalues())
def gen_configs_full(dom):
    if CM.__vdebug__:
        assert is_dom(dom),dom
    
    ns,vs = itertools.izip(*dom.iteritems())
    configs = [HDict(zip(ns,c)) for c in itertools.product(*vs)]
    return configs

def gen_configs_rand(n,dom):
    if CM.__vdebug__:
        assert n > 0,n
        assert is_dom(dom),dom        

    if n >= siz_of_dom(dom):
        return gen_configs_full(dom)

    rgen = lambda: [(k,random.choice(list(dom[k]))) for k in dom]
    configs = list(set(HDict(rgen()) for _ in range(n)))
    return configs

def gen_configs_tcover1(dom):
    """
    Return a set of tcover array of stren 1
    """
    if CM.__vdebug__:
        assert is_dom(dom), dom
        
    dom_used = dict((k,set(dom[k])) for k in dom)

    def mk():
        config = []
        for k in dom:
            if k in dom_used:
                v = random.choice(list(dom_used[k]))
                dom_used[k].remove(v)
                if not dom_used[k]:
                    dom_used.pop(k)
            else:
                v = random.choice(list(dom[k]))

            config.append((k,v))
        return HDict(config)

    configs = []
    while dom_used:
        configs.append(mk())

    return configs

def print_dtrace(dt,print_mcores_d=True):
    if CM.__vdebug__:
        assert isinstance(dt,DTrace)
        
    logger.info("ITER {}, ".format(dt.citer) +
                "{0:.2f}s, ".format(dt.etime) +
                "{0:.2f}s eval, ".format(dt.ctime) +
                "total: {} configs, {} covs, {} cores, ".format(0,0,0) +
                "new: {} configs, {} covs, {} cores, "
                .format(len(dt.configs),
                        len(dt.new_covs),
                        len(dt.new_cores)) +
                "{}".format("** progress **"
                            if dt.new_covs or dt.new_cores else ""))

    logger.debug('sel_core: ({}) {}'.format(
        stren_of_core(dt.sel_core), str_of_pncore(dt.sel_core)))
    logger.debug('create {} configs'.format(len(dt.configs)))
    logger.detail('\n' + str_of_configs(dt.configs,dt.covs))

    if print_mcores_d:
        mcores_d = merge_cores_d(dt.cores_d)
        logger.debug("mcores_d has {} items".format(len(mcores_d)))
        logger.detail('\n{}'.format(str_of_mcores_d(mcores_d)))
        logger.info(strens_str_of_mcores_d(mcores_d))


def str_of_summary(seed,iters,ntime,ctime,nconfigs,ncovs,tmpdir):
    s = "Summary: "
    s += "Seed {}, ".format(seed)
    s += "Iters {}, ".format(iters)
    s += "Time ({0:.2f}s, {0:.2f}s), ".format(ntime,ctime)
    s += "Configs {}, ".format(nconfigs)
    s += "Covs {}, ".format(ncovs)
    s += "Tmpdir {}".format(tmpdir)
    return s

def replay(dirname):
    def load_dir(dirname):
        iobj = CM.vload(os.path.join(dirname,'info'))
        dt_files = [os.path.join(dirname,f) for f in os.listdir(dirname)
                    if f.endswith('.tvn')]
        dts = [CM.vload(dt) for dt in dt_files]
        return iobj, dts
    
    iobj,dts = load_dir(dirname)
    seed,dom = iobj
    logger.info('seed: {}'.format(seed))
    logger.debug(str_of_dom(dom,print_simple=False))
    logger.debug(str_of_dom(dom,print_simple=True))    

    ntime = 0.0
    nsamples = 0
    ncovs = 0
    robjs = sorted(dts,key=lambda dt: dt.citer)
    for dt in dts:
        print_dtrace(dt)
        ntime += dt.etime
        nsamples += len(dt.configs)
        ncovs += len(dt.new_covs)
        
    niters = len(dts)
    logger.info(str_of_summary(seed,niters,
                               ntime,dt.ctime,
                               nsamples,ncovs,dirname))

    return niters,ntime,dt.ctime,nsamples,ncovs,dt.cores_d



#Tests
getpath = lambda f: os.path.realpath(os.path.expanduser(f))

#Otter
def prepare_otter(prog):
    dir_ = getpath('~/Src/Devel/iTree_stuff/expData/{}'.format(prog))
    dom_file = os.path.join(dir_,'possibleValues.txt')
    pathconds_d_file = os.path.join(dir_,'{}.tvn'.format('pathconds_d'))
    assert os.path.isfile(dom_file),dom_file
    assert os.path.isfile(pathconds_d_file),pathconds_d_file
    
    dom,_ = get_dom(dom_file)
    logger.info("dom_file '{}': {}"
                .format(dom_file,str_of_dom(dom,print_simple=True)))
    
    st = time()
    pathconds_d = CM.vload(pathconds_d_file)
    logger.info("'{}': {} path conds ({}s)"
                .format(pathconds_d_file,len(pathconds_d),time()-st))

    args={'pathconds_d':pathconds_d}
    get_cov = lambda config: get_cov_otter(config,args)
    return dom,get_cov,pathconds_d

def get_cov_otter(config,args):
    if CM.__vdebug__:
        assert is_config(config),config
        assert isinstance(args,dict) and 'pathconds_d' in args, args
    sids = set()        
    for cov,configs in args['pathconds_d'].itervalues():
        if any(config.hcontent.issuperset(c) for c in configs):
            for sid in cov:
                sids.add(sid)
    return sids

def run_gt(dom,pathconds_d,n=None,do_postprocess=True):
    """
    Obtain interactions using Otter's pathconds
    """
    if CM.__vdebug__:
        assert n is None or 0 <= n <= len(pathconds_d), n
        logger.warn("DEBUG MODE ON. Can be slow !")

    if n:
        rs = random.sample(pathconds_d.values(),n)
    else:
        rs = pathconds_d.itervalues()

    rs = [(HDict(c),covs) for covs,configs in rs for c in configs]
    allconfigs,allcovs = zip(*rs)
    logger.info("infer interactions using {} configs"
                .format(len(allconfigs)))
    st = time()
    cores_d,configs_d,covs_d = {},{},{}
    infer_covs(cores_d,allconfigs,allcovs,configs_d,covs_d,dom)
    logger.info("infer conds for {} covered lines ({}s)"
                .format(len(cores_d),time()-st))
    pp_cores_d = None
    if do_postprocess:
        logger.info("*** postprocess ***")
        pp_cores_d = postprocess(cores_d,covs_d,dom)

    return pp_cores_d,cores_d,configs_d,covs_d,dom

# Real executions
def void_run(cmd,print_cmd=True,print_outp=False):
    "just exec command, does not return anything"
    if print_cmd:print cmd
    try:
        rs_outp,rs_err = CM.vcmd(cmd)
        if print_outp: print rs_outp

        #IMPORTANT, command out the below allows
        #erroneous test runs, which can be helpful
        #to detect incorrect configs
        #assert len(rs_err) == 0, rs_err  
    except Exception as e:
        print e
        raise AssertionError("cmd '{}' failed".format(cmd))
    
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
    
    dom,_ = get_dom(dom_file)
    logger.info("dom_file '{}': {}"
                .format(dom_file,str_of_dom(dom,print_simple=True)))
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
        assert is_config(config),config
        
    tmpdir = '/var/tmp/'
    prog_exe = args['prog_exe']
    var_names = args['var_names']
    opts = ' '.join(config[vname] for vname in var_names)
    traces = os.path.join(tmpdir,'t.out')
    cmd = "{} {} > {}".format(prog_exe,opts,traces)
    void_run(cmd,print_cmd=False,print_outp=False)
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
    void_run(cmd,print_outp=True)
    #CM.pause()
    
    #run testsuite
    cmd = "{} {}".format(prog_exe,opts)
    void_run(cmd,print_outp=True)
    #CM.pause()

    #read traces from gcov
    #/path/prog.Linux.exe -> prog
    cmd = "gcov {}".format(prog_name)
    void_run(cmd,print_outp=True)
    gcov_dir = os.getcwd()
    sids = (parse_gcov(os.path.join(gcov_dir,f))
            for f in os.listdir(gcov_dir) if f.endswith(".gcov"))
    sids = set(CM.iflatten(sids))
    return sids

#Coreutils
def prepare_coreutils(prog_name):
    if CM.__vdebug__:
        assert isinstance(prog_name,str),prog_name
        
    bdir = getpath('~/Dropbox/git/config/benchmarks/coreutils')
    dom_file = os.path.join(bdir,"doms","{}.dom".format(prog_name))
    dom_file = getpath(dom_file)
    assert os.path.isfile(dom_file),dom_file

    dom,_ = get_dom(dom_file)
    logger.info("dom_file '{}': {}"
                .format(dom_file,str_of_dom(dom,print_simple=True)))

    bdir = os.path.join(bdir,'coreutils')
    prog_dir = os.path.join(bdir,'obj-gcov','src')
    prog_exe = os.path.join(prog_dir,prog_name)
    assert os.path.isfile(prog_exe),prog_exe
    logger.info("prog_exe: '{}'".format(prog_exe))

    dir_ = os.path.join(bdir,'src')
    assert os.path.isdir(dir_)
    args = {'var_names':dom.keys(),
            'prog_name': prog_name,
            'prog_exe': prog_exe,
            'get_cov': get_cov_coreutils,
            'prog_dir':prog_dir,
            'dir_': dir_}
    get_cov = lambda config: get_cov_wrapper(config,args)    
    return dom,get_cov

def getopts_coreutils(config,ks,delim=' '):
    opts = []
    for k in ks:
        if config[k] == "off":
            continue
        elif config[k] == "on":
            opts.append(k)
        elif k == "+" or config[k].startswith("="): #no space
            opts.append("{}{}".format(k,config[k]))
        else: #k v
            opts.append("{}{}{}".format(k,delim,config[k]))

    return ' '.join(opts)

def get_cov_coreutils(config,args):
    dir_ = args['dir_']
    prog_dir = args['prog_dir']
    
    prog_name = args['prog_name']
    prog_exe = args['prog_exe']
    var_names = args['var_names']
    opts = getopts_coreutils(config,var_names,delim=' ')

    #cleanup
    cmd = "rm -rf {}/*.gcov {}/*.gcda".format(dir_,prog_dir)
    void_run(cmd,print_outp=True)
    #CM.pause()
    
    #run testsuite
    cmd = "{} {}".format(prog_exe,opts)
    void_run(cmd,print_outp=True)
    #CM.pause()

    #read traces from gcov
    #/path/prog.Linux.exe -> prog
    src_dir = os.path.join(dir_,'src')
    cmd = "gcov {} -o {}".format(prog_name,prog_dir)
    void_run(cmd,print_outp=True)
    
    gcov_dir = os.getcwd()
    sids = (parse_gcov(os.path.join(gcov_dir,f))
            for f in os.listdir(gcov_dir) if f.endswith(".gcov"))
    sids = set(CM.iflatten(sids))
    return sids



#z3 stuff
def z3db_of_dom(dom):
    if CM.__vdebug__:
        assert is_dom(dom), dom
        
    z3db = OrderedDict()
    for k,vs in dom.iteritems():
        vs = sorted(list(vs))
        ttyp,tvals=z3.EnumSort(k,vs)
        rs = [vv for vv in zip(vs,tvals)]
        rs.append(('typ',ttyp))
        z3db[k]=(z3.Const(k,ttyp),dict(rs))
    return z3db

def z3expr_of_core(core,z3db,myf):
    if CM.__vdebug__:
        assert is_core(core) and core, core

    f = []
    for vn,vs in core.iteritems():
        vn_,vs_ = z3db[vn]
        f.append(z3util.myOr([vn_ == vs_[v] for v in vs]))

    return myf(f)

def z3expr_of_config(config,z3db):
    if CM.__vdebug__:
        assert is_config(config),config
        assert len(config) == len(z3db), (len(config), len(z3db))

    f = []
    for vn,vv in config.iteritems():
        vn_,vs_ = z3db[vn]
        f.append(vn_==vs_[vv])

    return z3util.myAnd(f)

def z3_get_sat_core(core,configs_expr,z3db):
    """
    Return a config satisfying core and not already in existing_configs
    >>> mdom = HDict([('a', frozenset(['1', '0'])), ('b', frozenset(['1', '0'])), ('c', frozenset(['1', '0', '2']))])
    >>> z3db = z3db_of_dom(mdom)

    >>> c1 = HDict([('a', '0'), ('b', '0'), ('c', '0')])
    >>> c2 = HDict([('a', '0'), ('b', '0'), ('c', '1')])
    >>> c3 = HDict([('a', '0'), ('b', '0'), ('c', '2')])

    >>> c4 = HDict([('a', '0'), ('b', '1'), ('c', '0')])
    >>> c5 = HDict([('a', '0'), ('b', '1'), ('c', '1')])
    >>> c6 = HDict([('a', '0'), ('b', '1'), ('c', '2')])

    >>> c7 = HDict([('a', '1'), ('b', '0'), ('c', '0')])
    >>> c8 = HDict([('a', '1'), ('b', '0'), ('c', '1')])
    >>> c9 = HDict([('a', '1'), ('b', '0'), ('c', '2')])

    >>> c10 = HDict([('a', '1'), ('b', '1'), ('c', '0')])
    >>> c11 = HDict([('a', '1'), ('b', '1'), ('c', '1')])
    >>> c12 = HDict([('a', '1'), ('b', '1'), ('c', '2')])

    >>> mconfigs = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11]
    >>> configs_expr =  z3util.myOr([z3expr_of_config(c,z3db) for c in mconfigs])

    >>> core = HDict([('a',frozenset(['1']))])
    >>> gc = z3_get_sat_core(core,configs_expr,z3db)
    >>> assert gc == c12
    
    >>> core = HDict([('a',frozenset(['0']))])
    >>> gc = z3_get_sat_core(core,configs_expr,z3db)
    >>> assert gc is None

    >>> core = HDict([('c',frozenset(['0','1']))])
    >>> gc = z3_get_sat_core(core,configs_expr,z3db)
    >>> assert gc is None

    >>> core = HDict([('c',frozenset(['0','2']))])
    >>> gc = z3_get_sat_core(core,configs_expr,z3db)
    >>> assert gc  == c12

    >>> core = HDict([('c',frozenset(['0','2']))])
    >>> configs = [c1]
    >>> gc = z3_get_sat_core(core,configs_expr,z3db)
    >>> print gc

    """
    core_expr = z3expr_of_core(core,z3db,z3util.myAnd)
    f = z3.And(core_expr,z3.Not(configs_expr))
    models = z3util.get_models(f,k=1)

    assert models is not None, models  #z3 cannot solve this
    if not models:  #not satisfy
        return None
    assert len(models)==1,models
    m = models[0]
    d = dict((str(v),m[v]) for v in m)
    return d


def doctestme():
    import doctest
    doctest.testmod()



"""
gt
ngircd:
vsftpd: (0,1,336), (1,4,101), (2,6,170), (3,5,1385), (4,24,410), (5,5,102), (6,6,35), (7,2,10)


old alg:
vsftpd: (0,1,336), (1,4,101), (2,6,170), (3,4,1373), (4,18,410), (5,8,114), (6,6,35), (7,2,10)
"""    
