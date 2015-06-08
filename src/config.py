#Smarter way to generate configs from cores
import tempfile
from time import time
import os.path
import itertools
import random

from config_miscs import HDict
import z3
import z3util
import vu_common as CM

logger = CM.VLog('config')
logger.level = CM.VLog.DETAIL
CM.VLog.PRINT_TIME = True
CM.__vdebug__ = False
do_comb_conj_disj = True
print_cov = False
#Data Structures

is_cov = lambda cov: (isinstance(cov,set) and
                      all(isinstance(sid,str) for sid in cov))
def str_of_cov(cov):
    return ','.join(sorted(cov)) if print_cov else str(len(cov))

is_valset = lambda vs: (isinstance(vs,frozenset) and vs and
                       all(isinstance(v,str) for v in vs))
def str_of_valset(s): return ','.join(sorted(s))
    
is_setting = lambda (k,v): isinstance(k,str) and isinstance(k,str)
def str_of_setting((k,v)): return '{}={}'.format(k,v)

is_csetting = lambda (k,vs): isinstance(k,str) and is_valset(vs)
def str_of_csetting((k,vs)): return '{}={}'.format(k,str_of_valset(vs))

is_dom = lambda dom: (isinstance(dom,HDict) and dom and
                    all(is_csetting(s) for s in dom.iteritems()))

def str_of_dom(dom,print_simple=False):
    if CM.__vdebug__:
        assert is_dom(dom),dom
        
    if print_simple:
        return("{} variables and {} possible configs"
               .format(len(dom),siz_of_dom(dom)))
    else:
        return '\n'.join("{}: {}".format(k,str_of_valset(dom[k]))
                         for k in dom)

is_config = lambda c: (isinstance(c,HDict) and c and
                       all(is_setting(s) for s in c.iteritems()))
def str_of_config(config,cov=None):
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

def str_of_core(core,delim=' '):
    if CM.__vdebug__:
        assert is_core(core), core
    if core:
        return delim.join(map(str_of_csetting,core.iteritems()))
    else:
        return 'true'

def is_pncore((pc,pd,nc,nd)):
    return all(c is None or is_core(c) for c in (pc,pd,nc,nd))

def sstr_of_pncore(pncore):
    if CM.__vdebug__:
        assert is_pncore(pncore),pncore    
    ss = ("{}: {}".format(s,str_of_core(c)) for s,c in
          zip('pc pd nc nd'.split(),pncore) if c is not None)
    return ', '.join(ss)

def str_of_pncore(pncore,dom=None):
    return fstr_of_pncore(pncore,dom) if dom else sstr_of_pncore(pncore)

is_cores_d = lambda cores_d: (isinstance(cores_d,dict) and
                              all(isinstance(sid,str) and is_pncore(c)
                                  for sid,c in cores_d.iteritems()))
def str_of_cores_d(cores_d,dom=None):
    if CM.__vdebug__:
        assert is_cores_d(cores_d),cores_d
        assert dom is None or is_dom(dom),dom
        
    return '\n'.join("{}. {}: {}"
                     .format(i+1,sid,str_of_pncore(cores_d[sid],dom))
                     for i,sid in enumerate(sorted(cores_d)))

is_mcores_d = lambda mcores_d: (isinstance(mcores_d,dict) and
                                all(is_pncore(c) and is_cov(cov)
                                    for c,cov in mcores_d.iteritems()))

def str_of_mcores_d(mcores_d,dom=None):
    if CM.__vdebug__:
        assert is_mcores_d(mcores_d),mcores_d
        assert dom is None or is_dom(dom),dom
    
    mc = sorted(mcores_d.iteritems(),
                key=lambda (core,cov): (stren_of_pncore(core),len(cov)))

    ss = ("{}. ({}) {}: {}"
          .format(i+1,
                  stren_of_pncore(core),str_of_pncore(core,dom),
                  str_of_cov(cov))
          for i,(core,cov) in enumerate(mc))

    return '\n'.join(ss)

def strens_of_mcores_d(mcores_d):
    """
    (strength,cores,sids)
    """
    if CM.__vdebug__:
        assert is_mcores_d(mcores_d),mcores_d
    
    strens = set(stren_of_pncore(core) for core in mcores_d)

    rs = []
    for stren in sorted(strens):
        cores = [c for c in mcores_d if stren_of_pncore(c) == stren]
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

pncore_mk_default = lambda : (None,None,None,None)
def settings_of_pncore(pncore):
    if CM.__vdebug__:
        assert is_pncore(pncore), pncore
    cores = (c for c in pncore if c)
    return set(s for c in cores for s in c.iteritems())

stren_of_pncore = lambda pncore: len(settings_of_pncore(pncore))
    
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
    key = (configs,core)
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
                    new_cd = HDict(e for e in new_cd.iteritems()
                                   if e not in new_cc.hcontent)
            
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
                                for sid,configs in covs_d))
is_configs_d = lambda configs_d: (isinstance(configs_d,dict) and
                                  all(is_config(config) and isinstance(cov,set)
                                      and all(isinstance(c,str) for c in cov)
                                      for config,cov in configs_d))


def infer_covs(cores_d,configs,covs,configs_d,covs_d,dom):
    if CM.__vdebug__:
        assert is_cores_d(cores_d),cores_d
        assert all(is_config(c) for c in configs) and configs, configs
        assert all(is_cov(c) for c in covs), covs
        assert len(configs) == len(covs), (len(configs),len(covs))
        assert is_configs_d(configs_d),configs_d
        assert is_covs_d(covs_d),covs_d
        assert is_dom(dom),dom
        
    new_covs,new_cores = set(),set()  #updated stuff
    
    if not configs:
        return new_covs,new_cores

    sids = set(cores_d.keys())

    #update configs_d and covs_d
    for config,cov in zip(configs,covs):
        for sid in cov:
            sids.add(sid)
            if sid not in covs_d:
                covs_d[sid] = set()
            covs_d[sid].add(config)

        assert config not in configs_d
        configs_d[config] = cov

            
    logger.detail("infer invs for {} cov using {} configs"
                  .format(len(sids),len(configs)))
    
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
def gen_configs_core(n,core,dom):
    """
    create n configs by changing n settings in core
    """
    if CM.__vdebug__:
        assert 0 < n <= len(core), n
        assert is_core(core) and core, core
        assert is_dom(dom),dom
    
    ks = random.sample(core.keys(),n)
    vss = (dom[x]-core[x] for x in ks)
    changes = [(k,v) for k,vs in zip(ks,vss) for v in vs]

    configs = []
    for x,y in changes:
        settings = []
        for k in dom:
            if k==x:
                v = y
            else:
                if k in core:
                    v = random.choice(list(core[k] if k in core else dom[k]))
                else:
                    v = random.choice(list(dom[k]))
            settings.append((k,v))

        configs.append(HDict(settings))
    return configs
    
def gen_configs_cores(core,dom):
    if CM.__vdebug__:
        assert is_pncore(core) and core, core        
        assert is_dom(dom),dom

    configs = (gen_configs_core(len(c),c,dom) for c in set(core) if c)
    configs = set(CM.iflatten(configs))
    return configs


def select_core(cores,ignore_strens,ignore_cores):
    if CM.__vdebug__:
        assert all(is_pncore(c) for c in cores) and cores,cores
        assert isinstance(ignore_strens,set),ignore_strens
        assert isinstance(ignore_cores,set),ignore_cores        

    min_stren = 2
    cores = [(core,stren_of_pncore(core)) for core in cores
             if core not in ignore_cores]
    cores = [(core,stren) for core,stren in cores
             if stren >= min_stren and stren not in ignore_strens]
    if cores:
        core = max(cores,key=lambda (c,stren):stren)[0]
        return core
    else:
        return None
    
def eval_samples(samples,get_cov,configs_d):
    if CM.__vdebug__:
        assert all(is_config(c) for c in samples), samples
        assert callable(get_cov),get_cov
        assert is_configs_d(configs_d),configs_d

    st = time()
    rs = ((c,get_cov(c)) for c in samples if c not in configs_d)
    samples,covs = zip(*rs)
    return samples,covs,time() - st

def intgen(dom,get_cov,seed=None,tmpdir=None,cover_siz=None,
          config_default=None,prefix='vu'):
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
    max_stuck = 2
    cur_stuck = 0
    ignore_strens = set()
    ignore_cores = set()
    configs_d = {}
    covs_d = {}
    sel_core = pncore_mk_default()
  
    #begin
    st = time()
    ct = st
    cov_time = 0.0
    
    #initial samples
    if cover_siz:
        cstren,max_confs = cover_siz
        if cstren == 0:
            if max_confs < 0:
                samples = gen_configs_full(dom)
                logger.info("gen all {} configs".format(len(samples)))
            else:
                samples = gen_configs_rand(max_confs,dom)
                logger.info("gen {} rand configs".format(len(samples)))
        else:
            samples = gen_configs_tcover(cstren,seed,tmpdir)
            samples_rand_n = max_confs - len(samples)

            if samples_rand_n:
                samples_ = gen_configs_rand(samples_rand_n,dom)
                samples.extend(samples_)

            samples = list(set(samples))
            logger.info("gen {} {}-cover configs"
                        .format(len(samples),cstren))
                                
    else:
        samples = gen_configs_tcover1(dom)

    if config_default:
        samples.append(config_default)

    samples,covs,ctime = eval_samples(samples,get_cov,configs_d)
    cov_time += ctime
    new_covs,new_cores = infer_covs(cores_d,samples,covs,configs_d,covs_d,dom)

    while True:
        
        #save info
        ct_ = time();etime = ct_ - ct;ct = ct_
        robj = (cur_iter,etime,cov_time,
                samples,covs,
                new_covs,new_cores,
                sel_core,
                cores_d)
        rfile = os.path.join(tmpdir,"{}.tvn".format(cur_iter))
        
        CM.vsave(rfile,robj)
        print_iter_stat(robj)
        if cover_siz:
            break
        
        cur_iter += 1
        sel_core = select_core(set(cores_d.values()),
                               ignore_strens,ignore_cores)
        if sel_core:
            ignore_cores.add(sel_core)
        else:
            logger.info('select no core for refinement, '
                        'done at iter {}'.format(cur_iter))

            break
        samples = gen_configs_cores(sel_core,dom)
        samples,covs,ctime = eval_samples(samples,get_cov,configs_d)
        cov_time += ctime
        new_covs,new_cores = infer_covs(cores_d,samples,covs,
                                        configs_d,covs_d,dom)

        if new_covs or new_cores: #progress
            cur_stuck = 0
            ignore_strens.clear()
            
        else: #no progress
            cur_stuck += 1
            if cur_stuck >= max_stuck:
                ignore_strens.add(stren_of_pncore(sel_core))
                cur_stuck = 0

    cores_d = analyze(cores_d,covs_d,dom)
    mcores_d = merge_cores_d(cores_d)
    logger.debug("mcores_d has {} items\n".format(len(mcores_d)) +
                 str_of_mcores_d(mcores_d,dom))
    logger.info(strens_str_of_mcores_d(mcores_d))
    
    logger.info(str_of_summary(seed,cur_iter,time()-st,cov_time,
                               len(configs_d),len(cores_d),
                               tmpdir))

    return cores_d,configs_d,covs_d

#Shortcuts
def intgen_full(dom,get_cov,tmpdir=None,prefix='vu'):
    return intgen(dom,get_cov,seed=None,tmpdir=tmpdir,
                 cover_siz=(0,-1),config_default=None,
                 prefix=prefix)

def intgen_rand(dom,get_cov,rand_n,seed=None,tmpdir=None,prefix='vu'):
    return intgen(dom,get_cov,seed=seed,tmpdir=tmpdir,
                 cover_siz=(0,rand_n),config_default=None,
                 prefix=prefix)

#postprocess
def analyze(cores_d,covs_d,dom):
    if CM.__vdebug__:
        assert is_cores_d(cores_d),cores_d
        assert is_covs_d(covs_d),covs_d
        assert is_dom(dom),dom
        
    logger.info("analyze {} interactions".format(len(cores_d)))
    logger.debug("verify ...")
    rs = [(sid,verify_pncore(core,covs_d[sid],dom))
          for sid,core in cores_d.iteritems()]
    logger.debug("simplify ...")
    rs = [(sid,simplify_pncore(core,dom)) for sid,core in rs]
    return dict(rs)

def reformat((pc,pd,nc,nd),dom):
    if pd:
        pd = neg_of_core(pd,dom)
    if nc:
        nc = neg_of_core(nc,dom)
    return (pc,pd,nc,nd)
    
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

    #if pc is None, i.e., no data then everything is None
    pncore = (pc,pd,nc,nd)
    if old_pncore != pncore:
        logger.debug("{} -> {}".
                     format(str_of_pncore(old_pncore),
                            str_of_pncore(pncore)))
        
    return pncore
        
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
        raise AssertionError("inconsistent ? {}"
                             .format(str_of_pncore((pc,pd,nc,nd))))

    pncore = (pc,pd,nc,nd)
    if old_pncore != pncore:
        logger.debug("{} -> {}".
                     format(str_of_pncore(old_pncore),
                            str_of_pncore(pncore)))        
    return pncore

def fstr_of_pncore((pc,pd,nc,nd),dom):
    """
    Assumption: all 4 cores are verified and simplified
    
    """
    if CM.__vdebug__:
        assert pc is None or is_core(pc) and pc, pc
        assert pd is None or is_core(pd) and pd, pd
        assert nc is None or is_core(nc) and nc, nc
        assert nd is None or is_core(nd) and nd, nd
        assert ((pc is None and pd is None) or
                (nc is None and nd is None)), (pc,pd,nc,nd)
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
            
    if (nc is None and nd is None):
        #pc & not(pd)
        ss = _cd(pc,pd,' & ')
    else:
        #not(nc & not(nd))  =  not(nc) | nd
        ss = _cd(nd,nc,' | ')
        
    return ss

#Miscs
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


def print_iter_stat(robj):
    (citer,etime,ctime,samples,covs,new_covs,new_cores,sel_core,cores_d) = robj
    logger.info("ITER {}, ".format(citer) +
                "{0:.2f}s, ".format(etime) +
                "{0:.2f}s eval, ".format(ctime) +
                "total: {} samples, {} covs, {} cores, "
                .format(0,0,0)+
                "new: {} samples, {} covs, {} cores, "
                .format(len(samples),len(new_covs),len(new_cores)) +
                "{}".format("** progress **"
                            if new_covs or new_cores else ""))
                            
                
    logger.debug('select a core of stren {}'.format(stren_of_pncore(sel_core)))
    logger.debug('sel_core: ' + str_of_pncore(sel_core))
    logger.debug('create {} samples'.format(len(samples)))
    logger.detail('\n' + str_of_configs(samples,covs))
    
    mcores_d = merge_cores_d(cores_d)
    logger.detail('mcores\n{}'.format(str_of_mcores_d(mcores_d)))
    logger.info(strens_str_of_mcores_d(mcores_d))
    return mcores_d

def str_of_summary(seed,iters,ntime,ctime,nsamples,ncovs,tmpdir):
    s = ("Summary: Seed {}, Iters {}, Time ({}, {}), Samples {}, Covs {}, Tmpdir '{}'"
         .format(seed,iters,ntime,ctime,nsamples,ncovs,tmpdir))
    return s

#Tests
getpath = lambda f: os.path.realpath(os.path.expanduser(f))
def void_run(cmd,print_outp=False):
    "just exec command, does not return anything"
    print cmd
    try:
        rs_outp,rs_err = CM.vcmd(cmd)
        if print_outp: print rs_outp
        assert len(rs_err) == 0, rs_err
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
        assert isinstance(prog_file,str),prog_file        
    import platform
    
    dir_ = getpath('~/Dropbox/git/config/benchmarks/examples')
    dom_file = getpath(os.path.join(dir_,dom_file))
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
            'get_cov': get_cov_motiv_gcov,  #_gcov
            'dir_': dir_}
    get_cov = lambda config: get_cov_wrapper(config,args)
    return dom,get_cov

def get_cov_motiv(config,args):
    """
    Traces read from stdin
    """
    tmpdir = '/var/tmp/'
    prog_exe = args['prog_exe']
    var_names = args['var_names']    
    opts = ' '.join(config[vname] for vname in var_names)
    traces = os.path.join(tmpdir,'t.out')
    cmd = "{} {} > {}".format(prog_exe,opts,traces)
    void_run(cmd,print_outp=False)
    traces = list(CM.iread_strip(traces))
    return traces

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
        assert isinstance(prog_file,str),prog_file
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
        else:
            opts.append("{}{}{}".format(k,delim,config[k]))

    return ' '.join(opts)

def get_cov_coreutils(config,args):
    dir_ = args['dir_']
    prog_dir = args['prog_dir']
    
    prog_name = args['prog_name']
    prog_exe = args['prog_exe']
    var_names = args['var_names']
    opts = getopts_coreutils(config,var_names,delim='=')

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
    get_cov = lambda config: otter_get_cov(config,args)
    return dom,get_cov,pathconds_d


def otter_get_cov(config,args):
    if CM.__vdebug__:
        assert is_config(config),config

    sids = set()        
    pathconds_d = args['pathconds_d']    
    for cov,samples in pathconds_d.itervalues():
        if any(config.hcontent.issuperset(sample) for sample in samples):
            for sid in cov:
                sids.add(sid)

    return sids

def run_gt(dom,pathconds_d,n=None):
    """
    Obtain interactions using Otter's pathconds
    """
    if CM.__vdebug__:
        assert n is None or 0 <= n <= len(pathconds_d), n

    if n:
        rs = random.sample(pathconds_d.values(),n)
    else:
        rs = pathconds_d.itervalues()

    rs = [(HDict(s),covs) for covs,samples in rs for s in samples]
    allsamples,allcovs = zip(*rs)

    logger.info("infer interactions using {} samples"
                .format(len(allsamples)))
    st = time()
    cores_d,configs_d,covs_d = {},{},{}
    infer_covs(cores_d,allsamples,allcovs,configs_d,covs_d,dom)
    logger.info("infer conds for {} covered lines ({}s)"
                .format(len(cores_d),time()-st))
    return cores_d,configs_d,covs_d

#z3 stuff
def z3db_of_dom(dom):
    if CM.__vdebug__:
        assert is_dom(dom), dom
        
    z3db = {}  #{'x':(x,{'true':True})}
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
