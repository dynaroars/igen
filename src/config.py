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
logger.level = CM.VLog.DEBUG
CM.VLog.PRINT_TIME = True
CM.__vdebug__ = True
do_comb_conj_disj = True

#Data Structures
print_cov = True
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
def str_of_dom(dom):
    return '\n'.join("{}: {}".format(k,str_of_valset(dom[k]))
                     for k in dom)

is_config = lambda c: (isinstance(c,HDict) and c and
                       all(is_setting(s) for s in c.iteritems()))
def str_of_config(config):
    return ' '.join(map(str_of_setting,config.iteritems()))

def str_of_configs(configs,covs=None):
    if covs:
        ss = ("{}: {}".format(str_of_config(c),str_of_cov(cov))
              for c,cov in zip(configs,covs))
    else:
        ss = (str_of_config(c) for c in configs)


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

def str_of_pncore((pc,pd,nc,nd)):
    if CM.__vdebug__:
        assert is_pncore((pc,pd,nc,nd)), (pc,pd,nc,nd)
        
    ss = []
    if pc is not None:
        ss.append("pc: {}".format(str_of_core(pc)))
    if pd is not None:
        ss.append("pd: {}".format(str_of_core(pd)))
    if nc is not None:
        ss.append("nc: {}".format(str_of_core(nc)))
    if nd is not None:
        ss.append("nd: {}".format(str_of_core(nd)))

    return ', '.join(ss)


is_cores_d = lambda cores_d: (isinstance(cores_d,dict) and
                              all(isinstance(sid,str) and is_pncore(c)
                                  for sid,c in cores_d.iteritems()))
def str_of_cores_d(cores_d):
    if CM.__vdebug__:
        assert is_cores_d(cores_d),cores_d
    return '\n'.join("{}. {}: {}".format(i+1,sid,
                                         str_of_pncore(cores_d[sid]))
                     for i,sid in enumerate(sorted(cores_d)))

is_mcores_d = lambda mcores_d: (isinstance(mcores_d,dict) and
                                all(is_pncore(c) and is_cov(cov)
                                    for c,cov in mcores_d.iteritems()))
def str_of_mcores_d(mcores_d):
    if CM.__vdebug__:
        assert is_mcores_d(mcores_d),mcores_d
    
    mc = sorted(mcores_d.iteritems(),
                key=lambda (core,cov): (stren_of_pncore(core),len(cov)))

    ss = ("{}. ({}) {}: {}"
          .format(i+1,
                  stren_of_pncore(core),str_of_pncore(core),
                  str_of_cov(cov))
          for i,(core,cov) in enumerate(mc))

    return '\n'.join(ss)

# Functions on Data Structures
config_c_implies = lambda c,d: len(d) == 0 or all(c[k] in d[k] for k in d)
config_d_implies = lambda c,d: len(d) == 0 or any(c[k] in d[k] for k in d)

def neg_of_core(core,dom):
    if CM.__vdebug__:
        assert is_core(core),core
        assert is_dom(dom),dom

    return HDict((k,dom[k]-core[k]) for k in core)

pncore_mk_default = lambda : (None,None,None,None)
def settings_of_pncore(pncore):
    if CM.__vdebug__:
        assert is_pncore(pncore), pncore
        
    settings = set()
    for c in pncore:
        if c:
            for s in c.iteritems():
                settings.add(s)
    return settings
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

    settings = []
    for k,vs in core.iteritems():
        vs_ = f(k,vs,len(dom[k]))
        if vs_:
            settings.append((k,frozenset(vs_)))
            
    core = HDict(settings)
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
    
def infer_covs(configs,covs,cores_d,dom,configs_cache):
    if CM.__vdebug__:
        assert all(is_config(c) for c in configs) and configs, configs
        assert all(is_cov(cov) for cov in covs), covs
        assert len(configs) == len(covs), (len(configs),len(covs))
        assert is_cores_d(cores_d), cores_d
        assert is_dom(dom), dom

    new_covs,new_cores = set(),set()  #updated stuff
    
    if not configs:
        return new_covs,new_cores

    sids = set(cores_d.keys())
    for cov in covs:
        for sid in cov:
            sids.add(sid)
            
    logger.debug("infer invs for {} cov using {} configs"
                 .format(len(sids),len(configs)))
    
    cache = {}
    for i,sid in enumerate(sorted(sids)):
        #logger.debug(sid)
        if sid in cores_d:
             pc,pd,nc,nd = cores_d[sid]
        else:
             pc,pd,nc,nd = pncore_mk_default()
             new_covs.add(sid)
             
        new_pc,new_pd,new_nc,new_nd = pc,pd,nc,nd
        
        pconfigs,nconfigs = [],[]
        for config,cov in zip(configs,covs):
            if sid in cov:
                pconfigs.append(config)
            else:
                nconfigs.append(config)

        if pconfigs:
            new_pc = infer_cache(pc,pconfigs,dom,cache)

        if nconfigs:
            new_nc = infer_cache(nc,nconfigs,dom,cache)

        if do_comb_conj_disj:
            if new_pc:
                nconfigs = [c for c in configs_cache
                           if sid not in configs_cache[c] and
                           config_c_implies(c,new_pc)]
                if nconfigs:
                    new_pd = infer_cache(pd,nconfigs,dom,cache) 
                    if new_pd:
                        new_pd = HDict(e for e in new_pd.iteritems()
                                       if e not in new_pc.hcontent)
                    
            if new_nc:
                pconfigs = [c for c in configs_cache
                            if sid in configs_cache[c] and
                            config_c_implies(c,new_nc)]
                if pconfigs:
                    new_nd = infer_cache(nc,pconfigs,dom,cache)
                    if new_nd:
                        new_nd = HDict(e for e in new_nd.iteritems()
                                       if e not in new_nc.hcontent)

        if not (new_pc,new_pd,new_nc,new_nd) == (pc,pd,nc,nd): #progress
            new_cores.add(sid)
            cores_d[sid] = (new_pc,new_pd,new_nc,new_nd)

    return new_covs,new_cores

#Interpretation algorithm


#Interative algorithm
def gen_configs_core(n,core,dom):
    """
    create n configs by changing n settings in core
    """
    if CM.__vdebug__:
        assert n > 0, n
        assert is_core(core) and core, core
        assert is_dom(dom),dom

    vnames = random.sample(core.keys(),n)
    vvals = [dom[x]-core[x] for x in vnames]
    
    changes = []
    for vname,vval in zip(vnames,vvals):
        for v in vval:
            changes.append((vname,v))

    configs = []
    for x,y in changes:
        settings = []
        for k in dom:
            if k==x:
                v = y
            else:
                if k in core:
                    v = random.choice(list(core[k]))
                else:
                    v = random.choice(list(dom[k]))
            settings.append((k,v))

        config = HDict(settings)
        configs.append(config)
    return configs
    
def gen_configs_cores(core,dom):
    if CM.__vdebug__:
        assert is_pncore(core) and core, core        
        assert is_dom(dom),dom

    cores = set(c for c in core if c)
    configs = []        
    for c in cores:
        configs.extend(gen_configs_core(len(c),c,dom))
        
    configs = set(configs)
    return configs

def select_core(cores,ignore_strens,ignore_cores):
    if CM.__vdebug__:
        assert all(is_pncore(c) for c in cores) and cores,cores
        assert isinstance(ignore_strens,set),ignore_strens
        assert isinstance(ignore_cores,set),ignore_cores        

    cores = [(core,stren_of_pncore(core)) for core in cores
             if core not in ignore_cores]
    cores = [(core,stren) for core,stren in cores
             if stren and stren not in ignore_strens]
    if cores:
        core = max(cores,key=lambda (c,stren):stren)[0]
        return core
    else:
        return None
    
def eval_samples(samples,get_cov,cache):
    if CM.__vdebug__:
        assert all(is_config(c) for c in samples), samples
        assert isinstance(cache,dict),cache

    st = time()
    samples_ = []
    for config in samples:
        if config not in cache:
            cov = get_cov(config)
            cache[config]=set(cov)
            samples_.append(config)

    samples = samples_
    covs = [cache[config] for config in samples]

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
    cur_iter = 0
    max_stuck = 2
    cur_stuck = 0
    ignore_strens = set()
    ignore_cores = set()
    configs_cache = {}
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
                logger.info("Gen all {} configs".format(len(samples)))
            else:
                samples = gen_configs_rand(max_confs,dom)
                logger.info("Gen {} rand configs".format(len(samples)))
        else:
            samples = gen_configs_tcover(cstren,seed,tmpdir)
            samples_rand_n = max_confs - len(samples)

            if samples_rand_n:
                samples_ = gen_configs_rand(samples_rand_n,dom)
                samples.extend(samples_)

            samples = list(set(samples))
            logger.info("Gen {} {}-cover configs"
                        .format(len(samples),cstren))
                                
    else:
        samples = gen_configs_tcover1(dom)

    if config_default:
        samples.append(config_default)

    samples,covs,ctime = eval_samples(samples,get_cov,configs_cache)
    cov_time += ctime
    new_covs,new_cores = infer_covs(samples,covs,cores_d,dom,configs_cache)

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
        samples,covs,ctime = eval_samples(samples,get_cov,configs_cache)
        cov_time += ctime
        new_covs,new_cores = infer_covs(samples,covs,cores_d,dom,configs_cache)

        if new_covs or new_cores: #progress
            cur_stuck = 0
            ignore_strens.clear()
            
        else: #no progress
            cur_stuck += 1
            if cur_stuck >= max_stuck:
                ignore_strens.add(stren_of_pncore(sel_core))
                cur_stuck = 0

    #postprocess
    #cores_d = postprocess(cores_d,configs_cache,dom)
    
    logger.info(str_of_summary(seed,cur_iter,time()-st,cov_time,
                               len(configs_cache),len(cores_d),
                               tmpdir))

    return cores_d,configs_cache

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
def analyze(cores_d,configs_cache,dom):
    if CM.__vdebug__:
        assert is_cores_d(cores_d),cores_d

    rs = {}
    for sid,core in cores_d.iteritems():
        print sid
        print str_of_pncore(core)
        if configs_cache:
            configs = [c for c in configs_cache if sid in configs_cache[c]]
            if configs:
                core = verify_pncore(core,configs,dom)
                print 'verify'
                print str_of_pncore(core)
                
        core = simplify_pncore(core,dom)
        print 'simplify'
        print str_of_pncore(core)
        print 'final', fstr_of_pncore(core,dom)
        rs[sid]=core
    return rs


def verify_pncore((pc,pd,nc,nd),configs,dom):
    if CM.__vdebug__:
        assert is_pncore((pc,pd,nc,nd)), (pc,pd,nc,nd)
        assert pc is not None, pc #this never could happen
        #nc is None => pd is None
        assert (nc is not None or pd is None), (nc,nd)
        assert all(is_config(c) for c in configs) and configs, configs
        assert is_dom(dom),dom
        
    #traces => pc & neg(pd)
        
    if pc:
        assert all(config_c_implies(c,pc) for c in configs), pc
        
    if pd:
        pd_n = neg_of_core(pd,dom)
        if not all(config_d_implies(c,pd_n) for c in configs):
            print 'pd {} invalid'.format(str_of_core(pd))
            pd = None

    #neg traces => nc & neg(nd)
    #pos traces => neg(nc & neg(nd))
    #post traces => nd | neg(nc) 
    if nc is not None and nd is None:
        nc_n = neg_of_core(nc,dom)
        print nc_n
        if not all(config_d_implies(c,nc_n) for c in configs):
            print 'nc {} invalid'.format(str_of_core(nc))
            nc = None
    elif nc is None and nd is not None:
        if not all(config_c_implies(c,nd) for c in configs):
            print 'nd {} invalid'.format(str_of_core(nd))
            nd = None
    elif nc is not None and nd is not None:
        nc_n = neg_of_core(nc,dom)
        if not all(config_c_implies(c,nd) or config_d_implies(c,nc_n)
                   for c in configs):
            print 'nc & nd invalid'
            nc = None
            nd = None

    #if pc is None, i.e., no data then everything is None
    
    return (pc,pd,nc,nd)
        
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
    fs = []
    if pc:
        f = z3expr_of_core(pc,z3db,myf=z3util.myAnd)
        fs.append(f)
    if pd:
        pd_n = neg_of_core(pd,dom)
        f = z3expr_of_core(pd_n,z3db,myf=z3util.myOr)
        fs.append(f)

    pf = z3util.myAnd(fs)

    fs = []
    if nd:
        f = z3expr_of_core(nd,z3db,myf=z3util.myAnd)
        fs.append(f)
    if nc:
        nc_n = neg_of_core(nc,dom)
        f = z3expr_of_core(nc_n,z3db,myf=z3util.myOr)
        fs.append(f)
    nf = z3util.myOr(fs)

    # print pc,pd
    # print pf    
    # print nc,nd
    # print nf
    
    if z3util.is_tautology(z3.Implies(pf,nf)):
        nc = None
        nd = None
    elif z3util.is_tautology(z3.Implies(nf,pf)):
        pc = None
        pd = None
    else:
        raise AssertionError("inconsistent ? {}"
                             .format(str_of_pncore((pc,pd,nc,nd))))

    return (pc,pd,nc,nd)

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

    def _cd(ccore,dcore):
        ss = []
        if ccore:
            ss.append(_f(ccore,' & '))
        if dcore:
            print 'dcore'
            print dcore
            assert is_dom(dom),dom
            dcore_n = neg_of_core(dcore,dom)
            ss.append(_f(dcore_n, ' | '))
        return ss
            
    ss = ''
    if (nc is None and nd is None):
        #pc & not(pd)
        ss = _cd(pc,pd)
        ss = ' & '.join(ss)
    else:
        #not(nc & not(nd))  =  not(nc) | nd
        ss = _cd(nd,nc)
        ss = ' | '.join(ss)
        
    return ss

#Miscs

def get_dom(dom_file):
    def get_lines(lines):
        rs = (line.split() for line in lines)
        rs = ((parts[0],frozenset(parts[1:])) for parts in rs)
        return rs

    dom_file = os.path.realpath(dom_file)
    dom = HDict(get_lines(CM.iread_strip(dom_file)))

    config_default = None
    dom_file_default = os.path.realpath(dom_file+'.default')
    if os.path.isfile(dom_file_default):
        rs = get_dom_lines(CM.iread_strip(dom_file_default))
        config_default = HDict((k,list(rs[k])[0]) for k in rs)

    return dom,config_default

#generate configs
siz_of_dom = lambda d: CM.vmul(len(y) for _,y in d)
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
                            
                
    logger.debug('sel_core: {}'.format(sel_core))
    logger.debug('samples\n'+str_of_configs(samples,covs))
    print str_of_cores_d(cores_d)
    mcores_d = merge_cores_d(cores_d)
    logger.debug('mcores\n{}'.format(str_of_mcores_d(mcores_d)))
    #logger.info(mcores_d.stren_str)
    return mcores_d

def str_of_summary(seed,iters,ntime,ctime,nsamples,ncovs,tmpdir):
    s = ("Summary: Seed {}, Iters {}, Time ({}, {}), Samples {}, Covs {}, Tmpdir '{}'"
         .format(seed,iters,ntime,ctime,nsamples,ncovs,tmpdir))
    return s

#Tests

def prepare_motiv(dom_file,prog_file):
    import ex_motiv_run
        
    dom,_ = get_dom(dom_file)
    prog =  os.path.realpath(prog_file)
    assert os.path.isfile(prog), prog
    logger.info("dom_file: '{}', ".format(dom_file) + 
                "prog_file: '{}'".format(prog_file))
    args = {'prog': prog,
            'varnames':dom.keys()}
    get_cov = lambda config: ex_motiv_run.get_cov(config,args)
    return dom,get_cov

def prepare_otter(prog):
    dir_ = '~/Src/Devel/iTree_stuff/expData/{}'.format(prog)
    dir_ = os.path.realpath(os.path.expanduser(dir_))
    dom_file = dir_ + '/possibleValues.txt'
    exp_dir = dir_ + '/rawExecutionPaths'.format(prog)
    pathconds_d_file = dir_ + '/pathconds_d.tvn'.format(prog)

    dom,_ = get_dom(dom_file)
    if os.path.isfile(pathconds_d_file):
        logger.info("read from '{}'".format(pathconds_d_file))
        pathconds_d = CM.vload(pathconds_d_file)
    else:
        import ex_otter        
        logger.info("read from '{}'".format(exp_dir))
        pathconds_d = ex_otter.get_data(exp_dir)

    logger.info("read {} pathconds".format(len(pathconds_d)))
    args = {'pathconds_d':pathconds_d}
    get_cov = lambda config: otter_get_cov(config,args)
    
    return dom,get_cov,pathconds_d

def otter_get_cov(config,args):
    if CM.__vdebug__:
        assert is_config(config),config
        
    pathconds_d = args['pathconds_d']
    sids = set()
    for covs,samples in pathconds_d.itervalues():
        if any(config.hcontent.issuperset(sample) for sample in samples):
            for sid in covs:
                sids.add(sid)

    sids = list(sids)
    return sids

    
def run_gt(dom,pathconds_d,n=None):
    """
    Obtain conditions using Otter's pathconds
    """
    if CM.__vdebug__:
        assert n is None or 0 <= n <= pathconds_d, n
    allsamples,allcovs = [],[]
    ks = random.sample(pathconds_d,n) if n else pathconds_d.iterkeys()
    for k in ks:
        covs,samples = pathconds_d[k]
        for sample in samples:
            allsamples.append(HDict(sample))
            allcovs.append(covs)

    logger.info("infer interactions using {} samples"
                .format(len(allsamples)))
    st = time()
    cores_d = {}
    configs_cache = {}
    infer_covs(allsamples,allcovs,cores_d,dom,{})
    logger.info("infer conds for {} covered lines ({}s)"
                .format(len(cores_d),time()-st))
    return cores_d


#z3 stuff
def z3db_of_dom(dom):
    if CM.__vdebug__:
        assert is_dom(dom), dom
        
    z3db = {}  #{'x':(x,{'true':True})}
    for k,vs in dom.iteritems():
        vs = sorted(list(vs))
        ttyp,tvals=z3.EnumSort(k,vs)
        rs = []
        for v,v_ in zip(vs,tvals):
            rs.append((v,v_))
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



    
