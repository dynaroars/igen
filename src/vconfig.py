import tempfile
import random
import os.path
import itertools
from collections import OrderedDict
from time import time

import z3
from vconfig_miscs import HDict
import vu_common as CM
import z3util

logger = CM.VLog('config')
logger.level = CM.VLog.DEBUG
CM.VLog.PRINT_TIME = True

vdebug = True
print_cov = True

### DATA STRUCTURES ###

is_cov = lambda cov: (isinstance(cov,frozenset) and
                      all(isinstance(sid,str) for sid in cov))

def str_of_cov(cov):
    if vdebug:
        assert is_cov(cov),cov
    
    return ','.join(sorted(cov)) if print_cov else str(len(cov))

is_valset = lambda vs: (isinstance(vs,frozenset) and vs and
                        all(isinstance(v,str) for v in vs))
def str_of_valset(s):
    if vdebug:
        assert is_valset(s),s

    return ','.join(sorted(s))        

## DOM ##
is_dom = lambda d: (isinstance(d,dict) and
                    all(isinstance(vn,str) and
                        is_valset(vs) for vn,vs in d.iteritems()))

def str_of_dom(dom):
    if vdebug:
        assert is_dom(dom),dom

    return '\n'.join("{}: {}".format(k,str_of_valset(vs))
                     for k,vs in sorted(dom.iteritems()))

def get_dom(dom_file):
    def get_lines(lines):
        rs = (line.split() for line in lines)
        rs = ((parts[0],frozenset(parts[1:])) for parts in rs)
        return rs
                      
    dom_file = os.path.realpath(dom_file)
    dom = OrderedDict(get_lines(CM.iread_strip(dom_file)))
    
    config_default = None
    dom_file_default = os.path.realpath(dom_file+'.default')
    if os.path.isfile(dom_file_default):
        rs = get_dom_lines(CM.iread_strip(dom_file_default))
        config_default = HDict((k,list(v)[0]) for k,v in rs)
                               
    if vdebug:
        assert is_dom(dom),dom
        assert (config_default is None
                or is_config(config_default)), config_default

    return dom,config_default

def z3_of_dom(dom):
    if vdebug:
        assert is_dom(dom),dom

    z3dom = {}  #{'x':(x,{'true':True})}
    for k,vs in dom.iteritems():
        vs = sorted(list(vs))
        ttyp,tvals=z3.EnumSort(k,vs)
        rs = []
        for v,v_ in zip(vs,tvals):
            rs.append((v,v_))
        rs.append(('typ',ttyp))

        z3dom[k]=(z3.Const(k,ttyp),dict(rs))
        
    return z3dom



## CONFIG ##
is_setting = lambda (vn,vv): isinstance(vn,str) and isinstance(vv,str)

def str_of_setting((vn,vv)):
    if vdebug:
        assert is_setting((vn,vv)),(vn,vv)

    return '{}={}'.format(vn,vv)


is_config = lambda c: (isinstance(c,HDict) and 
                       all(is_setting(s) for s in c.iteritems()))

def str_of_config(c):
    if vdebug:
        assert is_config(c), c

    return ' '.join(map(str_of_setting,c.iteritems()))


is_configs = lambda cs: (isinstance(cs,list) and
                         all(is_config(c) for c in cs))

def str_of_configs(configs,covs=None):
    if vdebug:
        assert is_configs(configs), configs
        assert (covs is None or 
                (len(covs) == len(configs) and
                 all(is_cov(cov) for cov in covs))), covs
        
    if  covs:
        return '\n'.join("{}. {}: {}"
                         .format(i+1,str_of_config(c),str_of_cov(cov))
                         for i,(c,cov) in enumerate(zip(configs,covs)))
    else:
        return '\n'.join("{}. {}".format(i,str_of_config(c))
                         for i,c in enumerate(configs))


def z3_of_configs(configs,z3dom):
    return z3util.myOr([z3_of_config(c,z3dom) for c in configs])
    

## CORE ##
is_csetting = lambda (vn,vs): isinstance(vn,str) and is_valset(vs)

def str_of_csetting((vn,vs)):
    if vdebug:
        assert is_csetting((vn,vs)), (vn,vs)

    return '{}={}'.format(vn,str_of_valset(vs))


is_core = lambda c: (c is None or
                     (isinstance(c,HDict) and
                      all(is_csetting(s) for s in c.iteritems())))

def str_of_core(c):
    if vdebug:
        assert is_core(c), c

    if c is None:
        return str(c) #never reached
    if not c:
        return "true"  #no constraint
    return ' '.join(map(str_of_csetting,c.iteritems()))


def z3_of_core(c,z3dom):
    if vdebug:
        assert is_core(c) and c,c

    f = []
    for vn,vs in c.iteritems():
        vn_,vs_ = z3dom[vn]
        f.append(z3util.myOr([vn_ == vs_[v] for v in vs]))

    return z3util.myAnd(f)


is_cdcore = lambda(cc,dc): is_core(cc) and is_core(dc)

def str_of_cdcore((cc,dc)):
    return "c({}), d({})".format(str_of_core(cc),str_of_core(dc))

def z3_of_cdcore((cc,dc),z3dom):
    """
    >>> z3dom = z3_of_dom(OrderedDict([('listen', frozenset(['1', '0'])), ('timeout', frozenset(['1', '0'])), ('ssl', frozenset(['1', '0'])), ('local', frozenset(['1', '0'])), ('anon', frozenset(['1', '0'])), ('log', frozenset(['1', '0'])), ('chunks', frozenset(['0', '65536', '4096', '2048']))]))

    >>> cdc = (None,None)
    >>> assert z3_of_cdcore(cdc,z3dom) is None

    >>> cdc = (HDict([('listen', frozenset(['1'])), ('timeout', frozenset(['1']))]), None)
    >>> z3_of_cdcore(cdc,z3dom)
    And(listen == 1, timeout == 1)

    """
    if vdebug:
        assert is_cdcore((cc,dc)), (cc,dc)

    conj = lambda: z3_of_core(cc,z3dom)
    disj = lambda: z3.Not(z3_of_core(dc,z3dom))


    if cc is None: # no data
        return None
    elif cc:
        if dc:
            return z3util.myAnd(conj(),disj())
        else:
            return conj()
    else:
        if dc:
            return disj()
        else:
            return z3util.TRUE
            

def z3_of_pncore((pc,nc),z3dom):
    z3_pc = z3_of_cdcore(pc,z3dom)
    if z3_pc:
        z3_pc = z3.simplify(z3_pc)
        
    z3_nc = z3_of_cdcore(nc,z3dom)
    if z3_nc:
        z3_nc = z3.Not(z3_nc)
        z3_nc = z3.simplify(z3_nc)

    print 'f1', z3_pc, 'f2', z3_nc
    
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
            logger.warn("inconsistent ? {}".format(str_of_pncore((pc,nc))))
            
        

    
    
is_pncore = lambda (pc,nc):  is_cdcore(pc) and is_cdcore(nc)

def str_of_pncore((pc,nc)):
    return "p: {}, n: {}".format(str_of_cdcore(pc),str_of_cdcore(nc))


def get_inv((p1,p2,n1,n2),configs,dom):
    """
    Return a *real* inv/overapprox over configs
    inv = (conj,disj)
    """
    #pos traces => p1 & not(n2)
    if p1: #x & y 
        if not all(c_implies(c,p1) for c in configs):
            p1 = None

    if n2: # y & z => not(y) | not(z)
        n2 = neg_of_core(n2,core,dom)
        if not all(d_implies(c,n2) for c in configs):
            n2 = None
    
    #neg traces => n1 & not(p2)
    #not(n1) || p2
    if n1 is None and p2 is None:
        pass
    elif n1 is not None and p2 is None:
        n1 = neg_of_core(n1,core,dom)
        if not all(d_implies(c,n1) for c in configs):
            n1 = None
    elif n1 is None and p2 is not None:
        if not all(c_implies(c,p2) for c in configs):
            p2 = None
    else:
        n1 = neg_of_core(n1,core,dom)
        if not all(d_implies(c,n1) or c_implies(c,p2) 
                   for c in configs):
            n1 = None
            p2 = None



## RESULTS ##
is_cores_d = lambda d: (isinstance(d,dict) and
                        (all(isinstance(sid,str) and is_pncore(c)
                             for sid,c in d.iteritems())))
                            
def str_of_cores_d(cores_d):
    if vdebug:
        assert is_cores_d(cores_d),cores_d
        
    return '\n'.join("{}. {}: {}"
                     .format(i+1,sid,str_of_pncore(cores_d[sid]))
                     for i,sid in enumerate(sorted(cores_d)))

def merge_cores_d(cores_d):
    if vdebug:
        assert is_cores_d(cores_d), cores_d
        
    mcores_d = OrderedDict()
    for sid,pncore in cores_d.iteritems():
        
        if pncore in mcores_d:
            mcores_d[pncore].add(sid)
        else:
            mcores_d[pncore] = set([sid])

    for sid in mcores_d:
        mcores_d[sid]=frozenset(mcores_d[sid])
        
    if vdebug:
        assert is_mcores_d(mcores_d), mcores_d
        
    return mcores_d


is_mcores_d = lambda d: (isinstance(d,dict) and
                         all(is_pncore(c) and is_cov(cov)
                             for c,cov in d.iteritems()))

def str_of_mcores_d(mcores_d):
    if vdebug:
        assert is_mcores_d(mcores_d),mcores_d

    mcores_d_ = sorted(mcores_d.iteritems(),
                       key=lambda (c,cov):(strength_pncore(c),len(cov)))

    return '\n'.join("{}. ({}) {}: {}"
                     .format(i+1,
                             strength_pncore(pncore),
                             str_of_pncore(pncore),
                             str_of_cov(cov))
                     for i,(pncore,cov) in enumerate(mcores_d_))

def mcores_d_strengths(mcores_d):
    if vdebug:
        assert is_mcores_d(mcores_d),mcores_d
    
    res = []
    sizs = [strength_pncore(c) for c in mcores_d]

    for siz in sorted(set(sizs)):
        siz_conds = [c for c in mcores_d if strength_pncore(c) == siz]
        cov = set()
        for c in siz_conds:
            for sid in mcores_d[c]:
                cov.add(sid)
        res.append((siz,len(siz_conds),len(cov)))
                  
    return res

def str_of_mcores_d_strengths(mstrengths):
    if isinstance(mstrengths,dict):
        mstrengths = mcores_d_strengths(mstrengths)
    
    ss = []
    for siz,ninters,ncovs in mstrengths:
        ss.append("({},{},{})".
                  format(siz,ninters,ncovs))
    return ','.join(ss)
        
# def strength_core(c):
#     if vdebug:
#         assert is_core(c),c
#     return len(c) if c else 0

#strength_pncore = lambda (cores): sum(map(strength_core,cores))

def settings_of_pncore(((pc,pd),(nc,nd))):
    settings = []
    for c in [pc,pd,nc,nd]:
        if c:
            settings.extend(c.hcontent)
    settings = list(set(settings))
    return settings

strength_pncore = lambda pncore: len(settings_of_pncore(pncore))


def post_process(mcores_d,dom):
    if vdebug:
        assert is_dom(dom)

    dom_z3 = z3_of_dom(dom)
    for pncore,sids in mcores_d.iteritems():
        pass
    
### MISCS UTILS ###


def load_dir(dirname):
    iobj = CM.vload(os.path.join(dirname,'info'))
    rfiles = [os.path.join(dirname,f) for f in os.listdir(dirname)
              if f.endswith('.tvn')]
    robjs = [CM.vload(rfile) for rfile in rfiles]
    return iobj, robjs

def print_iter_stat(robj):
    (citer,etime,ctime,samples,covs,new_covs,new_cores,sel_core,cores_d) = robj
    logger.info("iter {}, ".format(citer) +
                "{0:.2f}s, ".format(etime) +
                "{0:.2f}s eval, ".format(ctime) +
                "new: {} samples, {} covs, {} cores, "
                .format(len(samples),len(new_covs),len(new_cores)) +
                "{}".format("** progress **" if new_covs or new_cores else ""))
                
    logger.debug('sel_core\n{}'.format(str_of_pncore(sel_core)))
    logger.debug('samples\n'+str_of_configs(samples,covs))
    mcores_d = merge_cores_d(cores_d)
    logger.debug('mcores\n{}'.format(str_of_mcores_d(mcores_d)))
    mstrengths = mcores_d_strengths(mcores_d)
    logger.info(str_of_mcores_d_strengths(mstrengths))
    return mcores_d,mstrengths

def str_of_summary(seed,iters,ntime,ctime,nsamples,ncovs,tmpdir):
    s = ("Summary: Seed {}, Iters {}, Time ({}, {}), Samples {}, Covs {}, Tmpdir '{}'"
         .format(seed,iters,ntime,ctime,nsamples,ncovs,tmpdir))
    return s
    
def replay(dirname):
    iobj,robjs = load_dir(dirname)
    seed,dom = iobj
    logger.info('seed: {}'.format(seed))
    logger.debug(str_of_dom(dom))

    ntime = 0.0
    nsamples = 0
    ncovs = 0
    robjs = sorted(robjs,key=lambda robj: robj[0])
    for robj in robjs:
        if len(robj) == 9:
            (citer,etime,ctime,samples,covs,
             new_covs,new_cores,sel_core,cores_d) = robj
        else: #old format
            (citer,etime,samples,covs,
             new_covs,new_cores,sel_core,cores_d) = robj
            ctime = 0.0
            robj = (citer,etime,ctime,samples,covs,
                    new_covs,new_cores,sel_core,cores_d)

        mcores_d,mstrengths = print_iter_stat(robj)
        ntime+=etime
        nsamples+=len(samples)
        ncovs+=len(new_covs)
        #print mstrengths
        # minint = mstrengths[0][0]
        # maxint = mstrengths[-1][0]
        # tt = [citer+1,nsamples,ncovs,len(mcores_d),"{} ({},{})".format(len(mstrengths),minint,maxint)]
        #print "{}\\\\".format(" & ".join(map(str,tt)))
        
    niters = len(robjs)
    logger.info(str_of_summary(seed,niters,
                               ntime,ctime,
                               nsamples,ncovs,dirname))

    return niters,ntime,ctime,nsamples,ncovs,mcores_d,mstrengths

def mk_tcover(vals,cover_siz,tseed,tmpdir):
        """
        Call external program to generate t-covering arrays

        sage: mk_tcover([[1,2],[1,2,3],[4,5]], 2, 0,'/tmp/')
        cmd: cover -r 0 -s 0 -o  /tmp/casaOutput.txt /tmp/casaInput.txt
        [[1, 1, 5], [2, 1, 4], [2, 2, 5], [1, 2, 4], [2, 3, 4], [1, 3, 5]]

        #linux gives this
        [[2, 3, 4], [1, 3, 5], [1, 1, 4], [2, 1, 5], [2, 2, 4], [1, 2, 5]]
        """
        if vdebug:
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

### inference ###
def infer(configs,old_core,dom):
    """
    Overapproximation in *conjunctive* form
    """
    if vdebug:
        assert is_core(old_core),old_core
        assert is_configs(configs),configs
        assert is_dom(dom),dom

    if not configs:
        return old_core

    if old_core is None:  #not yet set
        old_core = min(configs,key=lambda c:len(c))
        old_core = HDict([(x,frozenset([y]))
                          for x,y in old_core.iteritems()])

    def f(x,s,ldx):
        s_ = set(s)
        for config in configs:
            if x in config:
                s_.add(config[x])
                if len(s_) == ldx:
                    return None
            else:
                return None
        return s_

    settings = []
    for x,ys in old_core.iteritems():
        ys_ = f(x,ys,len(dom[x]))
        if ys_:
            settings.append((x,frozenset(ys_)))
            
    core = HDict(settings)
    
    if vdebug:
        assert is_core(core), core

    return core    

def infer_cache(old_core,configs,dom,cache):
    if vdebug:
        assert is_core(old_core),old_core
        assert is_configs(configs),configs
        assert is_dom(dom),dom
    
    configs = frozenset(configs)
    key = (configs,old_core)
    if key in cache:
        core = cache[key]
    else:
        configs = list(configs)
        core = infer(configs,old_core,dom)
        cache[key] = core
        
    return core

#x=1&y=0&z=0 => x={1,2}&y={0,1}
c_implies = lambda config,core: all(config[k] in core[k] for k in core)
#x=1&y=1&z=0 => x={1}|y={3,1}
d_implies = lambda config,core: any(config[k] in core[k] for k in core)
def neg_of_core(core,dom): 
    if not core:
        return core
    else:
        return HDict([(k,dom[k]-core[k]) for k in core])

def infer_partition((old_pc,old_nc),
                    pconfigs,nconfigs,all_pconfigs,
                    dom,cache):
    if vdebug:
        assert is_core(old_pc),old_pc
        assert is_core(old_nc),old_pc
        assert is_configs(pconfigs),pconfigs
        assert is_configs(nconfigs),nconfigs
        assert is_configs(all_pconfigs),nconfigs
        assert is_dom(dom),dom
    
    pc = infer_cache(old_pc,pconfigs,dom,cache)
    if pc != old_pc and old_pc:
        nconfigs = [c for c in nconfigs if not c_implies(c,old_pc)]

    if pc:
        nconfigs = [c for c in nconfigs if c_implies(c,pc)]

    nc = infer_cache(old_nc,nconfigs,dom,cache)
    if pc and nc:
        #simplify
        nc = HDict([(k,vs) for k,vs in nc.iteritems()
                    if (k,vs) not in pc.hcontent])
    
    return pc,nc

def infer_covs(configs,covs,old_cores_d,dom,configs_cache):
    """
    old_cores_d = {sid : ((pcore1,pcore2),(ncore1,ncore2))}
    """
    if vdebug:
        assert is_configs(configs), configs
        assert all(is_cov(cov) for cov in covs), covs        
        assert is_cores_d(old_cores_d), old_cores_d
        assert is_dom(dom), dom
        
    new_covs,new_cores = set(),set()  #updated stuff
    
    if not configs:
        return new_covs,new_cores

    sids = set(old_cores_d.keys())
    for cov in covs:
        for sid in cov:
            sids.add(sid)
    
    cache = {}
    for sid in sorted(sids):
        if sid in old_cores_d:
            old_pcdc,old_ncdc = old_cores_d[sid]
        else:
            old_pcdc,old_ncdc = ((None, None), (None, None))
            new_covs.add(sid)

        pconfigs,nconfigs = [],[]
        for c,cov in zip(configs,covs):
            if sid in cov:
                pconfigs.append(c)
            else:
                nconfigs.append(c)

        all_pconfigs,all_nconfigs=[],[]
        for c,cov in configs_cache.iteritems():
            if sid in cov:
                all_pconfigs.append(c)
            else:
                all_nconfigs.append(c)
        
        new_pcdc = infer_partition(
            old_pcdc,pconfigs,nconfigs,all_pconfigs,dom,cache)

        new_ncdc = infer_partition(
            old_ncdc,nconfigs,pconfigs,all_nconfigs,dom,cache)
            
        old_pcdc = old_pcdc,old_ncdc
        new_pcdc = new_pcdc,new_ncdc

        if not old_pcdc == new_pcdc: #progress
            new_cores.add(sid)
            old_cores_d[sid] = new_pcdc
            
    return new_covs,new_cores

### generate configs

def gen_configs_full(dom):
    if vdebug:
        assert is_dom(dom), dom
        
    ns,vs = itertools.izip(*dom.iteritems())
    configs = [HDict(zip(ns,c)) for c in itertools.product(*vs)]

    if vdebug:
        assert is_configs(configs),configs
        
    return configs

def gen_configs_rand(n,dom):
    if vdebug:
        assert n > 0,n
        assert is_dom(dom),dom

    rgen = lambda : [(k,random.choice(list(vs))) for k,vs in dom.iteritems()]
    configs =  list(set(HDict(rgen()) for _ in range(n)))
    
    if vdebug:
        assert is_configs(configs),configs
        
    return configs

def gen_configs_tcover1(dom):
    """
    Return a set of tcover array of strength 1
    """
    if vdebug:
        assert is_dom(dom), dom
        
    dom_used = dict((k,set(vs)) for k,vs in dom.iteritems())

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

    if vdebug:
        assert is_configs(configs),configs
        
    return configs

def gen_configs_tcover(cover_siz,dom,tseed,tmpdir):
    vals = map(list,dom.values())
    vs = mk_tcover(vals,cover_siz,tseed,tmpdir)
    configs = [HDict(zip(dom.keys(),vs_)) for vs_ in vs]

    if vdebug:
        assert is_configs(configs),configs
        
    return configs
        
def gen_configs_core(n,core,dom):
    """
    create n configs by changing n settings in core
    """
    if vdebug:
        assert n > 0, n
        assert is_core(core), core
        assert is_dom(dom),dom

    if not core:
        return gen_configs_rand(n,dom)
    
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
    

def gen_configs_cores(pncore,dom):
    if vdebug:
        assert is_pncore(pncore), pncore
        assert is_dom(dom), dom

    pc1,_,nc1,_ = pncore        

    configs_p = []
    strength_pc1 = strength_core(pc1)
    if strength_pc1 > 0:
        configs_p = gen_configs_core(strength_pc1,pc1,dom)

    configs_n = []
    strength_nc1 = strength_core(nc1)
    if strength_nc1 > 0:
        configs_n = gen_configs_core(strength_nc1,nc1,dom)

    configs = list(set(configs_p + configs_n))
    
    if vdebug:
        assert is_configs(configs),configs
        
    return configs

### iterative refinement alg
min_sel_stren=0
def select_core(cores,ignore_sizs,ignore_cores):
    if vdebug:
        assert isinstance(ignore_sizs,set),ignore_sizs
        assert isinstance(ignore_cores,set),ignore_cores        

    cores = [core for core in cores if core not in ignore_cores]
    core_strengths = [strength_pncore(pncore) for pncore in cores]
    sizs = set(core_strengths) - ignore_sizs
    sizs = [stren for stren in sizs if stren >= min_sel_stren]
    if sizs:
        siz = max(sizs)
        cores_siz = [core for core,strength in zip(cores,core_strengths)
                     if strength==siz]

        core = max(cores_siz,key=lambda c:strength_pncore(c))
        return core  #tuple (core_l,dcore)
    else:
        return None

def eval_samples(samples,get_cov,cache):
    if vdebug:
        assert is_configs(samples),samples
        assert isinstance(cache,dict),cache

    st = time()
    samples_ = []
    for config in samples:
        if config not in cache:
            cov = get_cov(config)
            cache[config]=frozenset(cov)
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
    if vdebug:
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
    cores_d = OrderedDict()  #results {sid: (core_l,dcore)}
    cur_iter = 0
    max_stuck = 2
    cur_stuck = 0
    ignore_sizs = set()
    ignore_cores = set()
    configs_cache = OrderedDict()
    sel_core = ((None,None),(None,None))
  
    #begin
    st = time()
    ct = st
    cov_time = 0.0
    
    #initial samples
    if cover_siz:
        cstrength,max_confs = cover_siz
        if cstrength == 0:
            if max_confs < 0:
                samples = gen_configs_full(dom)
                logger.info("Gen all {} configs".format(len(samples)))
            else:
                samples = gen_configs_rand(max_confs,dom)
                logger.info("Gen {} rand configs".format(len(samples)))
        else:
            samples = gen_configs_tcover(cstrength,dom,seed,tmpdir)
            samples_rand_n = max_confs - len(samples)

            if samples_rand_n:
                samples_ = gen_configs_rand(samples_rand_n,dom)
                samples.extend(samples_)

            samples = list(set(samples))
            logger.info("Gen {} {}-cover configs"
                        .format(len(samples),cstrength))
                                
    else:
        samples = gen_configs_tcover1(dom)

    if is_config(config_default):
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
        CM.pause()
        if cover_siz:
            break
        
        cur_iter += 1
        sel_core = select_core(set(cores_d.values()),
                               ignore_sizs,ignore_cores)
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
            ignore_sizs.clear()
            
        else: #no progress
            cur_stuck += 1
            if cur_stuck >= max_stuck:
                #ignore_sizs.add(strength_pncore(sel_core))
                cur_stuck = 0

    #final refinement
    logger.info(str_of_summary(seed,cur_iter,time()-st,cov_time,
                               len(configs_cache),len(cores_d),
                               tmpdir))
    return cores_d


#Shortcuts
def intgen_full(dom,get_cov,tmpdir=None,prefix='vu'):
    return intgen(dom,get_cov,seed=None,tmpdir=tmpdir,
                 cover_siz=(0,-1),config_default=None,
                 prefix=prefix)

def intgen_pure_rand(dom,get_cov,rand_n,seed=None,tmpdir=None,prefix='vu'):
    return intgen(dom,get_cov,seed=seed,tmpdir=tmpdir,
                 cover_siz=(0,rand_n),config_default=None,
                 prefix=prefix)

### Post processing ###
# def mimplies(x,y):
#     """
#     None: False,  empty: True
#     """
#     if x is None or y is None:
#         return None
#     else:
#         return x.hcontent.issuperset(y.hcontent)

# def igraph(mcores_d):
#     g = {}
#     for pncore in mcores_d:
#         g[pncore]=[]
#         for pncore_ in mcores_d:
#             if pncore != pncore_:
#                 ccore,dcore = pncore
#                 ccore_,dcore_ = pncore_
#                 x = mimplies(ccore,ccore_)
#                 y = mimplies(dcore,dcore_)
#                 if x is None and y is None:
#                     r = False
#                 elif x is None:
#                     r = False
#                 elif y is None:
#                     r = False
#                 else:
#                     r = x and y
#                 if r:
#                     print "1. {} => {}".format(str_of_pncore(pncore),str_of_pncore(pncore_))
#                     print ccore, ccore_
#                     print "2. {} => {}".format(str_of_core(ccore), str_of_core(ccore_))
#                     print dcore,dcore_
#                     print "3. {} => {}".format(str_of_core(dcore), str_of_core(dcore_))
#                     print ""
#                     CM.pause()
#                     g[pncore].append(pncore_)
#     return g
    
# def str_of_igraph(g):
#     ss = []
#     for pncore,pncores in g.iteritems():
#         ss.append("{}: ({}) {}"
#                   .format(str_of_pncore(pncore),
#                           len(pncores),
#                           ', '.join(str_of_pncore(c) for c in pncores)))
#     return '\n'.join(ss)

### Experiments ###

def benchmark_stats(results_dir,strength_thres=100000000):
    niters_total = 0
    ntime_total = 0
    nctime_total = 0    
    nsamples_total = 0
    ncovs_total = 0
    nruns_total = 0
    mcores_d_s = []
    mstrengths_s = []
    for rdir in os.listdir(results_dir):
        rdir = os.path.join(results_dir,rdir)
        niters,ntime,ctime,nsamples,ncovs,mcores_d,mstrengths = replay(rdir)
        niters_total += niters
        ntime_total += ntime
        nctime_total += (ntime - ctime)
        nsamples_total += nsamples
        ncovs_total += ncovs
        mcores_d_s.append(mcores_d)
        mstrengths_s.append(mstrengths)
        nruns_total += 1

    nruns_total = float(nruns_total)
    logger.info("avg: " + 
                "iter {} ".format(niters_total/nruns_total) +
                "time {} ".format(ntime_total/nruns_total) +
                "xtime {} ".format(nctime_total/nruns_total) +                
                "samples {} ".format(nsamples_total/nruns_total) +
                "covs {} ".format(ncovs_total/nruns_total)
                )

    sres = {}
    for mstrengths in mstrengths_s:
        for strength,ninters,ncov in mstrengths:
            if strength >= strength_thres:
                strength = strength_thres
                
            if strength not in sres:
                sres[strength] = ([ninters],[ncov])
            else:
                inters,covs = sres[strength]
                inters.append(ninters)
                covs.append(ncov)


    for strength in sorted(sres):
        inters,covs = sres[strength]
        logger.info("({},{},{})"
                    .format(strength,
                            sum(inters)/float(len(mstrengths_s)),
                            sum(covs)/float(len(mstrengths_s))))
######


def prepare_motiv(dom_file,prog_file):
    import ex_motiv_run
        
    dom,_ = get_dom(dom_file)
    prog =  os.path.realpath(prog_file)
    logger.info("dom_file: '{}', ".format(dom_file) + 
                "prog_file: '{}'".format(prog_file))
    args = {'varnames':dom.keys(),
            'prog': prog}
    get_cov = lambda config: ex_motiv_run.get_cov(config,args)
    return dom,get_cov

def prepare_otter(prog):
    dir_ = '~/Src/Devel/iTree_stuff/expData/{}'.format(prog)
    dir_ = os.path.realpath(os.path.expanduser(dir_))
    dom_file = dir_ + '/possibleValues.txt'
    exp_dir = dir_ + '/rawExecutionPaths'.format(prog)
    pathconds_d_file = dir_ + '/pathconds_d.tvn'.format(prog)

    import ex_otter
    
    dom,_ = get_dom(dom_file)
    if os.path.isfile(pathconds_d_file):
        logger.info("read from '{}'".format(pathconds_d_file))
        pathconds_d = CM.vload(pathconds_d_file)
    else:
        logger.info("read from '{}'".format(exp_dir))
        pathconds_d = ex_otter.get_data(exp_dir)

    logger.info("read {} pathconds".format(len(pathconds_d)))
    args = {'pathconds_d':pathconds_d}
    get_cov = lambda config: ex_otter.get_cov(config,args)
    
    return dom,get_cov,pathconds_d

    
def run_gt(dom,pathconds_d):
    """
    Obtain conditions using Otter's pathconds
    """
    allsamples = []
    allcovs = []
    for covs,samples in pathconds_d.itervalues():
        for sample in samples:
            allsamples.append(HDict(sample))
            allcovs.append(covs)

    logger.info("infer conds using {} samples".format(len(allsamples)))
    st = time()
    cores_d = {}
    infer_cov(allsamples,allcovs,cores_d,dom)
    logger.info("infer conds for {} covered lines ({}s)"
                .format(len(cores_d),time()-st))
    return cores_d


# def test_motiv(dom,get_cov):
#     #listen time ssl local anon log chunks
#     #0 1 0 0 1 1 2
#     old_cores_d = {}        
#     c1 = HDict(zip(dom,'1 0 1 1 1 0 2'.split()))
#     c2 = HDict(zip(dom,'1 0 0 0 0 1 4'.split()))
#     c3 = HDict(zip(dom,'0 1 0 1 0 1 3'.split()))
#     c4 = HDict(zip(dom,'1 0 1 1 0 0 1'.split()))
#     configs = [c1,c2,c3,c4]
#     covs = [get_cov(config) for config in configs]
#     print(str_of_configs(configs,covs))    
#     new_covs,new_cores = infer_cov(configs,covs,old_cores_d,dom)
#     print(str_of_cores_d(old_cores_d))    
#     mcores_d = merge_cores_d(old_cores_d)
#     print(str_of_mcores_d(mcores_d))
#     CM.pause()
    
#     c5 = HDict(zip(dom,'0 0 0 0 1 1 3'.split()))
#     c6 = HDict(zip(dom,'0 1 1 1 0 1 4'.split()))
#     c7 = HDict(zip(dom,'0 1 0 0 1 1 2'.split()))
#     c8 = HDict(zip(dom,'1 0 1 1 1 0 3'.split()))

#     configs = [c5,c6,c7,c8]
#     covs = [get_cov(config) for config in configs]
#     print(str_of_configs(configs,covs))
#     new_covs,new_cores = infer_cov(configs,covs,old_cores_d,dom)
#     print(str_of_cores_d(old_cores_d))
#     mcores_d = merge_cores_d(old_cores_d)
#     print(str_of_mcores_d(mcores_d))

#     return old_cores_d

"""
Evaluate results:

Are inferred conditions precise ? 
-  If have ground truths, compare against ground truths  ...  see how many are wrong
-  If does not have ground truths, generate semi random configs to test 


How many configs generated to obtain such precision ? 
- Compare against randomness, t-covering





efficiency
motive:
(0,1,1),(2,5,5),(3,1,1),(5,2,2)

vsftpd:
gt
covs 2549, configs   (0,1,336),(1,3,73),(2,9,199),(3,4,1391),(4,16,403),(5,5,102),(6,6,35),(7,2,10)


                       
mine: (init sample 1)
covs 2549, configs 475 (0,1,336),(1,3,73),(2,9,199),(3,4,1385),(4,17,409),(5,5,102),(6,6,35),(7,2,10)
covs 2549, configs 363 (0,1,336),(1,3,73),(2,9,199),(3,4,1385),(4,17,409),(5,5,102),(6,6,35),(7,2,10)
configs 361 (0,1,336),(1,3,73),(2,9,199),(3,4,1385),(4,17,409),(5,5,102),(6,6,35),(7,2,10)
configs 326 (0,1,336),(1,3,73),(2,9,199),(3,4,1385),(4,17,409),(5,5,102),(6,6,35),(7,2,10)
configs 513 (0,1,336),(1,3,73),(2,9,199),(3,4,1385),(4,17,409),(5,5,102),(6,6,35),(7,2,10)

rd200:

rd400:
cov 2549, configs 400,  (0,1,336),(1,3,73),(2,9,199),(3,4,1385),(4,17,409),(5,4,98),(6,3,25),(8,1,2),(10,2,18),(14,1,1),(16,1,2),(30,1,1) 
(0,1,336),(1,3,73),(2,9,199),(3,4,1385),(4,17,409),(5,3,96),(6,4,10),(7,1,2),(8,1,19),(9,1,9),(12,1,1),(13,1,9),(30,1,1)


rd1000:
'(0,1,336),(1,3,73),(2,9,199),(3,4,1385),(4,17,409),(5,5,102),(6,5,34),(7,2,10),(9,1,1)'


"""



"""
Configurable influential variables ...  
Prev works do not try 

!listen !timeout ssl !local anon !log chunks=4 3 4 9



ngircd:
gt:
      
3090 (0,2,638),(1,4,580),(2,6,537),(3,16,829),(4,7,441),(5,3,63),(6,1,2)

mine 270 '(0,1,748),(1,3,11),(2,9,994),(3,16,829),(4,9,443),(5,3,63),(6,1,2)'
mine 305 '(0,1,748),(1,3,11),(2,9,994),(3,16,829),(4,9,443),(5,3,63),(6,1,2)'



432. line 'str.c:499'
core me p(tunable_anonymous_enable=1 tunable_local_enable=0 tunable_ssl_enable=0)
core gt p(tunable_anonymous_enable=1 tunable_local_enable=0 tunable_ssl_enable=0 tunable_tilde_user_enable=0 )
"""


def doctestme():
    import doctest
    doctest.testmod()



if __name__ == "__main__":
    print 'loaded'
    doctestme()





# def is_formula(f):
#     """
#     ('or',[('x','1'),('y','0')])
    
#     """
#     if is_csetting(f):
#         return True
#     else:
#         return (isinstance(f,tuple) and len(f)==2 and 
#                 isinstance(f[0],str) and 
#                 isinstance(f[1],list) and 
#                 all(is_formula(c) for c in f[1]))



# def z3_of_config(c,z3dom):
#     """
#     return a z3 formula
#     >>> z3_of_config(HDict(),{})
#     """
#     if vdebug:
#         assert is_config(c),c

#     f = []
#     for vn,vv in c.iteritems():
#         vn_,vs_ = z3dom[vn]
#         f.append(vn_==vs_[vv])

#     return z3util.myAnd(f)
    


    



# def str_of_cdcore(cc,dc):
#     conj = lambda: z3_of_core(ccore)
#     disj = lambda: z3.Not(z3_of_core(dcore))

#     if ccore is None and dcore is None:
#         return None
#     elif ccore is None and dcore is not None:
#         return disj() if dcore else z3util.TRUE
#     elif ccore is not None and dcore is None:
#         return conj() if ccore else z3util.TRUE
#     else:  #ccore is not None and dcore is not None:
#         if ccore and dcore:
#             return z3util.myAnd(conj(),disj())
#         elif ccore and not dcore:
#             return conj()
#         elif not ccore and dcore:
#             return disj()
#         else:    
# def str_of_pncore():

#     f1 = str_of_cdcore((p1,p2))



# def str_of_cdcore((ccore,dcore)):
#     if vdebug:
#         assert is_cdcore((ccore,dcore)), (ccore,dcore)

#     return 'p({}), n({})'.format(str_of_core(ccore),str_of_core(dcore))



# is_cdcores = lambda cs: (isinstance(cs,list) and
#                          all(is_cdcore(c) for c in cs))
