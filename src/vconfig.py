import tempfile
import random
import os.path
import itertools
from collections import OrderedDict
from time import time

import z3
import vu_common as CM

from vconfig_miscs import (Dom, Config, Configs,
                           Core, PCore, NCore, PNCore, CORES_D, MCORES_D,
                           is_cov)
import vconfig_miscs

vconfig_miscs.print_cov = False
logger = CM.VLog('config')
logger.level = CM.VLog.DEBUG
CM.VLog.PRINT_TIME = True


### MISCS UTILS ###


def load_dir(dirname):
    iobj = CM.vload(os.path.join(dirname,'info'))
    rfiles = [os.path.join(dirname,f) for f in os.listdir(dirname)
              if f.endswith('.tvn')]
    robjs = [CM.vload(rfile) for rfile in rfiles]
    return iobj, robjs

def print_iter_stat(robj):
    (citer,etime,ctime,samples,covs,new_covs,new_cores,sel_core,cores_d) = robj
    print ''
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
    logger.debug('samples\n'+samples.__str__(covs))
    mcores_d = cores_d.merge()
    logger.debug('mcores\n{}'.format(mcores_d))
    logger.info(mcores_d.strength_str)
    return mcores_d

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

        mcores_d = print_iter_stat(robj)
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

    return niters,ntime,ctime,nsamples,ncovs,mcores_d

def mk_tcover(vals,cover_siz,tseed,tmpdir):
        """
        Call external program to generate t-covering arrays

        sage: mk_tcover([[1,2],[1,2,3],[4,5]], 2, 0,'/tmp/')
        cmd: cover -r 0 -s 0 -o  /tmp/casaOutput.txt /tmp/casaInput.txt
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

### inference ###
def infer(configs,old_core,dom):
    """
    Overapproximation in *conjunctive* form
    """
    if CM.__vdebug__:
        assert old_core is None or isinstance(old_core,Core),old_core
        assert isinstance(configs,Configs),configs
        assert isinstance(dom,Dom),dom

    if not configs:
        return old_core

    if old_core is None:  #not yet set
        old_core = min(configs,key=lambda c:len(c))
        old_core = Core([(x,frozenset([y]))
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
            
    core = Core(settings)
    return core    

def infer_cache(old_core,configs,dom,cache):
    if CM.__vdebug__:
        assert old_core is None or isinstance(old_core,Core),old_core
        assert isinstance(configs,Configs),configs
        assert isinstance(dom,Dom),dom
    
    configs = frozenset(configs)
    key = (configs,old_core)
    if key in cache:
        core = cache[key]
    else:
        configs = Configs(configs)
        core = infer(configs,old_core,dom)
        cache[key] = core
        
    return core


def infer_partition(old_c,pconfigs,all_nconfigs,dom,cache):
                    
    if CM.__vdebug__:
        assert isinstance(old_c,PCore) or isinstance(old_c,NCore),old_c
        assert isinstance(pconfigs,Configs),pconfigs
        assert isinstance(all_nconfigs,Configs),all_nconfigs
        assert isinstance(dom,Dom),dom

    cc = infer_cache(old_c.conj,pconfigs,dom,cache)
    if cc:
        all_nconfigs = Configs(c for c in all_nconfigs if c.c_implies(cc))

    dc = infer_cache(old_c.disj,all_nconfigs,dom,cache)

    if cc and dc:
        #simplify
        dc = Core(e for e in dc.iteritems() if e not in cc.hcontent)

    return old_c.__class__((cc,dc))

def infer_covs(configs,covs,old_cores_d,dom,configs_cache):
    if CM.__vdebug__:
        assert isinstance(configs,Configs), configs
        assert all(is_cov(cov) for cov in covs), covs        
        assert isinstance(old_cores_d,CORES_D), old_cores_d
        assert isinstance(dom,Dom), dom

    new_covs,new_cores = set(),set()  #updated stuff
    
    if not configs:
        return new_covs,new_cores

    sids = set(old_cores_d.keys())
    for cov in covs:
        for sid in cov:
            sids.add(sid)

    logger.debug("infer {} cov".format(len(sids)))
    
    cache = {}
    for i,sid in enumerate(sorted(sids)):
        pconfigs,nconfigs = Configs(),Configs()
        for c,cov in zip(configs,covs):
            if sid in cov:
                pconfigs.append(c)
            else:
                nconfigs.append(c)

        all_pconfigs,all_nconfigs=Configs(),Configs()
        for c,cov in configs_cache.iteritems():
            if sid in cov:
                all_pconfigs.append(c)
            else:
                all_nconfigs.append(c)
        
        if sid in old_cores_d:
            old_c = old_cores_d[sid]
        else:
            old_c = PNCore.mk_default()
            new_covs.add(sid)

        new_pc = infer_partition(
            old_c.pcore,pconfigs,all_nconfigs,dom,cache)

        new_nc = infer_partition(
            old_c.ncore,nconfigs,all_pconfigs,dom,cache)
            
        new_c = PNCore((new_pc,new_nc))

        if not old_c == new_c: #progress
            new_cores.add(sid)
            old_cores_d[sid] = new_c

        print "{}/{}, sid {}".format(i,len(sids),sid)
        print 'all_pconfigs', len(all_pconfigs)
        print new_pc
        
        print 'all_nconfigs', len(all_nconfigs)
        print new_nc

        

    return new_covs,new_cores

        
def gen_configs_core(n,core,dom):
    """
    create n configs by changing n settings in core
    """
    if CM.__vdebug__:
        assert n > 0, n
        assert isinstance(core,Core) and core, core
        assert isinstance(dom,Dom),dom

    
    vnames = random.sample(core.keys(),n)
    vvals = [dom[x]-core[x] for x in vnames]
    
    changes = []
    for vname,vval in zip(vnames,vvals):
        for v in vval:
            changes.append((vname,v))

    configs = Configs()
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

        config = Config(settings)
        configs.append(config)
    return configs
    

def gen_configs_cores(core,dom):
    if CM.__vdebug__:
        assert isinstance(core,PNCore), core
        assert isinstance(dom,Dom), dom
    pc = core.pcore.conj
    pd = core.pcore.disj
    nc = core.ncore.conj
    nd = core.ncore.disj
    cores = set(c for c in [pc,pd,nc,nd] if c)
    print '\n'.join(map(str,cores))
        
    configs = []        
    for c in cores:
        configs.extend(gen_configs_core(len(c),c,dom))
        
    configs = Configs.uniq(configs)
    return configs

### iterative refinement alg
def core_c_implies(c1,c2):
    """
    c1 => c2
    #x&y&z => x&y
    >>> assert core_c_implies(dict(zip(['x','y','z'],[frozenset(['1']),frozenset(['1']),frozenset(['1'])])),dict(zip(['x','y'],[frozenset(['1']),frozenset(['1'])])))

    #x&y&z=1|2 => x,z=1|2|3
    >>> assert core_c_implies(dict(zip(['x','y','z'],[frozenset(['1']),frozenset(['1','2']),frozenset(['1'])])),dict(zip(['x','y'],[frozenset(['1']),frozenset(['1','2','3'])])))

    #x&y&z=1|2 => x,z=1|2
    >>> assert core_c_implies(dict(zip(['x','y','z'],[frozenset(['1']),frozenset(['1','2','3']),frozenset(['1'])])),dict(zip(['x','y'],[frozenset(['1']),frozenset(['1','2','3'])])))
    
    #not(x&y&z=1|2 => x,z=1)
    >>> c1 = dict(zip(['x','y','z'],[frozenset(['1']),frozenset(['1','2','3']),frozenset(['1'])]))
    >>> c2 = dict(zip(['x','y'],[frozenset(['1']),frozenset(['1'])]))
    >>> assert not core_c_implies(c1,c2)
    >>> assert not core_c_implies(c2,c1)

    """
    return all(k in c1 and c1[k].issubset(c2[k]) for k in c2)

       
def mycheck(pncore,configs,dom):
    """
    Check for correct rapprox over configs 
    """
    pc = pncore.pcore.conj
    pd = pncore.pcore.disj
    nc = pncore.ncore.conj
    nd = pncore.ncore.disj

    #pos traces => self.pcore.conj & self.pcore.disj.neg

    if pc:
        if not all(c.c_implies(pc) for c in configs):
            print 'pc invalid'
            pc = None

    if pd:
        pd_neg = pd.neg(dom)
        if not all(c.d_implies(pd_neg) for c in configs):
            print 'pd invalid'
            pd = None

    #neg traces => self.ncore.conj & self.ncore.disj.neg
    #pos traces => self.ncore.conj.neg | self.ncore.disj
    if nc is not None and nd is None:
        nc_neg = nc.neg(dom)
        if not all(c.d_implies(nc_neg) for c in configs):
            print 'nc invalid'
            nc = None
    elif nc is None and nd is not None:
        if not all(c.c_implies(nd) for c in configs):
            print 'nd invalid'
            nd = None
    elif nc is not None and nd is not None:
        nc_neg = nc.neg(dom)
        if not all(c.d_implies(nc_neg) or c.c_implies(nd)
                   for c in configs):
            print 'nc & nd invalid'
            nc = None
            nd = None

    return PNCore.mk(pc,pd,nc,nd)


def postprocess(cores_d,configs_cache,dom):
    if CM.__vdebug__:
        assert isinstance(cores_d,CORES_D), cores_d

    logger.debug("postprocessing")
    
    rs = []
    for sid,core in cores_d.iteritems():
        configs = [c for c,cov in configs_cache.iteritems()
                   if sid in cov]

        print sid
        print core
        print 'checking {}'.format(sid)
        core = mycheck(core,configs,dom)
        print core
        print 'simplify {}'.format(sid)        
        cdcore = core.z3simplify(dom)
        print 'final result {}'.format(cdcore)
        
        rs.append((sid,core))
        
    return CORES_D(rs)

min_sel_stren=0
def select_core(cores,ignore_sizs,ignore_cores):
    if CM.__vdebug__:
        assert isinstance(ignore_sizs,set),ignore_sizs
        assert isinstance(ignore_cores,set),ignore_cores        

    cores = [core for core in cores if core not in ignore_cores]
    core_strengths = [core.strength for core in cores]
    sizs = set(core_strengths) - ignore_sizs
    sizs = [stren for stren in sizs if stren >= min_sel_stren]
    if sizs:
        siz = max(sizs)
        cores_siz = [core for core,strength in zip(cores,core_strengths)
                     if strength==siz]

        core = max(cores_siz,key=lambda c:c.strength)
        return core  #tuple (core_l,dcore)
    else:
        return None

def eval_samples(samples,get_cov,cache):
    if CM.__vdebug__:
        assert isinstance(samples,Configs),samples
        assert isinstance(cache,dict),cache

    st = time()
    samples_ = Configs()
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
        assert isinstance(dom,Dom),dom
        assert callable(get_cov),get_cov
        assert (config_default is None or
                isinstance(config_default,Config)), config_default

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
    cores_d = CORES_D()  #results {sid: (core_l,dcore)}
    cur_iter = 0
    max_stuck = 2
    cur_stuck = 0
    ignore_sizs = set()
    ignore_cores = set()
    configs_cache = OrderedDict()
    sel_core = PNCore.mk_default()
  
    #begin
    st = time()
    ct = st
    cov_time = 0.0
    
    #initial samples
    if cover_siz:
        cstrength,max_confs = cover_siz
        if cstrength == 0:
            if max_confs < 0:
                samples = dom.gen_configs_full()
                logger.info("Gen all {} configs".format(len(samples)))
            else:
                samples = dom.gen_configs_rand(max_confs)
                logger.info("Gen {} rand configs".format(len(samples)))
        else:
            samples = dom.gen_configs_tcover(cstrength,seed,tmpdir)
            samples_rand_n = max_confs - len(samples)

            if samples_rand_n:
                samples_ = dom.gen_configs_rand(samples_rand_n)
                samples.extend(samples_)

            samples = Configs.uniq(samples)
            logger.info("Gen {} {}-cover configs"
                        .format(len(samples),cstrength))
                                
    else:
        samples = Dom.gen_configs_tcover1(dom)

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
                ignore_sizs.add(sel_core.strength)
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
        niters,ntime,ctime,nsamples,ncovs,mcores_d = replay(rdir)
        niters_total += niters
        ntime_total += ntime
        nctime_total += (ntime - ctime)
        nsamples_total += nsamples
        ncovs_total += ncovs
        mcores_d_s.append(mcores_d)
        mstrengths_s.append(mcores_d.strength_cts)
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
        
    dom,_ = Dom.get_dom(dom_file)
    prog =  os.path.realpath(prog_file)
    assert os.path.isfile(prog), prog
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
    
    dom,_ = Dom.get_dom(dom_file)
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
    allsamples = Configs()
    allcovs = []
    for covs,samples in pathconds_d.itervalues():
        for sample in samples:
            allsamples.append(Config(sample))
            allcovs.append(covs)

    logger.info("infer interactions using {} samples".format(len(allsamples)))
    st = time()
    cores_d = CORES_D()
    configs_cache = {}
    infer_covs(allsamples,allcovs,cores_d,dom,{})
    logger.info("infer conds for {} covered lines ({}s)"
                .format(len(cores_d),time()-st))
    return cores_d

def doctestme():
    import doctest
    doctest.testmod()



if __name__ == "__main__":
    print 'loaded'
    doctestme()


