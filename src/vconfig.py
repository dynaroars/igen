import tempfile
import random
import os.path
import itertools

from collections import OrderedDict
from time import time

from vconfig_miscs import HDict
import vu_common as CM

logger = CM.VLog('config')
logger.level = CM.VLog.DEBUG
CM.VLog.PRINT_TIME = True

vdebug = False
print_cov = False

### DATA STRUCTURES ###
is_cov = lambda cov: (isinstance(cov,frozenset) and
                    all(isinstance(sid,str) for sid in cov))
is_valset = lambda vs: (isinstance(vs,frozenset) and vs and
                        all(isinstance(v,str) for v in vs))

class Dom(OrderedDict):

    def __init__(self,d):
        if vdebug:
            assert all(isinstance(vn,str) and
                       is_valset(vs) for vn,vs in d.iteritems())

        OrderedDict.__init__(self,d)
    def __str__(self):
        return '\n'.join("{}: {}".format(k,str_of_valset(vs))
                         for k,vs in sorted(self.iteritems()))


    @staticmethod
    def get_dom(dom_file):
        
        def from_lines(lines):
            dom = OrderedDict()
            for line in lines:
                parts = line.split()
                varname = parts[0]
                varvals = parts[1:]
                dom[varname] = frozenset(varvals)
            return dom
        
        dom_file = os.path.realpath(dom_file)
        dom = Dom(from_lines(CM.iread_strip(dom_file)))

        default_config = None
        default_file = os.path.realpath(dom_file+'.default')
        if os.path.isfile(default_file):
            default_config = from_lines(CM.iread_strip(default_file))
            default_config = HDict((k,list(v)[0]) for k,v
                                   in default_config.iteritems())
                                   
        if vdebug:
            assert (defaul_configt is None or 
                    is_config(default_config)),default_config

        return dom,default_config

is_setting = lambda (vn,vv): isinstance(vn,str) and isinstance(vv,str)
is_csetting = lambda (vn,vs): isinstance(vn,str) and is_valset(vs)

is_config = lambda c: (isinstance(c,HDict) and 
                       all(is_setting(s) for s in c.iteritems()))
is_configs = lambda cs: (isinstance(cs,list) and
                         all(is_config(c) for c in cs))
is_core = lambda c: (c is None or
                     (isinstance(c,HDict) and
                      all(is_csetting(s) for s in c.iteritems())))

class PNCORE(tuple):
    def __init__(self,(pc,nc)):
        if vdebug:
            assert is_core(pc) and is_core(nc),(pc,nc)
        tuple.__init__(self,(pc,nc))

        self.pc = pc
        self.nc = nc

    def __str__(self):
        pcore_s = lambda: "p({})".format(str_of_core(self.pc))
        ncore_s = lambda: "n({})".format(str_of_core(self.nc))

        if self.pc and self.nc:
            return '{}, {}'.format(pcore_s(),ncore_s())
        elif not self.nc:
            return pcore_s()
        elif not self.pc:
            return ncore_s()
        else:
            raise AssertionError('unexpected cores {}, {}'.format(self.pc,self.nc))
    def __len__(self):
        return len_core(self.pc) + len_core(self.nc)

class CORES_D(OrderedDict):
    def __init__(self,cores_d):
        if vdebug:
            assert (all(isinstance(sid,str) and isinstance(c,PNCORE)
                        for sid,c in cores_d.iteritems())), cores_d
        OrderedDict.__init__(self,cores_d)

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
    def __init__(self,mcores_d):
        if vdebug:
            assert all(isinstance(c,PNCORE) and is_cov(cov)
                       for c,cov in mcores_d.iteritems()), mcores_d

        OrderedDict.__init__(self,mcores_d)
    def __str__(self):
        ss = []
        mcores_d_ = sorted(self.iteritems(),
                           key=lambda (c,cov):(len(c),len(cov)))

        for i,(pncore,covs) in enumerate(mcores_d_):
            s = ("{}. ({}) {}: {}"
                 .format(i+1,len(pncore),
                         pncore,
                         ','.join(covs) if print_cov else len(covs)))
            ss.append(s)
        return '\n'.join(ss)

    @property
    def lens(self):
        res = []
        sizs = [len(c) for c in self]

        for siz in sorted(set(sizs)):
            siz_conds = [c for c in self if len(c) == siz]
            cov = set()
            for c in siz_conds:
                for sid in self[c]:
                    cov.add(sid)
            res.append((siz,len(siz_conds),len(cov)))
        return res

    def str_of_lens(self):
        return ','.join("({},{},{})".format(siz,ninters,ncovs)
                        for siz,ninters,ncovs in self.lens)

### PRETTY PRINT ###
def str_of_cov(cov):
    if vdebug:
        assert is_cov(cov),cov
    
    return ','.join(sorted(cov)) if print_cov else str(len(cov))

def str_of_valset(s):
    if vdebug:
        assert is_valset(s),s
        
    return ','.join(sorted(s))


def str_of_setting((vn,vv)):
    if vdebug:
        assert is_setting((vn,vv)),(vn,vv)

    return '{}={}'.format(vn,vv)
    
def str_of_csetting((vn,vs)):
    if vdebug:
        assert is_csetting((vn,vs)), (vn,vs)

    return '{}={}'.format(vn,str_of_valset(vs))

def str_of_config(c):
    if vdebug:
        assert is_config(c), c

    return ' '.join(map(str_of_setting,c.iteritems()))

def str_of_core(c):
    if vdebug:
        assert is_core(c), c

    if c is None:
        return "false" #never reached
    if not c:
        return "true"  #no constraint

    return ' '.join(map(str_of_csetting,c.iteritems()))


def str_of_configs(configs,covs=None):
    if vdebug:
        assert is_configs(configs), configs
        assert (covs is None or 
                (len(covs) == len(configs) and
                 all(is_cov(cov) for cov in covs))), covs
        
    if  covs:
        return '\n'.join("{}. {}: {}"
                         .format(i,str_of_config(c),str_of_cov(cov))
                         for i,(c,cov) in enumerate(zip(configs,covs)))
    else:
        return '\n'.join("{}. {}".format(i,str_of_config(c))
                         for i,c in enumerate(configs))



### MISCS UTILS ###
len_core = lambda c: len(c) if c else 0




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
                "{} new samples, {} new covs, {} updated cores, "
                .format(len(samples),len(new_covs),len(new_cores)) +
                "{}".format("** progress **" if new_covs or new_cores else ""))
                
    logger.debug('sel_core\n{}'.format(sel_core))
    logger.debug('samples\n'+str_of_configs(samples,covs))
    mcores_d = cores_d.merge()
    logger.debug('mcores\n{}'.format(mcores_d))
    logger.info(mcores_d.str_of_lens)
    return mcores_d

def str_of_summary(seed,iters,ntime,ctime,nsamples,ncovs,tmpdir):
    s = ("Summary: Seed {}, Iters {}, Time ({}, {}), Samples {}, Covs {}, Tmpdir '{}'"
         .format(seed,iters,ntime,ctime,nsamples,ncovs,tmpdir))
    return s
    
def replay(dirname):
    logger.debug("load '{}'".format(dirname))
    iobj,robjs = load_dir(dirname)
    logger.debug("load '{}' robjs".format(len(robjs)))
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

def read_ugur_file(filename):
    lines = CM.iread_strip(filename)
    configs = []
    nskip = 0
    options = None
    for i,l in enumerate(lines):
        if not l.startswith("config"):
            if not options:
                options = l.split()
                options = options[:-1]
            continue
        
        config_cov = l.split(";")
        assert len(config_cov)==2, config_cov
        config,cov = config_cov
        cov = cov.strip().split()
        if not cov:
            nskip += 1
            continue

        config = config.strip()
        assert config.startswith("config")
        vals = list(config[6:])
        assert len(vals) == len(options), (vals,options)
        config = HDict(zip(options,vals))
        configs.append(config)

    logger.info("{}: read {} configs, skip {}"
                .format(filename,len(configs),nskip))

    return options,configs
        
def read_ugur_dir(dirname):
    configs = []
    old_options = None
    for fn in os.listdir(dirname):
        fn = os.path.join(dirname,fn)
        options, configs_ = read_ugur_file(fn)
        if old_options is None:
            old_options = options
            print '|options|={}'.format(len(old_options))            
        else:
            if old_options == options:
                configs.extend(configs_)
            else:
                print 'skip entire file: {}'.format(fn)

    logger.info("{}: read {} configs".format(dirname,len(configs)))
    return configs
    
### inference ###
def infer(configs,existing_core,dom):
    if vdebug:
        assert is_core(existing_core),existing_core
        assert is_configs(configs),configs
        assert isinstance(dom,Dom),dom

    if not configs:
        return existing_core

    if existing_core is None:  #not yet set
        existing_core = min(configs,key=lambda c:len(c))
        existing_core = HDict([(x,frozenset([y]))
                               for x,y in existing_core.iteritems()])


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
    for x,ys in existing_core.iteritems():
        ys_ = f(x,ys,len(dom[x]))
        if ys_:
            settings.append((x,frozenset(ys_)))
            
    core = HDict(settings)
    if vdebug:
        assert is_core(core), core

    return core    

def infer_cov(configs,covs,existing_cores_d,dom):
    """
    existing_cores_d = {l:pcoren}
    """
    
    if vdebug:
        assert is_configs(configs), configs
        assert is_covs(covs), covs        
        assert isinstance(existing_cores_d,CORES_D), existing_cores_d
        assert isinstance(dom,Dom), dom
        
    new_covs,new_cores = set(),set()  #updated stuff
    
    if not configs:
        return new_covs,new_cores

    sids = set(existing_cores_d.keys())
    for cov in covs:
        for sid in cov:
            sids.add(sid)

    cached = {}
    for li,sid in enumerate(sorted(sids)):
        if sid in existing_cores_d:
            existing_pcore,existing_ncore = existing_cores_d[sid]
        else:
            #don't have existing cores yet (use None, not emptyset)
            existing_pcore,existing_ncore = None, None
            new_covs.add(sid)
        
        configs_p,configs_n = [],[]

        #L,nL=[],[]
        for ci,(config,cov) in enumerate(zip(configs,covs)):
            if sid in cov:
                configs_p.append(config)
                #L.append(ci+1)
            else:
                configs_n.append(config)
                #nL.append(ci+1)


        configs_p = frozenset(configs_p)
        key_p = (configs_p,existing_pcore) 
        if key_p in cached:
            pcore  = cached[key_p]
        else:
            configs_p = list(configs_p)
            pcore = infer(configs_p,existing_pcore,dom)
            cached[key_p] = pcore

        configs_n = frozenset(configs_n)
        key_n = (configs_n,existing_ncore)
        if key_n in cached:
            ncore = cached[key_n]
        else:
            configs_n = list(configs_n)
            ncore = infer(configs_n,existing_ncore,dom)
            cached[key_n] = ncore

        if pcore != existing_pcore or ncore != existing_ncore:
            existing_cores_d[sid] = PNCORE((pcore,ncore))
            new_cores.add(sid)

    return new_covs,new_cores



### generate configs

def gen_configs_full(dom):
    if vdebug:
        assert isinstance(dom,Dom), dom
        
    ns,vs = itertools.izip(*dom.iteritems())
    configs = [HDict(zip(ns,c)) for c in itertools.product(*vs)]

    if vdebug:
        assert is_configs(configs),configs
        
    return configs

def gen_configs_rand(n,dom):
    if vdebug:
        assert n>=0,n
        assert isinstance(dom,Dom),dom

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
        assert isinstance(dom,Dom), dom
        
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
        assert n >= 0, n
        assert is_core(core), core
        assert isinstance(dom,Dom),dom

    if n == 0 or not core:
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
        assert isinstance(pncore,PNCORE), pncore
        assert isinstance(dom,Dom), dom

    pcore,ncore = pncore
    configs_p = gen_configs_core(len_core(pcore),pcore,dom)
    configs_n = gen_configs_core(len_core(ncore),ncore,dom)
    configs = list(set(configs_p + configs_n))
    
    if vdebug:
        assert is_configs(configs),configs
        
    return configs

### iterative refinement alg
def select_core(cores,ignore_sizs,ignore_cores):
    if vdebug:
        assert cores and isinstance(cores,frozenset), cores
        assert isinstance(ignore_sizs,set),ignore_sizs
        assert isinstance(ignore_cores,set),ignore_cores        


    cores = [core for core in cores if core not in ignore_cores]
    core_lens = [len(pncore) for pncore in cores]
    sizs = set(core_lens) - ignore_sizs
    
    if sizs:
        siz = max(sizs)
        cores_siz = [core for core,core_len in zip(cores,core_lens)
                     if core_len==siz]

        core = max(cores_siz,key=lambda (cp,cn):(len_core(cp),len_core(cn)))
        return core  #tuple (core_l,ncore)
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

def risce(dom,get_cov,seed=None,tmpdir=None,cover_siz=None,config_default=None,prefix='vu'):
    """
    cover_siz=(0,n):  generates n random configs
    """
    if vdebug:
        assert isinstance(dom,Dom),dom
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
    cores_d = CORES_D(OrderedDict())  #results {sid: (core_l,ncore)}
    cur_iter = 0
    max_stuck = 2
    cur_stuck = 0
    ignore_sizs = set([0,1]) #ignore sizes
    ignore_cores = set()
    configs_cache = OrderedDict()
    sel_core = PNCORE((None,None))
  
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
    new_covs,new_cores = infer_cov(samples,covs,cores_d,dom)

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
        sel_core = select_core(frozenset(cores_d.values()),
                               ignore_sizs,ignore_cores)
        if sel_core:
            ignore_cores.add(sel_core)
        else:
            logger.info('select no core for refinement,'
                        'done at iter {}'.format(cur_iter))
            break

        samples = gen_configs_cores(sel_core,dom)
        samples,covs,ctime = eval_samples(samples,get_cov,configs_cache)
        cov_time += ctime
        new_covs,new_cores = infer_cov(samples,covs,cores_d,dom)


        if new_covs or new_cores: #progress
            cur_stuck = 0
            ignore_sizs = set([0,1])
            
        else: #no progress
            cur_stuck += 1
            if cur_stuck >= max_stuck:
                ignore_sizs.add(len(sel_core))
                cur_stuck = 0

    logger.info(str_of_summary(seed,cur_iter,time()-st,cov_time,
                               len(configs_cache),len(cores_d),
                               tmpdir))
    return cores_d


def implies(x,y):
    """
    None: False,  empty: True
    """
    if x is None or y is None:
        return None
    else:
        return x.hcontent.issuperset(y.hcontent)

# def igraph(mcores_d):
#     g = {}
#     for pncore in mcores_d:
#         g[pncore]=[]
#         for pncore_ in mcores_d:
#             if pncore != pncore_:
#                 pcore,ncore = pncore
#                 pcore_,ncore_ = pncore_
#                 x = implies(pcore,pcore_)
#                 y = implies(ncore,ncore_)
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
#                     print pcore, pcore_
#                     print "2. {} => {}".format(str_of_core(pcore), str_of_core(pcore_))
#                     print ncore,ncore_
#                     print "3. {} => {}".format(str_of_core(ncore), str_of_core(ncore_))
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

def check_core_l(core_l,cov):
    """
    core_l is a likely inv for cov, i.e., cov => core_l. So check it, we want to see if not(core_l) => not(cov).
    So generate rand configs by changing core_l,  and verify that those don't cover anything in cov
    """
    pass

def check_ncore(ncore,cov):
    """
    ncore is a likely cov cond for cov, i.e., ncore => cov,  so we want to see if not(cov) => not ncore
    So generate random configs (or reuse existing ones from check_core_l) and check that each config whose cov not in cov 
    doesn't satisfy ncore
    """
    pass

# def compare_cores(cores_d_gt,cores_d):
#     """
#     Compare results with the ground truths
#     (citer,etime,samples,covs,new_covs,new_cores,sel_core,cores_d) = vload('/var/tmp/vupak_Gg/36.tvn')
#     cores_d_gt = vload('vsftpd_full.cores.tvn')
    
#     """
    
#     missCovs = set()
#     for i,(sid,cores_gt) in enumerate(cores_d_gt.iteritems()):
#         if sid not in cores_d:
#             logger.error("{}. sid '{}' not covered".format(i+1,sid))
#             missCovs.add(line)
#             continue

#         pcore_gt,ncore_gt = cores_gt

#         cores_me = cores_d[line]
#         pcore_me,ncore_me = cores_me
        
#         if hash(pcore_me) != hash(pcore_gt) or hash(ncore_me) != hash(ncore_gt):
#             print("{}. line '{}'\ncore me {}\ncore gt {}"
#                   .format(i+1,line,
#                           str_of_pncore(cores_me),
#                           str_of_pncore(cores_gt)))

#     logger.warn("{} covs missing".format(len(missCovs)))
#     logger.warn("{} cores diff".format("1"))

def benchmark_stats(results_dir,strength_thres=100000000):
    niters_total = 0
    ntime_total = 0
    nctime_total = 0    
    nsamples_total = 0
    ncovs_total = 0
    nruns_total = 0
    mcores_d_s = []
    for rdir in os.listdir(results_dir):
        rdir = os.path.join(results_dir,rdir)
        niters,ntime,ctime,nsamples,ncovs,mcores_d = replay(rdir)
        niters_total += niters
        ntime_total += ntime
        nctime_total += (ntime - ctime)
        nsamples_total += nsamples
        ncovs_total += ncovs
        mcores_d_s.append(mcores_d)
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
    for mcores_d in mcores_d_s:
        for strength,ninters,ncov in mcores_d.lens:
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
                            sum(inters)/float(len(mcores_d)),
                            sum(covs)/float(len(mcores_d))))
######


def prepare_motiv():
    import ex_motiv_run
    dom,_ = Dom.get_dom('ex_motiv.dom')
    args = {'varnames':dom.keys(),
            'prog': '~/Dropbox/git/config/src/ex_motiv.exe'}
    get_cov = lambda config: ex_motiv_run.get_cov(config,args)
    return dom,get_cov

def prepare_otter(prog):
    dir_ = '~/Src/Devel/iTree_stuff/expData/{}'.format(prog)
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
