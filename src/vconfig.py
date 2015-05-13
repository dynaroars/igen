import tempfile
import random
import os.path
import itertools

from collections import OrderedDict
from time import time

from vconfig_miscs import HDict
import vu_common as CM

logger = CM.VLog('config')
logger.level = CM.VLog.INFO
CM.VLog.PRINT_TIME = True

vdebug = False
print_cov = False

### DATA STRUCTURES ###
is_valset = lambda s: (isinstance(s,frozenset) and s and
                      all(isinstance(e,str) for e in s))
is_dom = lambda d: (isinstance(d,dict) and 
                    all(isinstance(x,str) and
                        is_valset(ys) for x,ys in d.iteritems()))

is_setting = lambda s: (isinstance(s,tuple) and len(s)==2 and
                        isinstance(s[0],str) and isinstance(s[1],str))
is_csetting = lambda s: (isinstance(s,tuple) and len(s)==2 and
                        isinstance(s[0],str) and is_valset(s[1]))

is_config = lambda c: (isinstance(c,HDict) and
                       all(is_setting(s) for s in c.iteritems()))
is_core = lambda c: (c is None or
                     (isinstance(c,HDict) and
                      all(is_csetting(s) for s in c.iteritems())))
is_configs = lambda cs: (isinstance(cs,list) and
                         all(is_config(c) for c in cs))

is_cores_pn = lambda (cl,cnl): is_core(cl) and is_core(cnl)
is_cores_pns = lambda cs: (isinstance(cs,list) and
                            all(is_cores_pn(c) for c in cs))
is_cores_d = lambda d: (isinstance(d,dict) and
                        (all(isinstance(l,str) and is_cores_pn(c)
                             for l,c in d.iteritems())))
is_mcores_d = lambda d: (isinstance(d,dict) and
                        all(is_cores_pn(c) and is_cov(cov)
                            for c,cov in d.iteritems()))

is_cov = lambda c: (isinstance(c,frozenset) and
                    all(isinstance(s,str) for s in c))
is_covs = lambda cs: (isinstance(cs,list) and
                      all_is_cov(c) for c in cs)

### PRETTY PRINT ###
def str_of_dom(dom):
    if vdebug:
        assert is_dom(dom)

    return '\n'.join('{}: {}'.format(
        k,','.join(vs)) for k,vs in dom.iteritems())
        
def str_of_setting(s):
    if vdebug:
        assert is_setting(s),s

    (x,y) = s
    return '{}={}'.format(x,y)
    
def str_of_csetting(s):
    if vdebug:
        assert is_csetting(s), s

    (x,ys) = s
    return '{}={}'.format(x,','.join(sorted(ys)))

def str_of_config(c):
    if vdebug:
        assert is_config(c), c

    return ' '.join(map(str_of_setting,c.iteritems()))

def str_of_core(c):
    if vdebug:
        assert is_core(c), c

    if c is None:
        return "false"  #never reached
    if not c:
        return "true" #no constraint

    return ' '.join(map(str_of_csetting,c.iteritems()))

def str_of_cores_pn(cores_pn):
    if vdebug:
        assert is_cores_pn(cores_pn), cores_pn
        
    (core_p,core_n) = cores_pn
    s = []
    core_p_s = lambda : "p({})".format(str_of_core(core_p))
    core_n_s = lambda: "n({})".format(str_of_core(core_n))

    if core_p and core_n:
        return '{}, {}'.format(core_p_s(),core_n_s())
    elif not core_n:
        return core_p_s()
    elif not core_p:
        return core_n_s()
    else:
        raise AssertionError('unexpected cores {}, {}'.format(core_p,core_n))
        

def str_of_configs(configs,covs=None):
    if vdebug:
        assert is_configs(configs), configs
        assert (covs is None or is_covs(covs) and
                len(covs) == len(configs)), covs
        
    if covs and print_cov:
        return '\n'.join("{}. {} : {}".format(
            i,str_of_config(c),' '.join(cov))
                         for i,(c,cov) in enumerate(zip(configs,covs)))
    else:
        return '\n'.join("{}. {}"
                         .format(i,str_of_config(c))
                         for i,c in enumerate(configs))

## PRETTY PRINT RESULTS
def str_of_cores_d(cores_d):
    if vdebug:
        assert is_cores_d(cores_d),cores_d
        
    ss = []
    for i,line in enumerate(sorted(cores_d)):
        s = ("{}. {}: {}"
             .format(i+1,
                     line,
                     str_of_cores_pn(cores_d[line])))
        ss.append(s)
    return '\n'.join(ss)

        
def str_of_mcores_d(mcores_d):
    if vdebug:
        assert is_mcores_d(mcores_d),mcores_d

    ss = []
    mcores_d_ = sorted(mcores_d.iteritems(),
                       key=lambda (c,cov):(len_cores_pn(c),len(cov)))
                       

    for i,(cores_pn,covs) in enumerate(mcores_d_):
        s = ("{}. ({}) {}: {}"
             .format(i+1,len_cores_pn(cores_pn),
                     str_of_cores_pn(cores_pn),
                     ','.join(covs) if print_cov else len(covs)))
        ss.append(s)
    return '\n'.join(ss)


def mcores_d_lens(mcores_d):
    if vdebug:
        assert is_mcores_d(mcores_d),mcores_d
    
    res = []
    sizs = [len_cores_pn(c) for c in mcores_d]

    for siz in sorted(set(sizs)):
        siz_conds = [c for c in mcores_d if len_cores_pn(c) == siz]
        cov = set()
        for c in siz_conds:
            for sid in mcores_d[c]:
                cov.add(sid)
        res.append((siz,len(siz_conds),len(cov)))
                  
    return res

def str_of_mcores_d_lens(mlens):
    if isinstance(mlens,dict):
        mlens = mcores_d_lens(mlens)
    
    ss = []
    for siz,ninters,ncovs in mlens:
        ss.append("({},{},{})".
                  format(siz,ninters,ncovs))
    return ','.join(ss)
        

### MISCS UTILS ###
len_core = lambda c: len(c) if c else 0
len_cores_pn = lambda (x,y): len_core(x) + len_core(y)
def get_dom_lines(lines):
    dom = OrderedDict()
    for line in lines:
        parts = line.split()
        varname = parts[0]
        varvals = parts[1:]
        dom[varname] = frozenset(varvals)
    return dom
    
def get_dom(dom_file):
    dom_file = os.path.realpath(dom_file)
    dom = get_dom_lines(CM.iread_strip(dom_file))
    
    config_default = None
    dom_file_default = os.path.realpath(dom_file+'.default')
    if os.path.isfile(dom_file_default):
        config_default = get_dom_lines(CM.iread_strip(dom_file_default))
        config_default = HDict((k,list(v)[0])
                               for k,v in config_default.iteritems())

    if vdebug:
        assert is_dom(dom),dom
        assert config_default is None or is_config(config_default),config_default

    return dom,config_default


def load_dir(dirname):
    iobj = CM.vload(os.path.join(dirname,'info'))
    rfiles = [os.path.join(dirname,f) for f in os.listdir(dirname)
              if f.endswith('.tvn')]
    robjs = [CM.vload(rfile) for rfile in rfiles]
    return iobj, robjs

def print_iter_stat(robj):
    (citer,etime,ctime,samples,covs,new_covs,new_cores,rcore,cores_d) = robj

    logger.info("iter {} ({}s, {}s eval) ".format(citer,etime,ctime) +
                "{} new samples, {} new covs, {} updated cores, "
                .format(len(samples),len(new_covs),len(new_cores)) +
                "{}".format("** progress **" if new_covs or new_cores else ""))
                
    logger.debug('rcore\n{}'.format(str_of_cores_pn(rcore)))
    logger.debug('samples\n'+str_of_configs(samples,covs))
    mcores_d = merge_cores_d(cores_d)
    logger.debug('mcores\n{}'.format(str_of_mcores_d(mcores_d)))
    mlens = mcores_d_lens(mcores_d)
    logger.info(str_of_mcores_d_lens(mlens))
    return mcores_d,mlens

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
             new_covs,new_cores,rcore,cores_d) = robj
        else: #old format
            (citer,etime,samples,covs,
             new_covs,new_cores,rcore,cores_d) = robj
            ctime = 0.0
            robj = (citer,etime,ctime,samples,covs,
                    new_covs,new_cores,rcore,cores_d)

        mcores_d,mlens = print_iter_stat(robj)
        ntime+=etime
        nsamples+=len(samples)
        ncovs+=len(new_covs)
        #print mlens
        # minint = mlens[0][0]
        # maxint = mlens[-1][0]
        # tt = [citer+1,nsamples,ncovs,len(mcores_d),"{} ({},{})".format(len(mlens),minint,maxint)]
        #print "{}\\\\".format(" & ".join(map(str,tt)))
        
    niters = len(robjs)
    logger.info(str_of_summary(seed,niters,
                               ntime,ctime,
                               nsamples,ncovs,dirname))

    return niters,ntime,ctime,nsamples,ncovs,mcores_d,mlens

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
        assert is_dom(dom),dom

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
    existing_cores_d = {l:core_pn}
    """
    
    if vdebug:
        assert is_configs(configs), configs
        assert is_covs(covs), covs        
        assert isinstance(existing_cores_d,dict), existing_cores_d
        assert is_dom(dom), dom
        
    new_covs,new_cores = set(),set()  #updated stuff
    
    if not configs:
        return new_covs,new_cores

    lines = set(existing_cores_d.keys())
    for cov in covs:
        for line in cov:
            lines.add(line)

    cached = {}
    for li,line in enumerate(sorted(lines)):
        # if line != 'str.c:499':
        #     continue
        
        if line in existing_cores_d:
            existing_core_p,existing_core_n = existing_cores_d[line]
        else:
            #don't have existing cores yet (use None, not emptyset)
            existing_core_p,existing_core_n = None, None
            new_covs.add(line)
        
        configs_p,configs_n = [],[]

        #L,nL=[],[]
        for ci,(config,cov) in enumerate(zip(configs,covs)):
            if line in cov:
                configs_p.append(config)
                #L.append(ci+1)
            else:
                configs_n.append(config)
                #nL.append(ci+1)


        configs_p = frozenset(configs_p)
        key_p = (configs_p,existing_core_p) 
        if key_p in cached:
            core_p  = cached[key_p]
        else:
            configs_p = list(configs_p)
            core_p = infer(configs_p,existing_core_p,dom)
            cached[key_p] = core_p

        configs_n = frozenset(configs_n)
        key_n = (configs_n,existing_core_n)
        if key_n in cached:
            core_n = cached[key_n]
        else:
            configs_n = list(configs_n)
            core_n = infer(configs_n,existing_core_n,dom)
            cached[key_n] = core_n

        if core_p != existing_core_p or core_n != existing_core_n:
            existing_cores_d[line] = (core_p,core_n)            
            new_cores.add(line)

        # if line == 'str.c:499':
        #     print '*******************************'            
        #     print '\nline {}/{}'.format(li,len(lines))
        #     print str_of_configs(configs_p)
        #     save_file('/var/tmp/vuha0',configs_p)
        #     print str_of_core(core_p)
        #print 'pos'
        #print 'L', L
        #print str_of_configs(configs_p)
        #print 'core', str_of_core(core_p)

        #print 'neg'
        #print nL
        # print str_of_configs([configs_n)
        # print 'core', str_of_core(core_n)

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
        assert n>=0,n
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
        assert n >= 0, n
        assert is_core(core), core
        assert is_dom(dom),dom

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
    

def gen_configs_cores(cores_pn,dom):
    if vdebug:
        assert is_cores_pn(cores_pn), cores_pn
        assert is_dom, dom

    core_p,core_n = cores_pn
    configs_p = gen_configs_core(len_core(core_p),core_p,dom)
    configs_n = gen_configs_core(len_core(core_n),core_n,dom)
    configs = list(set(configs_p + configs_n))
    
    if vdebug:
        assert is_configs(configs),configs
        
    return configs

### iterative refinement alg
def select_core(cores,ignore_sizs,ignore_cores):
    if vdebug:
        assert isinstance(ignore_sizs,set),ignore_sizs
        assert isinstance(ignore_cores,set),ignore_cores        

    cores = [core for core in cores if core not in ignore_cores]
    
    core_lens = [len_cores_pn(cores_pn) for cores_pn in cores]
    sizs = set(core_lens) - ignore_sizs

    if sizs:
        siz = max(sizs)
        cores_siz = [core for core,core_len in zip(cores,core_lens)
                     if core_len==siz]

        core= random.choice(cores_siz)
        return core  #tuple (core_l,core_n)
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
            cache[config]=get_cov(config)
            samples_.append(config)

    samples = samples_
    covs = [cache[config] for config in samples]

    return samples,covs,time() - st

def iterative_refine(dom,get_cov,seed=None,tmpdir=None,pure_random_n=None,cover_siz=None,config_default=None,prefix='vu'):
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
    cores_d = OrderedDict()  #results {line: (core_l,core_n)}
    cur_iter = 0
    max_stuck = 2
    cur_stuck = 0
    ignore_sizs = set([0,1]) #ignore sizes
    ignore_cores = set()
    configs_cache = OrderedDict()
    rcore = (None,None)
  
    #begin
    st = time()
    ct = st
    cov_time = 0.0
    
    #initial samples
    if pure_random_n:
        if pure_random_n < 0:
            samples = gen_configs_full(dom)
            logger.info("Gen all {} configs".format(len(samples)))
        else:
            samples = gen_configs_rand(pure_random_n,dom)
            logger.info("Gen {} rand configs".format(len(samples)))
    elif cover_siz:
        csiz,max_confs = cover_siz
        samples_tcover = gen_configs_tcover(csiz,dom,seed,tmpdir)
        samples_rand_n = max_confs - len(samples_tcover)

        if samples_rand_n:
            samples_rand = gen_configs_rand(samples_rand_n,dom)
        else:
            samples_rand = []
            
        samples = list(set(samples_tcover + samples_rand))
        logger.info("Gen {} configs ({} {}-cover, {} rand)"
                    .format(len(samples),
                            len(samples_tcover),csiz,
                            len(samples_rand)))
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
                rcore,
                cores_d)
        rfile = os.path.join(tmpdir,"{}.tvn".format(cur_iter))        
        CM.vsave(rfile,robj)
        print_iter_stat(robj)
        
        if pure_random_n or cover_siz:
            break
        

        cur_iter += 1
        rcore = select_core(set(cores_d.values()),
                            ignore_sizs,ignore_cores)
        if rcore:
            ignore_cores.add(rcore)
        else:
            logger.info('select no core for refinement,'
                        'done at iter {}'.format(cur_iter))
            break

        samples = gen_configs_cores(rcore,dom)
        samples,covs,ctime = eval_samples(samples,get_cov,configs_cache)
        cov_time += ctime
        new_covs,new_cores = infer_cov(samples,covs,cores_d,dom)


        if new_covs or new_cores: #progress
            cur_stuck = 0
            ignore_sizs = set([0,1])
            
        else: #no progress
            cur_stuck += 1
            if cur_stuck >= max_stuck:
                ignore_sizs.add(len_cores_pn(rcore))
                cur_stuck = 0

    logger.info(str_of_summary(seed,cur_iter,time()-st,cov_time,
                               len(configs_cache),len(cores_d),
                               tmpdir))
    return cores_d

### Post processing ###
def merge_cores_d(cores_d):
    if vdebug:
        assert is_cores_d(cores_d), cores_d
        
    mcores_d = OrderedDict()
    for line,core_pn in cores_d.iteritems():
        
        if core_pn in mcores_d:
            mcores_d[core_pn].add(line)
        else:
            mcores_d[core_pn] = set([line])

    for line in mcores_d:
        mcores_d[line]=frozenset(mcores_d[line])
        
    if vdebug:
        assert is_mcores_d(mcores_d), mcores_d
        
    return mcores_d

def implies(x,y):
    """
    None: False,  empty: True
    """
    if x is None or y is None:
        return None
    else:
        return x.hcontent.issuperset(y.hcontent)

def igraph(mcores_d):
    g = {}
    for core_pn in mcores_d:
        g[core_pn]=[]
        for core_pn_ in mcores_d:
            if core_pn != core_pn_:
                core_p,core_n = core_pn
                core_p_,core_n_ = core_pn_
                x = implies(core_p,core_p_)
                y = implies(core_n,core_n_)
                if x is None and y is None:
                    r = False
                elif x is None:
                    r = False
                elif y is None:
                    r = False
                else:
                    r = x and y
                if r:
                    print "1. {} => {}".format(str_of_cores_pn(core_pn),str_of_cores_pn(core_pn_))
                    print core_p, core_p_
                    print "2. {} => {}".format(str_of_core(core_p), str_of_core(core_p_))
                    print core_n,core_n_
                    print "3. {} => {}".format(str_of_core(core_n), str_of_core(core_n_))
                    print ""
                    CM.pause()
                    g[core_pn].append(core_pn_)
    return g
    
def str_of_igraph(g):
    ss = []
    for core_pn,core_pns in g.iteritems():
        ss.append("{}: ({}) {}"
                  .format(str_of_cores_pn(core_pn),
                          len(core_pns),
                          ', '.join(str_of_cores_pn(c) for c in core_pns)))
    return '\n'.join(ss)

### Experiments ###

def check_core_l(core_l,cov):
    """
    core_l is a likely inv for cov, i.e., cov => core_l. So check it, we want to see if not(core_l) => not(cov).
    So generate rand configs by changing core_l,  and verify that those don't cover anything in cov
    """
    pass

def check_core_n(core_n,cov):
    """
    core_n is a likely cov cond for cov, i.e., core_n => cov,  so we want to see if not(cov) => not core_n
    So generate random configs (or reuse existing ones from check_core_l) and check that each config whose cov not in cov 
    doesn't satisfy core_n
    """
    pass

def compare_cores(cores_d_gt,cores_d):
    """
    Compare results with the ground truths
    (citer,etime,samples,covs,new_covs,new_cores,rcore,cores_d) = vload('/var/tmp/vupak_Gg/36.tvn')
    cores_d_gt = vload('vsftpd_full.cores.tvn')
    
    """
    
    missCovs = set()
    for i,(line,cores_gt) in enumerate(cores_d_gt.iteritems()):
        if line not in cores_d:
            logger.error("{}. line '{}' not covered".format(i+1,line))
            missCovs.add(line)
            continue

        core_p_gt,core_n_gt = cores_gt

        cores_me = cores_d[line]
        core_p_me,core_n_me = cores_me
        
        if hash(core_p_me) != hash(core_p_gt) or hash(core_n_me) != hash(core_n_gt):
            print("{}. line '{}'\ncore me {}\ncore gt {}"
                  .format(i+1,line,
                          str_of_cores_pn(cores_me),
                          str_of_cores_pn(cores_gt)))

    logger.warn("{} covs missing".format(len(missCovs)))
    logger.warn("{} cores diff".format("1"))

def benchmark_stats(results_dir,strength_thres=100000000):
    niters_total = 0
    ntime_total = 0
    nctime_total = 0    
    nsamples_total = 0
    ncovs_total = 0
    nruns_total = 0
    mcores_d_s = []
    mlens_s = []
    for rdir in os.listdir(results_dir):
        rdir = os.path.join(results_dir,rdir)
        niters,ntime,ctime,nsamples,ncovs,mcores_d,mlens = replay(rdir)
        niters_total += niters
        ntime_total += ntime
        nctime_total += (ntime - ctime)
        nsamples_total += nsamples
        ncovs_total += ncovs
        mcores_d_s.append(mcores_d)
        mlens_s.append(mlens)
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
    for mlens in mlens_s:
        print str_of_mcores_d_lens(mlens)
        for strength,ninters,ncov in mlens:
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
                            sum(inters)/float(len(mlens_s)),
                            sum(covs)/float(len(mlens_s))))
######


def prepare_motiv():
    import ex_motiv_run
    dom,_ = get_dom('ex_motiv.dom')
    args = {'varnames':dom.keys(),
            'prog': '/Users/tnguyen/Dropbox/git/config/new/ex_motiv.exe'}
    get_cov = lambda config: ex_motiv_run.get_cov(config,args)
    return dom,get_cov

def prepare_otter(prog):
    dir_ = '/Users/tnguyen/Src/Devel/iTree_stuff/expData/{}'.format(prog)
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


def test_motiv(dom,get_cov):
    #listen time ssl local anon log chunks
    #0 1 0 0 1 1 2
    existing_cores_d = {}        
    c1 = HDict(zip(dom,'1 0 1 1 1 0 2'.split()))
    c2 = HDict(zip(dom,'1 0 0 0 0 1 4'.split()))
    c3 = HDict(zip(dom,'0 1 0 1 0 1 3'.split()))
    c4 = HDict(zip(dom,'1 0 1 1 0 0 1'.split()))
    configs = [c1,c2,c3,c4]
    covs = [get_cov(config) for config in configs]
    print(str_of_configs(configs,covs))    
    new_covs,new_cores = infer_cov(configs,covs,existing_cores_d,dom)
    print(str_of_cores_d(existing_cores_d))    
    mcores_d = merge_cores_d(existing_cores_d)
    print(str_of_mcores_d(mcores_d))
    CM.pause()
    
    c5 = HDict(zip(dom,'0 0 0 0 1 1 3'.split()))
    c6 = HDict(zip(dom,'0 1 1 1 0 1 4'.split()))
    c7 = HDict(zip(dom,'0 1 0 0 1 1 2'.split()))
    c8 = HDict(zip(dom,'1 0 1 1 1 0 3'.split()))

    configs = [c5,c6,c7,c8]
    covs = [get_cov(config) for config in configs]
    print(str_of_configs(configs,covs))
    new_covs,new_cores = infer_cov(configs,covs,existing_cores_d,dom)
    print(str_of_cores_d(existing_cores_d))
    mcores_d = merge_cores_d(existing_cores_d)
    print(str_of_mcores_d(mcores_d))

    return existing_cores_d

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
