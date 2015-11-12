import os.path
import numpy
import itertools
from time import time
import vu_common as CM

import config_common as CC
import igen_alg as IA

logger = CM.VLog('analysis')
logger.level = CC.logger_level
CM.VLog.PRINT_TIME = True

from collections import namedtuple
fields = ['niters',
          'ncores',
          'itime',
          'xtime',
          'ncovs',          
          'nconfigs',
          'n_min_configs',
          'min_ncovs',
          'vscores',
          'fscores',
          'r_fvscores',
          'influence_scores',
          'm_strens',
          'm_strens_str',
          'm_vtyps']
Results = namedtuple("Results",' '.join(fields))
                     
class Analysis(object):
    @staticmethod
    def is_run_dir(d):
        """
        ret True if d is a run_dir that consists of *.tvn, pre, post files
        ret False if d is a benchmark dir that consists of run_dirs
        ret None otherwise
        """
        if __debug__:
            assert os.path.isdir(d), d
            
        fs = os.listdir(d)
        if (fs.count('pre') == 1 and fs.count('post') == 1 and
            any(f.endswith('.tvn') for f in fs)):
            return True
        else:
            ds = [os.path.join(d,f) for f in fs]
            if (ds and all(os.path.isdir(d_) and
                           Analysis.is_run_dir(d_) for d_ in ds)):
                return False
            else:
                return None

    @staticmethod
    def check_pp_cores_d(pp_cores_d,dom):
        if not hasattr(pp_cores_d.values()[0],'vstr'):
            logger.warn("Old format, has no vstr .. re-analyze")
            return pp_cores_d.analyze(dom,covs_d=None)
        else:
            return pp_cores_d
            
    @staticmethod
    def replay(dir_, show_iters, do_min_configs, cmp_gt, cmp_rand):
        """
        Replay and analyze execution info from saved info in dir_
        do_min_configs has 3 possible values
        1. None: don't find min configs
        2. f: (callable(f)) find min configs using f
        3. anything else: find min configs using existing configs
        """
        if __debug__:
            assert isinstance(dir_,str), dir_
            assert isinstance(show_iters,bool),show_iters
            assert cmp_gt is None or isinstance(cmp_gt,str), cmp_gt
            assert cmp_rand is None or callable(cmp_rand), cmp_rand
        
        logger.info("replay dir: '{}'".format(dir_))
        seed,dom,dts,pp_cores_d,itime_total = IA.DTrace.load_dir(dir_)
        logger.info('seed: {}'.format(seed))
        logger.debug(dom.__str__())

        dts = sorted(dts,key=lambda dt: dt.citer)        
        if show_iters:
            for dt in dts:
                dt.show()

        pp_cores_d = Analysis.check_pp_cores_d(pp_cores_d,dom)
        mcores_d = pp_cores_d.merge(show_detail=True)
        
        #print summary
        xtime_total = itime_total - sum(dt.xtime for dt in dts)
        last_dt = max(dts,key=lambda dt: dt.citer) #last iteration
        nconfigs = last_dt.nconfigs
        ncovs = last_dt.ncovs
        
        logger.info(IA.DTrace.str_of_summary(
            seed,len(dts),itime_total,xtime_total,nconfigs,ncovs,dir_))

        #min config
        if not do_min_configs: #None
            n_min_configs = 0
            min_ncovs = 0
        elif callable(do_min_configs):
            f = do_min_configs
            min_configs, min_ncovs = HighCov.get_minset_f(
                mcores_d,set(pp_cores_d),f,dom)
            n_min_configs = len(min_configs)            
        else:
            #reconstruct information
            configs_d = CC.Configs_d()
            for dt in dts:
                for c in dt.cconfigs_d:
                    configs_d[c] = dt.cconfigs_d[c]

            min_configs, min_ncovs = HighCov.get_minset_configs_d(
                mcores_d,set(pp_cores_d),configs_d,dom)
            n_min_configs = len(min_configs)

        #Additional analysis
        influence_scores = Influence.get_influence(mcores_d,ncovs,dom)
        fscores,vscores,gt_pp_cores_d = Metrics.get_scores(dts,dom,cmp_gt)

        #rand search
        r_f = cmp_rand
        if callable(r_f):
            r_pp_cores_d,r_cores_d,r_configs_d,r_covs_d,_ = r_f(nconfigs)
            if gt_pp_cores_d:
                r_fscore = Metrics.fscore_cores_d(r_pp_cores_d,gt_pp_cores_d,dom)
            else:
                r_fscore = None
                
            r_vscore = Metrics.vscore_cores_d(r_cores_d,dom)
            logger.info("rand: configs {} cov {} vscore {} fscore {}"
                        .format(len(r_configs_d),len(r_covs_d),
                                r_vscore,r_fscore))
            last_elem_f = lambda l: l[-1][1] if l and len(l) > 0 else None
            logger.info("cegir: configs {} cov {} vscore {} fscore {}"
                        .format(nconfigs,ncovs,
                                last_elem_f(vscores),last_elem_f(fscores)))

            r_fvscores = (r_fscore,r_vscore)
        else:
            r_fvscores = None

        rs = Results(niters=len(dts),
                     ncores=len(mcores_d),
                     itime=itime_total,
                     xtime=xtime_total,
                     ncovs=ncovs,
                     nconfigs=nconfigs,
                     n_min_configs=n_min_configs,
                     min_ncovs=min_ncovs,
                     vscores=vscores,
                     fscores=fscores,
                     r_fvscores=r_fvscores,
                     influence_scores=influence_scores,
                     m_strens=mcores_d.strens,
                     m_strens_str=mcores_d.strens_str,
                     m_vtyps=mcores_d.vtyps)
        return rs

    @staticmethod
    def replay_dirs(dir_,show_iters,do_min_configs,cmp_gt,cmp_rand):
        dir_ = CM.getpath(dir_)
        logger.info("replay_dirs '{}'".format(dir_))
        
        niters_total = 0
        ncores_total = 0        
        nitime_total = 0
        nxtime_total = 0    
        nconfigs_total = 0
        ncovs_total = 0
        nminconfigs_total = 0
        min_ncovs_total = 0        
        strens_s = []
        strens_str_s = []
        vtyps_s = []

        #modified by ugur
        niters_arr = []
        ncores_arr = []
        nitime_arr = []
        nxtime_arr = [] 
        nconfigs_arr = []
        ncovs_arr = []
        nminconfigs_arr = []
        min_ncovs_arr = []        
        counter = 0
        csv_arr = []

        for rdir in sorted(os.listdir(dir_)):
            rdir = os.path.join(dir_,rdir)
            rs = Analysis.replay(
                rdir,show_iters,do_min_configs,cmp_gt,cmp_rand)
            
            niters_total += rs.niters
            ncores_total += rs.ncores
            nitime_total += rs.itime
            nxtime_total += rs.xtime
            ncovs_total += rs.ncovs            
            nconfigs_total += rs.nconfigs
            min_ncovs_total += rs.min_ncovs
            nminconfigs_total += rs.n_min_configs
            strens_s.append(rs.m_strens)
            strens_str_s.append(rs.m_strens_str)
            vtyps_s.append(rs.m_vtyps)

            niters_arr.append(rs.niters)
            ncores_arr.append(rs.ncores)
            nitime_arr.append(rs.itime)
            nxtime_arr.append(rs.xtime)
            nconfigs_arr.append(rs.nconfigs)
            ncovs_arr.append(rs.ncovs)
            min_ncovs_arr.append(rs.min_ncovs)
            nminconfigs_arr.append(rs.n_min_configs)
            csv_arr.append("{},{},{},{},{},{},{},{},{},{},{}".format(
                counter,rs.niters,rs.ncores,rs.itime,rs.xtime,
                rs.nconfigs,rs.ncovs,rs.n_min_configs,rs.min_ncovs,
                ','.join(map(str, rs.m_vtyps)),
                ','.join(map(str, rs.m_strens))))
            counter += 1

        nruns_total = float(len(strens_s))

        ss = ["iter {}".format(niters_total/nruns_total),
              "results {}".format(ncores_total/nruns_total),
              "time {}".format(nitime_total/nruns_total),
              "xtime {}".format(nxtime_total/nruns_total),
              "configs {}".format(nconfigs_total/nruns_total),
              "covs {}".format(ncovs_total/nruns_total),
              "minconfigs {}".format(nminconfigs_total/nruns_total)]
        
        logger.info("STATS of {} runs (averages): {}".format(nruns_total,', '.join(ss)))
        
        ssMed = ["iter {}".format(numpy.median(niters_arr)),
                 "results {}".format(numpy.median(ncores_arr)),
                 "time {}".format(numpy.median(nitime_arr)),
                 "xtime {}".format(numpy.median(nxtime_arr)),
                 "configs {}".format(numpy.median(nconfigs_arr)),
                 "nminconfigs {}".format(numpy.median(nminconfigs_arr)),
                 "covs {}".format(numpy.median(ncovs_arr))]
        logger.info("STATS of {} runs (medians) : {}".format(nruns_total,', '.join(ssMed)))
        
        ssSIQR = ["iter {}".format(Analysis.siqr(niters_arr)),
                  "results {}".format(Analysis.siqr(ncores_arr)),
                  "time {}".format(Analysis.siqr(nitime_arr)),
                  "xtime {}".format(Analysis.siqr(nxtime_arr)),
                  "configs {}".format(Analysis.siqr(nconfigs_arr)),
                  "nminconfigs {}".format(Analysis.siqr(nminconfigs_arr)),                  
                  "covs {}".format(Analysis.siqr(ncovs_arr))]
        logger.info("STATS of {} runs (SIQR)   : {}".format(nruns_total,', '.join(ssSIQR)))

        sres = {}
        for i,(strens,strens_str) in enumerate(zip(strens_s,strens_str_s)):
            logger.debug("run {}: {}".format(i+1,strens_str))
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
        
        conjs = [c for c,_,_ in vtyps_s]
        disjs = [d for _,d,_ in vtyps_s]
        mixs = [m for _,_,m in vtyps_s]
        
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
        try:
            return (numpy.percentile(arr, 75, interpolation='higher') - 
                    numpy.percentile(arr, 25, interpolation='lower'))/2
        except TypeError:
            return (numpy.percentile(arr, 75) - numpy.percentile(arr, 25))/2
                    
    
    @staticmethod
    def debug_find_configs(sid,configs_d,find_in):
        if find_in:
            cconfigs_d = dict((c,cov) for c,cov in configs_d.iteritems()
                           if sid in cov)
        else:
            cconfigs_d = dict((c,cov) for c,cov in configs_d.iteritems()
                              if sid not in cov)

        logger.info(cconfigs_d)


import z3
import z3util        
class HighCov(object):
    """
    Use interactions to generate high cov configs

    Utils
    >>> from z3 import And,Or,Bools,Not
    >>> a,b,c,d,e,f = Bools('a b c d e f')
    
    >>> exprs_d = {'a':a,\
    'b':b,\
    'a & b': z3.And(a,b),\
    'b | c': z3.Or(b,c),\
    'a & b & (c|d)': z3.And(a,b,z3.Or(c,d)),\
    'b & c' : z3.And(b,c),\
    'd | b' : z3.Or(b,d),\
    'e & !a': z3.And(e,Not(a)),\
    'f & a': z3.And(f,a)}
    
    >>> print '\\n'.join(map(str,sorted(map(str,exprs_d))))
    a
    a & b
    a & b & (c|d)
    b
    b & c
    b | c
    d | b
    e & !a
    f & a
    
    >>> exprs_d = HighCov.prune(exprs_d)
    >>> print '\\n'.join(map(str,sorted(exprs_d)))
    a & b & (c|d)
    b & c
    e & !a
    f & a
    
    >>> fs = HighCov.pack(exprs_d)
    >>> print '\\n'.join(sorted(map(HighCov.str_of_pack,fs)))
    (a & b & (c|d); b & c; f & a)
    (e & !a)


    >>> HighCov.prune({'a':None})
    Traceback (most recent call last):
    ...
    AssertionError: {'a': None}

    >>> HighCov.pack({'a':None})
    Traceback (most recent call last):
    ...
    AssertionError: {'a': None}
    """
    
    @staticmethod

    def prune(d):
        """
        Ret the strongest elements by removing those implied by others
        """
        if __debug__:
            assert (d and isinstance(d,dict) and
                    all(z3.is_expr(v) for v in d.itervalues())), d
        
        def _len(e):
            #simply heuristic to try most restrict conjs first
            if z3util.is_expr_var(e):  #variables such as x,y
                return 1
            else: #conj/disj
                nchildren = len(e.children())
                if nchildren:
                    if z3.is_app_of(e,z3.Z3_OP_OR):
                        return 1.0 / nchildren
                    elif z3.is_app_of(e,z3.Z3_OP_AND):
                        return nchildren
                    elif z3.is_app_of(e,z3.Z3_OP_EQ) and nchildren == 2:
                        return 1
                    else:
                        logger.warn("cannot compute _len of {}".format(e))
                        return nchildren
                else:
                    logger.warn("f:{} has 0 children".format(e))
                    return 1

        #sort by most restrict conj, also remove None ("true")
        fs = sorted([f for f in d if d[f]],
                    key=lambda f: _len(d[f]),reverse=True)

        implied = set()
        for f in fs:
            if f in implied:
                continue
            
            for g in fs:
                if f is g or g in implied:
                    continue
                e = z3.Implies(d[f],d[g])
                is_implied = z3util.is_tautology(e)
                if is_implied:
                    implied.add(g)
                #print "{} => {} {}".format(f,g,is_implied)
                
                
        return dict((f,d[f]) for f in fs if f not in implied)

    @staticmethod
    def pack2(fs,d):
        if __debug__:
            assert all(isinstance(f,tuple) for f in fs),fs
            assert all(f in d and z3.is_expr(d[f])
                       for f in fs), (fs, d)
        fs_ = []
        packed = set()
        for f,g in itertools.combinations(fs,2):
            if f in packed or g in packed or (f,g) in d:
                continue

            e = z3.And(d[f],d[g])
            if z3util.is_sat(e):
                packed.add(f)
                packed.add(g)
                fs_.append(f+g)
                d[f+g] = e
            else:
                d[(f,g)] = None  #cache not sat result

        fs = [f for f in fs if f not in packed]
        return fs_ + fs
                
    @staticmethod
    def pack(d):
        """
        Pack together elements that have no conflicts.
        The results are {tuple -> z3expr}
        It's good to first prune them (call prune()).
        """
        if __debug__:
            assert all(z3.is_expr(v) for v in d.itervalues()), d
                       
        #change format, results are tuple(elems)
        d = dict((tuple([f]),d[f]) for f in d)
        fs = d.keys()
        fs_ = HighCov.pack2(fs,d)        
        while len(fs_) < len(fs):
            fs = fs_
            fs_ = HighCov.pack2(fs,d)

        fs = set(fs_)
        return dict((f,d[f]) for f in d if f in fs_)

    @staticmethod
    def indep(fs,dom):
        """
        Apply prune and pack on fs
        """
        z3db = dom.z3db

        #prune
        d = dict((c,c.z3expr(z3db,dom)) for c in fs)
        d = HighCov.prune(d)
        logger.info("prune: {} remains".format(len(d)))
        logger.debug("\n{}".format('\n'.join(
            "{}. {}".format(i+1,str(c)) for i,c
            in enumerate(sorted(d)))))

        #pack
        d = HighCov.pack(d)
        logger.info("pack: {} remains".format(len(d)))
        logger.debug("\n{}".format('\n'.join(
            "{}. {}".format(i+1,HighCov.str_of_pack(c))
            for i,c in enumerate(d))))

        assert all(v for v in d.itervalues())
        return d,z3db
    
    @staticmethod
    def str_of_pack(pack):
        #c is a tuple
        def f(c):
            try:
                return c.vstr
            except Exception:
                return str(c)

        return '({})'.format('; '.join(map(f,pack)))

    
    @staticmethod
    def get_minset_f(mcores_d,remain_covs,f,dom):
        d,z3db = HighCov.indep(mcores_d.keys(),dom)

        st = time()
        ncovs = len(remain_covs)  #orig covs
        minset_d = CC.Configs_d()  #results
        
        for pack,expr in d.iteritems():
            #Todo: and not one of the existing ones
            nexpr = z3util.myOr(minset_d)            
            configs = dom.gen_configs_exprs([expr],[nexpr],k=1)
            if not configs:
                logger.warn("Cannot create configs from {}"
                            .format(HighCov.str_of_pack(pack)))
            else:
                config = configs[0]
                covs,xtime = f(config)
                remain_covs = remain_covs - covs
                minset_d[config]=covs

        minset_ncovs = ncovs-len(remain_covs)
        logger.info("minset: {} configs cover {}/{} sids (time {}s)"
                     .format(len(minset_d),
                             minset_ncovs,ncovs,time()-st))
        logger.debug('\n{}'.format(minset_d))                
        return minset_d.keys(), minset_ncovs
        
    
    @staticmethod
    def get_minset_configs_d(mcores_d,remain_covs,configs_d,dom):
        """
        Use interactions to generate a min set of configs 
        from configs_d that cover remain_covs locations.

        This essentially reuse generated configs. 
        If use packed elems then could be slow because
        have to check that all generated configs satisfy packed elem
        """
        d,z3db = HighCov.indep(mcores_d.keys(),dom)
        packs = set(d)  #{(pack,expr)}
        ncovs = len(remain_covs)  #orig covs
        remain_configs = set(configs_d)
        minset_d = CC.Configs_d()  #results
        
        #some init setup
        exprs_d = [(c,c.z3expr(z3db)) for c in configs_d] #slow
        exprs_d = exprs_d + d.items()
        exprs_d = dict(exprs_d)

        def _f(pack,covs):
            covs_ = set.union(*(mcores_d[c] for c in pack))
            return len(covs - covs_)  #smaller better

        def _g(pack,e_configs):
            """
            Select a config from e_configs that satisfies c_expr
            WARNING: DOES NOT WORK WITH OTTER's PARTIAL CONFIGS
            """
            e_expr = z3util.myOr([exprs_d[c] for c in e_configs])
            p_expr = exprs_d[pack]
            expr = z3.And(p_expr,e_expr)
            configs = dom.gen_configs_expr(expr,k=1)
            if not configs:
                return None
            else:
                config = configs[0]
                assert config in e_configs,\
                    ("ERR: gen nonexisting config {})"\
                     "Do not use Otter's full-run programs".format(config))
                 
                return config

        def _get_min(configs,covs):
            return min(configs,key=lambda c: len(covs - configs_d[c]))

        st = time()
        while remain_covs and remain_configs:
            #create a config
            if packs:
                pack = min(packs,key=lambda p: _f(p,remain_covs))
                packs.remove(pack)
                
                config = _g(pack,remain_configs)
                if config is None:
                    logger.warn("none of the avail configs => {}"
                                .format(HighCov.str_of_pack(pack)))
                    config = _get_min(remain_configs,remain_covs)
            else:
                config = _get_min(remain_configs,remain_covs)                

            assert config and config not in minset_d, config
            minset_d[config]=configs_d[config]
            remain_covs = remain_covs - configs_d[config]
            remain_configs = [c for c in remain_configs
                              if len(remain_covs - configs_d[c])
                              != len(remain_covs)]

        minset_ncovs = ncovs - len(remain_covs)
        logger.debug("minset: {} configs cover {}/{} sids (time {}s)"
                     .format(len(minset_d),
                             minset_ncovs, ncovs, time()-st))
        logger.detail('\n{}'.format(minset_d))

        return minset_d.keys(), minset_ncovs
        
class Metrics(object):
    @staticmethod
    def vscore_core(core,dom):
        vs = [len(core[k] if k in core else dom[k]) for k in dom]
        #print core,vs,sum(vs)
        return sum(vs)

    @staticmethod
    def vscore_pncore(pncore,dom):
        #"is not None" is correct because empty Core is valid
        #and in fact has max score
        vs = [Metrics.vscore_core(c,dom) for c in pncore
              if c is not None]
        #print "pncore nnew", pncore, sum(vs)
        return sum(vs)

    @staticmethod
    def vscore_cores_d(cores_d,dom):
        vs = [Metrics.vscore_pncore(c,dom) for c in cores_d.itervalues()]
        return sum(vs)

    #f-score    
    @staticmethod
    def get_settings(c,dom):
        """
        If a var k is not in c, then that means k = all_poss_of_k
        """
        if c is None:
            settings = set()
        elif isinstance(c,IA.Core):
            #if a Core is empty then it will have max settings
            rs = [k for k in dom if k not in c]
            rs = [(k,v) for k in rs for v in dom[k]]
            settings = set(c.settings + rs)
        else:
            settings = c
        return settings
        
    @staticmethod    
    def fscore_core(me,gt,dom):
        def fscore(me,gt):
            """
            #false pos: in me but not in gt
            #true pos: in me that in gt
            #false neg: in gt but not in me
            #true neg: in gt and also not in me
            """
            tp = len([s for s in me if s in gt])
            fp = len([s for s in me if s not in gt])
            fn = len([s for s in gt if s not in me])
            return tp,fp,fn
        me = Metrics.get_settings(me,dom)
        gt = Metrics.get_settings(gt,dom)
        return fscore(me,gt)

    @staticmethod
    def fscore_pncore(me,gt,dom):
        rs = [Metrics.fscore_core(m,g,dom) for m,g in zip(me,gt)]
        tp = sum(x for x,_,_ in rs)
        fp = sum(x for _,x,_ in rs)
        fn = sum(x for _,_,x in rs)
        return tp,fp,fn

    @staticmethod
    def fscore_cores_d(me,gt,dom):
        """
        Precision(p) = tp / (tp+fp)
        Recall(r) = tp / (tp+fn)
        F-score(f) = 2*p*r / (p+r) 1: identical ... 0 completely diff
        

        >>> pn1=["a","b"]
        >>> pd1=["c"]
        >>> nc1=[]
        >>> nd1=[]
        
        >>> pn2=["c"]
        >>> pd2=["d","e", "f"]
        >>> nc2=[]
        >>> nd2=[]
   
        >>> pn3=["a","b", "c"]
        >>> pd3=["c"]
        >>> nc3=[]
        >>> nd3=[]
        
        >>> pn4=["c"]
        >>> pd4=["d","e"]
        >>> nc4=[]
        >>> nd4=[]
        
        >>> pn5=["c","a"]
        >>> pd5=["d"]
        >>> nc5=["f"]
        >>> nd5=["g"]
        
        >>> pn6=["a"]
        >>> pd6=["e"]
        >>> nc6=["u"]
        >>> nd6=["v"]

        >>> me = {'l1': [pn1,pd1,nc1,nd1], 'l2': [pn2,pd2,nc2,nd2], 'l5': [pn5,pd5,nc5,nd5]}
        >>> gt={'l1': [pn3,pd3,nc3,nd3], 'l2': [pn4,pd4,nc4,nd4], 'l6': [pn6,pd6,nc6,nd6]}
        >>> Metrics.fscore_cores_d(me,gt,{})
        0.5714285714285714
        """
        rs = [Metrics.fscore_pncore(me[k],gt[k],dom)
              for k in gt if k in me]
        f_sum = 0.0        
        for tp,fp,fn in rs:
            if fp==fn==0:  #if these are 0 then me = gt
                f=1.0
            else:
                p=0.0 if (tp+fp)==0 else (float(tp)/(tp+fp))
                r=0.0 if (tp+fn)==0 else (float(tp)/(tp+fn))
                f=0.0 if (r+p)==0 else float(2*r*p)/(r+p)
            f_sum += f
        return f_sum/len(gt)

    @staticmethod
    def get_scores(dts,dom,cmp_gt):
        #Evol scores
        
        #Using f-score when ground truth is avail
        #Note here we use analyzed pp_cores_d instead of just cores_d
        if cmp_gt:
            gt_dir = cmp_gt
            logger.debug("load gt dir '{}'".format(gt_dir))            
            _,_,gt_dts,gt_pp_cores_d,_ = IA.DTrace.load_dir(gt_dir)
            assert len(gt_dts)==1, "is this the ground truth dir ??"

            #get pp_cores_d at each iteration
            pp_cores_ds = []
            for dt in dts:
                configs_ds = [dt_.cconfigs_d for dt_ in dts
                              if dt_.citer <= dt.citer]
                covs_d = CC.Covs_d()
                for configs_d in configs_ds:
                    for config,cov in configs_d.iteritems():
                        for sid in cov:
                            covs_d.add(sid,config)

                pp_cores_d = dt.cores_d.analyze(dom,covs_d)
                pp_cores_ds.append(pp_cores_d)
            fscores = [
                (dt.citer,
                 Metrics.fscore_cores_d(pp_cores_d,gt_pp_cores_d,dom),
                 dt.nconfigs)
                for dt, pp_cores_d in zip(dts,pp_cores_ds)]
            logger.info("fscores (iter, fscore, configs): {}".format(
            ' -> '.join(map(str,fscores))))
        else:
            gt_pp_cores_d=None
            fscores = None
            
        #Using  a simple metrics that just gives more pts for more
        #info (new cov or new results). Note here we use cores_d
        vscores = [(dt.citer,
                    Metrics.vscore_cores_d(dt.cores_d,dom),
                    dt.nconfigs)
                   for dt in dts]
        logger.info("vscores (iter, vscore, configs): {}".format(
            ' -> '.join(map(str,vscores))))

        return fscores,vscores,gt_pp_cores_d
    
class Influence(object):
    @staticmethod
    def get_influence(mcores_d,ncovs,dom,do_settings=True):
        if __debug__:
            assert (mcores_d and
                    isinstance(mcores_d,IA.Mcores_d)), mcores_d
            assert ncovs > 0, ncovs
            assert isinstance(dom,IA.Dom), dom            

        if do_settings:
            ks = set((k,v) for k,vs in dom.iteritems() for v in vs)
            def g(core):
                pc,pd,nc,nd = core
                settings = []
                if pc:
                    settings.extend(pc.settings)
                if pd:
                    settings.extend(pd.neg(dom).settings)
                #nd | neg(nc) 
                if nc:
                    settings.extend(nc.neg(dom).settings)
                if nd:
                    settings.extend(nd.settings)

                return set(settings)

            _str = CC.str_of_setting

        else:
            ks = dom.keys()
            def g(core):
                core = (c for c in core if c)
                return set(s for c in core for s in c)
            _str = str


        g_d = dict((pncore,g(pncore)) for pncore in mcores_d)
        rs = []
        for k in ks:
            v = sum(len(mcores_d[pncore])
                    for pncore in g_d if k in g_d[pncore])
            rs.append((k,v))
            
        rs = sorted(rs,key=lambda (k,v):(v,k),reverse=True)
        rs = [(k,v,float(v)/ncovs) for k,v in rs]
        logger.info("influence opts (opts, uniq, %) {}"
                    .format(', '.join(map(
                        lambda (k,v,p): "({}, {}, {})"
                        .format(_str(k),v,p),rs))))
        return rs

                    
            
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
