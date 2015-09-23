import os.path
import numpy
import itertools
from time import time
import vu_common as CM

import config_common as CC
import config as CF

logger = CM.VLog('analysis')
logger.level = CF.logger.level
CM.VLog.PRINT_TIME = True

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
        
    @staticmethod
    def replay(dir_,show_iters,do_min_configs,cmp_gt):
        """
        Replay execution info from saved info in dir_
        do_min_configs has 3 possible values
        1. None: don't find min configs
        2. f: (callable(f)) find min configs using f
        3. anything else: find min configs using existing configs
        """
        logger.info("replay dir: '{}'".format(dir_))
        seed,dom,dts,pp_cores_d,itime_total = Analysis.load_dir(dir_)
        logger.info('seed: {}'.format(seed))
        logger.debug(dom.__str__())

        dts = sorted(dts,key=lambda dt: dt.citer)        
        if show_iters:
            for dt in dts:
                dt.show()
                logger.debug("evol pts: {}".format(
                    Metrics.score_cores_d(dt.cores_d,dom)))

        if hasattr(pp_cores_d.values()[0],'vstr'):
            mcores_d = pp_cores_d.merge(show_detail=True)            
        else:
            logger.warn("Old format, has no vstr .. re-analyze")
            pp_cores_d = pp_cores_d.analyze(dom,covs_d=None)
            mcores_d = pp_cores_d.merge(show_detail=True)
            
        #print summary
        xtime_total = itime_total - sum(dt.xtime for dt in dts)
        last_dt = max(dts,key=lambda dt: dt.citer) #last iteration
        nconfigs = last_dt.nconfigs
        ncovs = last_dt.ncovs
        
        logger.info(Analysis.str_of_summary(
            seed,len(dts),itime_total,xtime_total,nconfigs,ncovs,dir_))

        evol_scores = Metrics.get_evol_scores(dts,dom,cmp_gt)
        influence_scores = Influence.get_influence(mcores_d,ncovs,dom)

        #min config
        if not do_min_configs: #None
            n_min_configs = 0
        elif callable(do_min_configs):
            f = do_min_configs
            min_configs = HighCov.get_minset_f(
                mcores_d,set(pp_cores_d),f,dom)
            n_min_configs = len(min_configs)            
        else:
            #reconstruct information
            configs_d = CC.Configs_d()
            for dt in dts:
                for c in dt.cconfigs_d:
                    configs_d[c] = dt.cconfigs_d[c]

            min_configs = HighCov.get_minset_configs_d(
                mcores_d,set(pp_cores_d),configs_d,dom)
            n_min_configs = len(min_configs)

        return (len(dts),len(mcores_d),
                itime_total,xtime_total,
                nconfigs,ncovs,
                n_min_configs,
                evol_scores,
                influence_scores,
                mcores_d.strens,
                mcores_d.strens_str,
                mcores_d.vtyps)

    @staticmethod
    def replay_dirs(dir_,show_iters,do_min_configs,cmp_gt):
        dir_ = CM.getpath(dir_)
        logger.info("replay_dirs '{}'".format(dir_))
        
        niters_total = 0
        nresults_total = 0        
        nitime_total = 0
        nxtime_total = 0    
        nconfigs_total = 0
        ncovs_total = 0
        nminconfigs_total = 0
        strens_s = []
        strens_str_s = []
        ntyps_s = []

        #modified by ugur
        niters_arr = []
        nresults_arr = []
        nitime_arr = []
        nxtime_arr = [] 
        nconfigs_arr = []
        ncovs_arr = []
        nminconfigs_arr = []
        counter = 0
        csv_arr = []

        for rdir in sorted(os.listdir(dir_)):
            rdir = os.path.join(dir_,rdir)
            rs = Analysis.replay(rdir,show_iters,do_min_configs)
            (niters,nresults,itime,xtime,
             nconfigs,ncovs,n_min_configs,
             evol_scores,influence_scores,
             strens,strens_str,ntyps) = rs
            niters_total += niters
            nresults_total += nresults
            nitime_total += itime
            nxtime_total += xtime
            nconfigs_total += nconfigs
            ncovs_total += ncovs
            nminconfigs_total += n_min_configs
            strens_s.append(strens)
            strens_str_s.append(strens_str)
            ntyps_s.append(ntyps)

            niters_arr.append(niters)
            nresults_arr.append(nresults)
            nitime_arr.append(itime)
            nxtime_arr.append(xtime)
            nconfigs_arr.append(nconfigs)
            ncovs_arr.append(ncovs)
            nminconfigs_arr.append(n_min_configs)
            csv_arr.append("{},{},{},{},{},{},{},{},{},{}".format(
                counter,niters,nresults,itime,xtime,nconfigs,ncovs,n_min_configs,
                ','.join(map(str, ntyps)),','.join(map(str, strens))))
            counter += 1

        nruns_total = float(len(strens_s))

        ss = ["iter {}".format(niters_total/nruns_total),
              "results {}".format(nresults_total/nruns_total),
              "time {}".format(nitime_total/nruns_total),
              "xtime {}".format(nxtime_total/nruns_total),
              "configs {}".format(nconfigs_total/nruns_total),
              "covs {}".format(ncovs_total/nruns_total),
              "minconfigs {}".format(nminconfigs_total/nruns_total)]
        
        logger.info("STATS of {} runs (averages): {}".format(nruns_total,', '.join(ss)))
        
        ssMed = ["iter {}".format(numpy.median(niters_arr)),
                 "results {}".format(numpy.median(nresults_arr)),
                 "time {}".format(numpy.median(nitime_arr)),
                 "xtime {}".format(numpy.median(nxtime_arr)),
                 "configs {}".format(numpy.median(nconfigs_arr)),
                 "nminconfigs {}".format(numpy.median(nminconfigs_arr)),
                 "covs {}".format(numpy.median(ncovs_arr))]
        logger.info("STATS of {} runs (medians) : {}".format(nruns_total,', '.join(ssMed)))
        
        ssSIQR = ["iter {}".format(Analysis.siqr(niters_arr)),
                  "results {}".format(Analysis.siqr(nresults_arr)),
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
        if CM.__vdebug__:
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
        if CM.__vdebug__:
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
        if CM.__vdebug__:
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
                
        logger.info("minset: {} configs cover {}/{} sids (time {}s)"
                     .format(len(minset_d),
                             ncovs-len(remain_covs),ncovs,time()-st))
        logger.debug('\n{}'.format(minset_d))                
        return minset_d.keys()
        
    
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

            
        logger.debug("minset: {} configs cover {}/{} sids (time {}s)"
                     .format(len(minset_d),
                             ncovs-len(remain_covs),ncovs,time()-st))
        logger.detail('\n{}'.format(minset_d))

        return minset_d.keys()
        
    
class Metrics(object):
    @staticmethod
    def score_core(core,dom):
        vs = (len(core[k] if k in core else dom[k]) for k in dom)
        return sum(vs)

    @staticmethod
    def score_pncore(pncore,dom):
        #"is not None" is correct because empty Core is valid and in fact has max score
        vs = [Metrics.score_core(c,dom) for c in pncore
              if c is not None] 
        return sum(vs)

    @staticmethod
    def score_cores_d(cores_d,dom):
        vs = [Metrics.score_pncore(c,dom) for c in cores_d.itervalues()]
        return sum(vs)

    #f-score    
    @staticmethod
    def get_settings(c,dom):
        """
        If a var k is not in c, then that means k = all_poss_of_k
        """
        if c is None:
            settings = set()
        elif isinstance(c,CF.Core):
            #if a Core is empty then it will have max settings
            rs = [k for k in dom if k not in c]
            rs = [(k,v) for k in rs for v in dom[k]]
            settings = set(c.settings + rs)
        else:
            settings = c
        return settings
    
    @staticmethod
    def fscore(me,gt):
        """
        #false pos: in me but not in gt
        #true pos: in me that in gt
        #false neg: in gt but not in me
        #true neg: in gt and also not in me
        """
        if CM.__vdebug__:
            assert me is not None,me
            assert gt is not None,gt

        tp = len([s for s in me if s in gt])
        fp = len([s for s in me if s not in gt])
        fn = len([s for s in gt if s not in me])
        return tp,fp,fn
    
    @staticmethod
    def fscore_core(me,gt,dom):
        me = Metrics.get_settings(me,dom)
        gt = Metrics.get_settings(gt,dom)
        return Metrics.fscore(me,gt)

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
            p=0.0 if (tp+fp)==0 else (float(tp)/(tp+fp))
            r=0.0 if (tp+fn)==0 else (float(tp)/(tp+fn))
            f=0.0 if (r+p)==0 else float(2*r*p)/(r+p)
            f_sum += f
        return f_sum/len(gt)

    @staticmethod
    def get_evol_scores(dts,dom,cmp_gt):
        if CM.__vdebug__:
            assert all(isinstance(t,CF.DTrace) for t in dts), dts
            assert isinstance(dom,CF.Dom), dom
            assert (cmp_gt is None or isinstance(cmp_gt,str)), cmp_gt

        if cmp_gt:
            assert os.path.isdir(cmp_gt),cmp_gt
            gt_dir = cmp_gt
            logger.debug("load gt dir '{}'".format(gt_dir))
            #Note: I actually want to compare against cores_d before
            #postprocessing
            _,_,gt_dts,_,_ = Analysis.load_dir(gt_dir)
            assert len(gt_dts)==1
            gt_cores_d = gt_dts[0].cores_d

            _f = lambda cores_d: Metrics.fscore_cores_d(
                cores_d,gt_cores_d,dom)
        else:
            _f = lambda _: 1

        scores = [(dt.citer,
                   Metrics.score_cores_d(dt.cores_d,dom),
                   _f(dt.cores_d),
                   len(dt.cores_d))
                  for dt in dts]

        if CM.__vdebug__:
            assert all(v >= 0 and f>=0 and c >= 0
                       for _,v,f,c in scores), scores

        logger.info("evol scores (iter, vscore, fscore, configs): {}".format(
            ' -> '.join(map(str,scores))))
        return scores
    
class Influence(object):
    @staticmethod
    def get_influence(mcores_d,ncovs,dom,do_settings=True):
        if CM.__vdebug__:
            assert (mcores_d and
                    isinstance(mcores_d,CF.Mcores_d)), mcores_d
            assert isinstance(dom,CF.Dom), dom            

        if do_settings:
            ks = set((k,v) for k,vs in dom.iteritems() for v in vs)
            def g(core):
                pc,pd,nc,nd = core
                settings = []
                if pc:
                    settings.extend(pc.settings)
                if pd:
                    settings.extend(pc.neg(dom).settings)
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
        logger.debug("influence opts (opts, uniq, %) {}"
                     .format(', '.join(map(
                         lambda (k,v,p): "({}, {}, {})"
                         .format(_str(k),v,p),rs))))
        return rs

                    
            
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
