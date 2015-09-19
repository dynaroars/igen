import os.path
import numpy
import vu_common as CM
import itertools
from time import time

logger = CM.VLog('analysis')
logger.level = CM.VLog.DEBUG
CM.VLog.PRINT_TIME = True

class DTrace(object):
    """
    Object for saving information (for later analysis)
    """
    def __init__(self,citer,itime,xtime,
                 nconfigs,ncovs,ncores,
                 cconfigs_d,new_covs,new_cores,
                 sel_core,cores_d):

        self.citer = citer
        self.itime = itime
        self.xtime = xtime
        self.nconfigs = nconfigs
        self.ncovs = ncovs
        self.ncores = ncores
        self.cconfigs_d = cconfigs_d
        self.new_covs = new_covs
        self.new_cores = new_cores
        self.sel_core = sel_core
        self.cores_d = cores_d
        
    def show(self):
        logger.info("ITER {}, ".format(self.citer) +
                    "{}s, ".format(self.itime) +
                    "{}s eval, ".format(self.xtime) +
                    "total: {} configs, {} covs, {} cores, "
                    .format(self.nconfigs,self.ncovs,self.ncores) +
                    "new: {} configs, {} covs, {} updated cores, "
                    .format(len(self.cconfigs_d),
                            len(self.new_covs),len(self.new_cores)) +
                    "{}".format("** progress **"
                                if self.new_covs or self.new_cores else ""))

        logger.debug('select core: ({}) {}'.format(self.sel_core.sstren,
                                                   self.sel_core))
        logger.debug('create {} configs'.format(len(self.cconfigs_d)))
        logger.detail("\n"+str(self.cconfigs_d))
        mcores_d = self.cores_d.merge()
        logger.debug("infer {} interactions".format(len(mcores_d)))
        logger.detail('\n{}'.format(mcores_d))
        logger.info("strens: {}".format(mcores_d.strens_str))



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
    def replay(dir_,show_iters,do_min_configs):
        """
        Replay execution info from saved info in dir_
        """
        logger.info("replay dir: '{}'".format(dir_))
        seed,dom,dts,pp_cores_d,itime_total = Analysis.load_dir(dir_)
        logger.info('seed: {}'.format(seed))
        logger.debug(dom.__str__())

        dts = sorted(dts,key=lambda dt : dt.citer)        
        if show_iters:
            for dt in dts:
                dt.show()
                logger.debug("evol score: {}".format(
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

        if do_min_configs:
            #reconstruct information
            from config import Configs_d
            configs_d = Configs_d()
            covs = set()
            for dt in dts:
                for c in dt.cconfigs_d:
                    configs_d[c] = dt.cconfigs_d[c]
                    
                for sid in dt.new_covs:
                    covs.add(sid)

            min_configs = HighCov.get_min_configs(
                mcores_d,covs,configs_d,dom)
            n_min_configs = len(min_configs)
        else:
            n_min_configs = 0

            
        evol_scores = [Metrics.score_cores_d(dt.cores_d,dom) for dt in dts]
        evol_scores = [(i,a,a-b) for i,(a,b) in enumerate(zip(evol_scores,[0]+evol_scores))]
        logger.info("evol scores: {}".format(evol_scores))
        
        return (len(dts),len(mcores_d),
                itime_total,xtime_total,
                nconfigs,ncovs,
                n_min_configs,
                evol_scores,
                mcores_d.strens,mcores_d.strens_str,mcores_d.vtyps)

    @staticmethod
    def replay_dirs(dir_,show_iters,do_min_configs,do_metrics):
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
             evol_scores,
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


        # def str_of_strens(strens):
        #     return ', '.join("({}, {}, {})".format(siz,ncores,ncov)
        #                      for siz,ncores,ncov in strens)

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
    def str_of_pack(pack):
        #c is a tuple
        def f(c):
            try:
                return c.vstr
            except Exception:
                return str(c)

        return '({})'.format('; '.join(map(f,pack)))

    @staticmethod
    def gen_configs(d,dom):
        """
        Use an SMT solver to generate |d| configurations
        """
        if CM.__vdebug__:
            assert all(isinstance(k,tuple) and
                       z3.is_expr(v) for k,v in d.itertiems()),d
        
        z3db = dom.z3db

    @staticmethod
    def get_min_configs(mcores_d,remain_covs,configs_d,dom):
        """
        Use interactions to generate a min set of configs 
        from configs_d that cover *all* remain_covs locations.

        This essentially reuse generated configs. 
        If use packed elems then could be slow because
        have to check that all generated configs satisfy packed elem
        """
        z3db = dom.z3db

        #prune
        d = dict((c,c.z3expr(z3db,dom)) for c in mcores_d)
        d = HighCov.prune(d)
        logger.debug("prune: {} indep cores".format(len(d)))
        logger.detail("\n{}".format('\n'.join(
            "{}. {}".format(i+1,str(c)) for i,c
            in enumerate(sorted(d)))))

        #pack
        d = HighCov.pack(d)
        logger.debug("{} packed cores".format(len(d)))
        logger.detail("\n{}".format('\n'.join(
            "{}. {}".format(i+1,HighCov.str_of_pack(c))
            for i,c in enumerate(d))))

        assert all(v for v in d.itervalues())
        
        #some init setup
        exprs_d = [(c,c.z3expr(z3db)) for c in configs_d] #slow
        exprs_d = exprs_d + d.items()
        exprs_d = dict(exprs_d)

        packs = set(d)  #{(pack,expr)}
        ncovs = len(remain_covs)  #orig covs
        remain_configs = set(configs_d)

        from config import Configs_d
        minset_d = Configs_d()  #results

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
                              if len(remain_covs - configs_d[c]) != len(remain_covs)]

            
        logger.debug("minset: {} configs cover {}/{} sids (time {}s)"
                     .format(len(minset_d),
                             ncovs-len(remain_covs),ncovs,time()-st))
        logger.detail('\n{}'.format(minset_d))

        return minset_d.keys()
        


class Metrics(object):
    @staticmethod
    def score_core(core,dom):
        vs = []
        for k in dom:
            if k in core:
                vs.append(len(core[k]))
            else:
                vs.append(len(dom[k]))

        return sum(vs)

    @staticmethod
    def score_pncore(pncore,dom):
        #"is not None" is correct because empty Core is valid and in fact has max score
        vs = [Metrics.score_core(c,dom) for c in pncore if c is not None] 
        return sum(vs)

    @staticmethod
    def score_cores_d(cores_d,dom):
        vs = [Metrics.score_pncore(c,dom) for c in cores_d.itervalues()]
        return sum(vs)

    
    
        
