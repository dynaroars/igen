"""
Analyses on the resulting interactions
"""
import abc
import itertools
from time import time
import vu_common as CM
import config_common as CC
import alg as IA
import config as IA_OLD  #old data

logger = CM.VLog('alg_miscs')
logger.level = CC.logger_level
CM.VLog.PRINT_TIME = True

import z3
import z3util

class AddOns(object):
    __metaclass__ = abc.ABCMeta
    
    #init vars
    dts = None
    dom = None
    mcores_d = None
    pp_cores_d = None
    z3db = None
    
    #to be computed
    configs_d = CC.Configs_d()  #{config -> set(cov)}
    ccovs_d = CC.Covs_d()  #{cov -> set(configs)}
    ncovs_d = CC.Covs_d()  #{cov -> set(configs)}
    
    @classmethod
    def compute_configs_d(cls):
        if cls.configs_d:
            return

        assert isinstance(cls.configs_d, CC.Configs_d) and not cls.configs_d
        assert cls.dts
        
        for dt in cls.dts:
            for config in dt.cconfigs_d:
                cls.configs_d[config] = dt.cconfigs_d[config]
                
    @classmethod
    def compute_covs_ds(cls):
        if cls.ccovs_d or cls.ncovs_d:
            return

        assert isinstance(cls.configs_d, CC.Configs_d) and cls.configs_d        
        assert isinstance(cls.ccovs_d, CC.Covs_d) and not cls.ccovs_d
        assert isinstance(cls.ncovs_d, CC.Covs_d) and not cls.ncovs_d

        covs = set()
        for config in cls.configs_d:
            for cov in cls.configs_d[config]:
                covs.add(cov)

        for config in cls.configs_d:
            covered = cls.configs_d[config]            
            for loc in covered:
                cls.ccovs_d.add(loc, config)

            ncovered = covs - covered                
            for loc in ncovered:
                cls.ncovs_d.add(loc, config)
                
    # @staticmethod
    # def compute_same_covs(covs_d):
    #     d = {}
    #     for loc, configs in covs_d.iteritems():
    #         k = frozenset(configs)
    #         if k not in d:
    #             d[k] = set()
    #         d[k].add(loc)

    #     rs = set(frozenset(s) for s in d.itervalues())
    #     return rs
        
        
class HighCov(AddOns):
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
    {}

    >>> HighCov.pack({})
    {}

    >>> HighCov.pack({'a':None})
    Traceback (most recent call last):
    ...
    AssertionError: {'a': None}
    
    """

    @classmethod
    def go(cls, do_min_configs):
        assert do_min_configs is not None

        cores = set(cls.pp_cores_d)
        if callable(do_min_configs):
            return cls.get_minset_f(cores, f=do_min_configs)
        else:
            cls.compute_configs_d()
            return cls.get_minset_configs_d(cores)

        return min_configs, min_ncovs
    
    @classmethod
    def prune(cls, d):
        """
        Ret the strongest elements by removing those implied by others
        """
        assert d and isinstance(d, dict)
        assert all(v is None or z3.is_expr(v) for v in d.itervalues()), d
        
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
                    key=lambda f: _len(d[f]), reverse=True)

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

    @classmethod
    def pack2(cls, fs, d):
        assert all(isinstance(f,tuple) for f in fs),fs
        assert all(f in d and z3.is_expr(d[f]) for f in fs), (fs, d)
                   
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
                
    @classmethod
    def pack(cls, d):
        """
        Pack together elements that have no conflicts.
        The results are {tuple -> z3expr}
        It's good to first prune them (call prune()).
        """
        assert all(z3.is_expr(v) for v in d.itervalues()), d
                       
        #change format, results are tuple(elems)
        d = dict((tuple([f]),d[f]) for f in d)
        fs = d.keys()
        fs_ = cls.pack2(fs,d)        
        while len(fs_) < len(fs):
            fs = fs_
            fs_ = cls.pack2(fs,d)

        fs = set(fs_)
        return dict((f,d[f]) for f in d if f in fs_)

    @classmethod
    def indep(cls, fs):
        """
        Apply prune and pack on fs
        """
        #prune
        d = dict((c, c.z3expr(cls.z3db, cls.dom)) for c in fs)
        d = cls.prune(d)
        logger.debug("prune: {} remains".format(len(d)))
        logger.debug("\n{}".format('\n'.join(
            "{}. {}".format(i+1,str(c)) for i,c
            in enumerate(sorted(d)))))

        #pack
        d = cls.pack(d)
        logger.debug("pack: {} remains".format(len(d)))
        logger.debug("\n{}".format('\n'.join(
            "{}. {}".format(i+1, cls.str_of_pack(c))
            for i,c in enumerate(d))))

        assert all(v for v in d.itervalues())
        return d
    
    @classmethod
    def str_of_pack(cls, pack):
        #c is a tuple
        def f(c):
            try:
                return c.vstr
            except Exception:
                return str(c)

        return '({})'.format('; '.join(map(f,pack)))

    
    @classmethod
    def get_minset_f(cls, remain_covs, f):
        d = cls.indep(cls.mcores_d.keys())

        st = time()
        ncovs = len(remain_covs)  #orig covs
        minset_d = CC.Configs_d()  #results
        
        for pack,expr in d.iteritems():
            #Todo: and not one of the existing ones
            nexpr = z3util.myOr(minset_d)            
            configs = cls.dom.gen_configs_exprs(
                [expr], [nexpr], k=1, config_cls=IA.Config)
                
            if not configs:
                logger.warn("Cannot create configs from {}"
                            .format(cls.str_of_pack(pack)))
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
        
    
    @classmethod
    def get_minset_configs_d(cls, remain_covs):
        """
        Use interactions to generate a min set of configs 
        from configs_d that cover remain_covs locations.

        This essentially reuses generated configs. 
        If use packed elems then could be slow because
        have to check that all generated configs satisfy packed elem
        """
        d = cls.indep(cls.mcores_d.keys())
        packs = set(d)  #{(pack,expr)}
        ncovs = len(remain_covs)  #orig covs
        remain_configs = set(cls.configs_d)
        minset_d = CC.Configs_d()  #results
        
        #some init setup
        exprs_d = [(c, c.z3expr(cls.z3db)) for c in cls.configs_d] #slow
        exprs_d = exprs_d + d.items()
        exprs_d = dict(exprs_d)

        def _f(pack,covs):
            covs_ = set.union(*(cls.mcores_d[c] for c in pack))
            return len(covs - covs_)  #smaller better

        def _g(pack,e_configs):
            """
            Select a config from e_configs that satisfies c_expr
            WARNING: DOES NOT WORK WITH OTTER's PARTIAL CONFIGS
            """
            e_expr = z3util.myOr([exprs_d[c] for c in e_configs])
            p_expr = exprs_d[pack]
            expr = z3.And(p_expr,e_expr)
            configs = cls.dom.gen_configs_expr(expr, k=1, config_cls=IA.Config)
            if not configs:
                return None
            else:
                config = configs[0]
                assert config in e_configs,\
                    ("ERR: gen nonexisting config {})"\
                     "Do not use Otter's full-run programs".format(config))
                 
                return config

        def _get_min(configs,covs):
            return min(configs,key=lambda c: len(covs - cls.configs_d[c]))

        st = time()
        while remain_covs and remain_configs:
            #create a config
            if packs:
                pack = min(packs,key=lambda p: _f(p,remain_covs))
                packs.remove(pack)
                
                config = _g(pack,remain_configs)
                if config is None:
                    logger.warn("no avail configs => {}"
                                .format(cls.str_of_pack(pack)))
                    config = _get_min(remain_configs,remain_covs)
            else:
                config = _get_min(remain_configs,remain_covs)                

            assert config and config not in minset_d, config
            minset_d[config] = cls.configs_d[config]
            remain_covs = remain_covs - cls.configs_d[config]
            remain_configs = [c for c in remain_configs
                              if len(remain_covs - cls.configs_d[c])
                              != len(remain_covs)]

        minset_ncovs = ncovs - len(remain_covs)
        logger.info("minset: {} configs cover {}/{} sids (time {}s)"
                    .format(len(minset_d),
                            minset_ncovs, ncovs, time()-st))
        logger.detail('\n{}'.format(minset_d))

        return minset_d.keys(), minset_ncovs

class Similarity(AddOns):
    @classmethod
    def vscore_core(cls, core):
        vs = [len(core[k] if k in core else cls.dom[k]) for k in cls.dom]
        return sum(vs)

    @classmethod
    def vscore_pncore(cls, pncore):
        #"is not None" is correct because empty Core is valid
        #and in fact has max score
        vs = [cls.vscore_core(c) for c in pncore if c is not None]
        return sum(vs)

    @classmethod
    def vscore_cores_d(cls, cores_d):
        vs = [cls.vscore_pncore(c) for c in cores_d.itervalues()]
        return sum(vs)

    #f-score    
    @classmethod
    def get_settings(cls, c):
        """
        If a var k is not in c, then that means k = all_poss_of_k
        """

        if c is None:
            settings = set()
        elif isinstance(c,IA.Core):
            #if a Core is empty then it will have max settings
            rs = [k for k in cls.dom if k not in c]
            rs = [(k,v) for k in rs for v in cls.dom[k]]
            settings = set(c.settings + rs)
        else:
            settings = c
        return settings
        
    @classmethod    
    def fscore_core(cls, me, gt):
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
        me = cls.get_settings(me)
        gt = cls.get_settings(gt)
        return fscore(me, gt)

    @classmethod
    def fscore_pncore(cls, me, gt):
        #me and gt are pncore's (c,d,cd,dc)
        rs = [cls.fscore_core(m, g) for m,g in zip(me, gt)]
        tp = sum(x for x,_,_ in rs)
        fp = sum(x for _,x,_ in rs)
        fn = sum(x for _,_,x in rs)
        return tp,fp,fn

    @classmethod
    def fscore_cores_d(cls, me, gt):
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
        >>> gt = {'l1': [pn3,pd3,nc3,nd3], 'l2': [pn4,pd4,nc4,nd4], 'l6': [pn6,pd6,nc6,nd6]}
        >>> Similarity.fscore_cores_d(me,gt,{})
        0.5714285714285714

        >>> me = {'l1': [pn1,pd1,nc1,nd1], 'l2': [pn2,pd2,nc2,nd2]}
        >>> gt = {'l1': [pn1,pd1,nc1,nd1], 'l2': [pn2,pd2,nc2,nd2], 'l6': [pn6,pd6,nc6,nd6]}
        >>> Similarity.fscore_cores_d(me,gt,{})
        0.6666666666666666

        >>> me = {'l1': [pn1,pd1,nc1,nd1], 'l2': [pn2,pd2,nc2,nd2]}
        >>> gt = {'l1': [pn1,pd1,nc1,nd1], 'l2': [pn2,pd2,nc2,nd2]}
        >>> Similarity.fscore_cores_d(me,gt,{})
        1.0

        """
        rs = [cls.fscore_pncore(me[k], gt[k])
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

    @classmethod
    def compare_gt(cls, cmp_gt):
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
            for dt in cls.dts:
                configs_ds = [dt_.cconfigs_d for dt_ in cls.dts
                              if dt_.citer <= dt.citer]
                covs_d = CC.Covs_d()
                for configs_d in configs_ds:
                    for config,cov in configs_d.iteritems():
                        for sid in cov:
                            covs_d.add(sid,config)

                pp_cores_d = dt.cores_d.analyze(cls.dom, covs_d)
                pp_cores_ds.append(pp_cores_d)

            fscores = [
                (dt.citer,
                 cls.fscore_cores_d(pp_cores_d,gt_pp_cores_d),
                 dt.nconfigs)
                for dt, pp_cores_d in zip(cls.dts, pp_cores_ds)]

            fscores_ = []
            for dt, pp_cores_d in zip(cls.dts, pp_cores_ds):
                rs = (dt.citer,
                      cls.fscore_cores_d(pp_cores_d, gt_pp_cores_d),
                      dt.nconfigs)
                fscores_.append(rs)

            logger.info("fscores (iter, fscore, configs): {}".format(
                ' -> '.join(map(str,fscores))))
            
        else:
            gt_pp_cores_d=None
            fscores = None
            
        #Using  a simple metrics that just gives more pts for more
        #info (new cov or new results). Note here we use cores_d
        vscores = [(dt.citer,
                    cls.vscore_cores_d(dt.cores_d),
                    dt.nconfigs)
                   for dt in cls.dts]
        logger.info("vscores (iter, vscore, configs): {}".format(
            ' -> '.join(map(str, vscores))))

        return fscores, vscores, gt_pp_cores_d
    
class Influence(AddOns):
    """
    Compute influence metrics of options. 
    Returns a dict of (opt, uniq, %), e.g., 
    opt z=4 appears in the interactions of 2 (uniq) locations, 
    which is 25% of the 8 covered locations.
    """
    
    @classmethod
    def go(cls, ncovs, do_settings=True):
        assert ncovs > 0, ncovs
        assert isinstance(do_settings, bool), do_settings
        
        if do_settings:
            ks = set((k,v) for k,vs in cls.dom.iteritems() for v in vs)
            def g(core):
                pc,pd,nc,nd = core
                settings = []
                if pc:
                    settings.extend(pc.settings)
                if pd:
                    settings.extend(pd.neg(cls.dom).settings)
                #nd | neg(nc) 
                if nc:
                    settings.extend(nc.neg(cls.dom).settings)
                if nd:
                    settings.extend(nd.settings)

                return set(settings)

            _str = CC.str_of_setting

        else:
            ks = cls.dom.keys()
            def g(core):
                core = (c for c in core if c)
                return set(s for c in core for s in c)
            _str = str


        g_d = dict((pncore, g(pncore)) for pncore in cls.mcores_d)
        rs = []
        for k in ks:
            v = sum(len(cls.mcores_d[pncore])
                    for pncore in g_d if k in g_d[pncore])
            rs.append((k,v))
            
        rs = sorted(rs, key = lambda (k, v) : (v, k), reverse=True)
        rs = [(k, v, 100. * v / ncovs) for k, v in rs]
        logger.info("influence opts (opt, uniq, %) {}"
                    .format(', '.join(map(
                        lambda (k, v, p): "({}, {}, {})"
                        .format(_str(k), v, p), rs))))
        return rs

    
class Undetermined(AddOns):
    """
    Determine potentially imprecise invariants (too weak).
    E.g., if a location has true invbut there exist some config 
    that does not cover it then true is too weak.

    Essentially this method can determine that an inferred inv of a loc 
    having the form conj1 & .. & conjn is too weak if the real inv of 
    that loc is conj1 & .. & conj2 & some_other_expr
    """
    @classmethod
    def go(cls):
        def check(configs, expr, cache):
            #check if forall c in configs. c => expr 
            k = hash((frozenset(configs), z3util.fhash(expr)))
            if k in cache: return cache[k]
            
            rs = []
            for config in configs:
                try:
                    cexpr = cache[config]
                except KeyError:
                    cexpr = config.z3expr(cls.z3db)
                    cache[config] = cexpr
                rs.append(cexpr)

            rs = z3util.is_tautology(z3.Implies(z3util.myOr(rs), expr))
            cache[k] = rs
            return rs

        cls.compute_configs_d()
        cls.compute_covs_ds()

        cache = {}        
        ok_mcores_d, weak_mcores_d = IA.Mcores_d(), IA.Mcores_d()
        for pncore in cls.mcores_d:
            if isinstance(pncore, IA_OLD.PNCore):
                pncore_ = IA.PNCore((
                    pncore.pc if pncore.pc is None else IA.Core(pncore.pc),
                    pncore.pd if pncore.pd is None else IA.Core(pncore.pd),
                    pncore.nc if pncore.nc is None else IA.Core(pncore.nc),
                    pncore.nd if pncore.nd is None else IA.Core(pncore.nd)))
                pncore_.vtyp = pncore.vtyp
                pncore = pncore_
                
            expr = pncore.z3expr(cls.z3db, cls.dom)  #None = True
            nexpr = expr if expr is None else z3.Not(expr)
            covs = cls.mcores_d[pncore]
            for cov in covs:
                if cov not in cls.ncovs_d:
                    isOK = True
                else:
                    if expr is None: #true interaction but some locs are not covered
                        isOK = False
                    else:
                        assert cls.ncovs_d[cov]
                        isOK = check(cls.ncovs_d[cov], nexpr, cache)

                (ok_mcores_d if isOK else weak_mcores_d).add(pncore, cov)
                
        return ok_mcores_d, weak_mcores_d

            
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
