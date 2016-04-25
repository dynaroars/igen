import itertools
from time import time
import vu_common as CM
import config_common as CC
import alg as IA

logger = CM.VLog('alg_miscs')
logger.level = CC.logger_level
CM.VLog.PRINT_TIME = True

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
        logger.debug("prune: {} remains".format(len(d)))
        logger.debug("\n{}".format('\n'.join(
            "{}. {}".format(i+1,str(c)) for i,c
            in enumerate(sorted(d)))))

        #pack
        d = HighCov.pack(d)
        logger.debug("pack: {} remains".format(len(d)))
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
            configs = dom.gen_configs_exprs(
                [expr], [nexpr], k=1, config_cls=IA.Config)
                
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
            configs = dom.gen_configs_expr(expr, k=1, config_cls=IA.Config)
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
        logger.info("minset: {} configs cover {}/{} sids (time {}s)"
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
        #me and gt are pncore's (c,d,cd,dc)
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
        >>> gt = {'l1': [pn3,pd3,nc3,nd3], 'l2': [pn4,pd4,nc4,nd4], 'l6': [pn6,pd6,nc6,nd6]}
        >>> Metrics.fscore_cores_d(me,gt,{})
        0.5714285714285714

        >>> me = {'l1': [pn1,pd1,nc1,nd1], 'l2': [pn2,pd2,nc2,nd2]}
        >>> gt = {'l1': [pn1,pd1,nc1,nd1], 'l2': [pn2,pd2,nc2,nd2], 'l6': [pn6,pd6,nc6,nd6]}
        >>> Metrics.fscore_cores_d(me,gt,{})
        0.6666666666666666

        >>> me = {'l1': [pn1,pd1,nc1,nd1], 'l2': [pn2,pd2,nc2,nd2]}
        >>> gt = {'l1': [pn1,pd1,nc1,nd1], 'l2': [pn2,pd2,nc2,nd2]}
        >>> Metrics.fscore_cores_d(me,gt,{})
        1.0

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

            fscores_ = []
            for dt, pp_cores_d in zip(dts,pp_cores_ds):
                rs = (dt.citer,
                      Metrics.fscore_cores_d(pp_cores_d,gt_pp_cores_d,dom),
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
                    isinstance(mcores_d, IA.Mcores_d)), mcores_d
            assert ncovs > 0, ncovs
            assert isinstance(dom, IA.Dom), dom            

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
        logger.info("influence opts (opt, uniq, %) {}"
                    .format(', '.join(map(
                        lambda (k,v,p): "({}, {}, {})"
                        .format(_str(k),v,p),rs))))
        return rs

    

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
