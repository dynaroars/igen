import os.path
import vu_common as CM
import config_common as CC
import math
import alg as IA
import alg_igen

logger = CM.VLog('analysis')
logger.level = CC.logger_level
CM.VLog.PRINT_TIME = True


#Python implementation of the median and percentile functions
#so that I don't have to use numpy.median in Python 2
def percentile(l, percent, key=lambda x:x):
    """
    http://code.activestate.com/recipes/511478/ 
    Find the percentile of a list of values.
    @parameter percent - a float value from 0.0 to 1.0.
    @parameter key - optional key function to compute value from each element of l.
    @return - the percentile of the values
    """
    if not l:
        return None
    l = sorted(l)    
    k = (len(l)-1) * percent
    f = math.floor(k)
    c = math.ceil(k)
    if f == c:
        return key(l[int(k)])
    d0 = key(l[int(f)]) * (c-k)
    d1 = key(l[int(c)]) * (k-f)
    return d0 + d1

median = lambda l: percentile(l, percent=0.5)
siqr = lambda l: (percentile(l, 75) - percentile(l, 25)) / 2
    

class LoadData(object):
    data = {}
    
    def __init__(self, seed, dom, dts, pp_cores_d, itime_total):
        self._seed = seed
        self._dom = dom
        self._dts = dts
        self._pp_cores_d = pp_cores_d
        self._itime_total = itime_total

    @property
    def seed(self): return self._seed
    @property
    def dom(self): return self._dom
    @property
    def dts(self): return self._dts
    @property
    def last_dt(self):
        return max(self.dts, key=lambda dt: dt.citer) #last iteration
        
    @property
    def pp_cores_d(self): return self._pp_cores_d
    @pp_cores_d.setter
    def pp_cores_d(self, d):
        assert IA.compat(d, IA.Cores_d)
        self._pp_cores_d = d
    @property
    def itime_total(self): return self._itime_total

    @property
    def mcores_d(self): return self._mcores_d
    @mcores_d.setter
    def mcores_d(self, d):
        assert IA.compat(d, IA.Mcores_d)
        self._mcores_d = d

    #data computed on demand 
    @property
    def z3db(self):
        try:
            return self._z3db
        except AttributeError:
            self._z3db = CC.Z3DB(self.dom)
            return self._z3db

    @property
    def configs_d(self):
        try:
            return self._configs_d
        except AttributeError:
            assert self.dts
            configs_d = CC.Configs_d()
            for dt in self.dts:
                for config in dt.cconfigs_d:
                    configs_d[config] = dt.cconfigs_d[config]

            self._configs_d = configs_d
            return self._configs_d

    @property
    def covs(self):
        try:
            return self._covs
        except AttributeError:
            covs = set()
            for config in self.configs_d:
                for cov in self.configs_d[config]:
                    covs.add(cov)
            self._covs = covs
            return self._covs

    @property
    def ccovs_d(self):
        try:
            return self._ccovs_d
        except AttributeError:
            ccovs_d = CC.Covs_d()
            for config in self.configs_d:
                covered = self.configs_d[config]            
                for loc in covered:
                    ccovs_d.add(loc, config)
            self._ccovs_d = ccovs_d
            return self._ccovs_d

    @property
    def ncovs_d(self):
        try:
            return self._ncovs_d
        except AttributeError:
            ncovs_d = CC.Covs_d()
            for config in self.configs_d:
                covered = self.configs_d[config]
                ncovered = self.covs - covered
                for loc in ncovered:
                    ncovs_d.add(loc, config)    
            self._ncovs_d = ncovs_d
            return self._ncovs_d

    @classmethod
    def load_dir(cls, dir_):
        assert dir_
        if dir_ in cls.data:
            return cls.data[dir_]
        else:
            seed, dom, dts, pp_cores_d, itime_total = alg_igen.DTrace.load_dir(dir_)
            ld = LoadData(seed,dom,dts,pp_cores_d,itime_total)
            cls.data[dir_] = ld
            return ld

    @classmethod
    def load_cmp_dir(cls, dir_):
        ld = cls.load_dir(dir_)
        assert len(ld.dts) == 1, "correct cmp dir??"
        return ld
    
from collections import namedtuple
fields = ['niters',
          'ncores',
          'itime',
          'xtime',
          'ncovs',          
          'nconfigs',
          'n_minconfigs',
          'min_ncovs',
          'gt_fscore',
          'rd_fscore',
          'influences',
          'm_strens',
          'm_strens_str',
          'm_vtyps']
AnalysisResults = namedtuple("AnalysisResults",' '.join(fields))

class Analysis(object):
    NOTDIR = 0
    RUNDIR = 1
    BENCHMARKDIR = 2

    @classmethod
    def get_dir_stat(cls, d):
        """
        ret RUNDIR if d is a run_dir that consists of *.tvn, pre, post files
        ret BECHMARKDIR if d is a benchmark dir that consists of run_dirs
        ret NOTDIR otherwise
        """
        assert os.path.isdir(d), d
            
        fs = os.listdir(d)
        if (fs.count('pre') == 1 and fs.count('post') == 1 and
            any(f.endswith('.tvn') for f in fs)):
            return cls.RUNDIR
        else:
            ds = [os.path.join(d,f) for f in fs]
            if (ds and
                all(os.path.isdir(d_) and
                    cls.get_dir_stat(d_) == cls.RUNDIR for d_ in ds)):
                return cls.BENCHMARKDIR
            else:
                return cls.NOTDIR

    @classmethod
    def replay(cls, dir_, show_iters,
               do_minconfigs,
               do_influence,
               do_evolution,
               do_precision,
               cmp_rand, cmp_dir):
        """
        Replay and analyze execution info from saved info in dir_

        """
        assert isinstance(dir_,str), dir_
        assert isinstance(show_iters, bool), show_iters
        assert isinstance(do_influence, bool), do_influence
        assert isinstance(do_evolution, bool), do_evolution
        assert isinstance(do_precision, bool), do_precision
        assert cmp_dir is None or isinstance(cmp_dir, str), cmp_dir
        assert cmp_rand is None or callable(cmp_rand), cmp_rand


        logger.info("replay dir: '{}'".format(dir_))
        ld = LoadData.load_dir(dir_)
        
        logger.info('seed: {}'.format(ld.seed))
        logger.debug(ld.dom.__str__())

        ld.dts.sort(key=lambda dt: dt.citer)        
        if show_iters:
            for dt in ld.dts:
                dt.show(ld.z3db, ld.dom)

        if not hasattr(ld.pp_cores_d.values()[0], 'vstr'):
            logger.warn("Old format, has no vstr .. re-analyze")
            ld.pp_cores_d = ld.pp_cores_d.analyze(ld.dom, ld.z3db, covs_d=None)

        ld.mcores_d = ld.pp_cores_d.merge(ld.dom, ld.z3db)
        ld.mcores_d.show_results()
        
        #print summary
        xtime_total = ld.itime_total - sum(dt.xtime for dt in ld.dts)
        nconfigs = ld.last_dt.nconfigs
        ncovs = ld.last_dt.ncovs
        
        logger.info(alg_igen.DTrace.str_of_summary(
            ld.seed,
            len(ld.dts),
            ld.itime_total,
            xtime_total,
            nconfigs,
            ncovs,
            dir_))

        # do_minconfigs has 3 possible values
        # 1. None: don't find min configs
        # 2. f: (callable(f)) find min configs using f
        # 3. anything else: find min configs using existing configs
        if do_minconfigs is None:
            minconfigs = []
            min_ncovs = 0
        else:
            logger.info("*Min Configs")
            from alg_miscs import MinConfigs
            mc = MinConfigs(ld)
            if callable(do_minconfigs):
                minconfigs, min_ncovs = mc.search_f(f=do_minconfigs)
            else:
                minconfigs, min_ncovs = mc.search_existing()

        ### Influential Options/Settings ###
        influences = None
        if do_influence:
            logger.info("*Influence")
            from alg_miscs import Influence
            influences = Influence(ld).search(ncovs)

        ### Precision ###
        equivs, weaks, strongs, nones = None, None, None, None
        if do_precision:
            logger.info("*Precision")
            from alg_miscs import Precision
            ud = Precision(ld)
            equivs, weaks = ud.check_existing()
            if cmp_dir: #compare to ground truths
                logger.info("cmp to results in '{}'".format(cmp_dir))
                equivs, weaks, strongs, nones = ud.check_gt(cmp_dir)

        ### Evolutionar: F-scores ###
        rd_fscore, gt_fscore, gt_pp_cores_d = None, None, None

        if do_evolution:
            logger.info("* Evolution")
            from alg_miscs import Similarity
            sl = Similarity(ld)

            #compare to ground truths
            if cmp_dir: 
                logger.info("cmp to results in '{}'".format(cmp_dir))
                gt_fscores, gt_pp_cores_d, gt_ncovs, gt_nconfigs = sl.get_fscores(cmp_dir)
                
                last_elem_f = lambda l: l[-1][1] if l and len(l) > 0 else None
                gt_fscore = last_elem_f(gt_fscores)
                logger.info("configs ({}/{}) cov ({}/{}) fscore {}"
                            .format(nconfigs, gt_nconfigs, ncovs, gt_ncovs, gt_fscore))
                
            #compare to rand search
            callf = cmp_rand
            if callable(callf) and gt_pp_cores_d:
                r_pp_cores_d,r_cores_d,r_configs_d,r_covs_d,_ = callf(ld.seed, nconfigs)
                rd_fscore = sl.fscore_cores_d(r_pp_cores_d, gt_pp_cores_d)

                logger.info("rand: configs {} cov {} fscore {}"
                            .format(len(r_configs_d),len(r_covs_d), rd_fscore))

        #return analyzed results
        rs = AnalysisResults(niters=len(ld.dts),
                             ncores=len(ld.mcores_d),
                             itime=ld.itime_total,
                             xtime=xtime_total,
                             ncovs=ncovs,
                             nconfigs=nconfigs,
                             n_minconfigs=len(minconfigs),
                             min_ncovs=min_ncovs,
                             gt_fscore=gt_fscore,
                             rd_fscore=rd_fscore,
                             influences=influences,
                             m_strens=ld.mcores_d.strens,
                             m_strens_str=ld.mcores_d.strens_str,
                             m_vtyps=ld.mcores_d.vtyps)
        return rs

    @classmethod
    def replay_dirs(cls, dir_, show_iters,
                    do_minconfigs,
                    do_influence,
                    do_evolution,
                    do_precision,
                    cmp_rand,
                    cmp_dir):
        
        dir_ = CM.getpath(dir_)
        logger.info("replay_dirs '{}'".format(dir_))
        
        strens_arr = []
        strens_str_arr = []
        vtyps_arr = []
        niters_arr = []
        ncores_arr = []
        nitime_arr = []
        nxtime_arr = [] 
        nconfigs_arr = []
        ncovs_arr = []
        fscores = []
        rfscores = []
        nminconfigs_arr = []
        min_ncovs_arr = []        
        for rdir in sorted(os.listdir(dir_)):
            rdir = os.path.join(dir_,rdir)
            o = Analysis.replay(rdir, show_iters,
                                do_minconfigs,
                                do_influence,
                                do_evolution,
                                do_precision,
                                cmp_rand, cmp_dir)
            
            strens_arr.append(o.m_strens)
            strens_str_arr.append(o.m_strens_str)
            vtyps_arr.append(o.m_vtyps)
            fscores.append(o.gt_fscore)
            rfscores.append(o.rd_fscore)
            
            niters_arr.append(o.niters)
            ncores_arr.append(o.ncores)
            nitime_arr.append(o.itime)
            nxtime_arr.append(o.xtime)
            nconfigs_arr.append(o.nconfigs)
            ncovs_arr.append(o.ncovs)
            nminconfigs_arr.append(o.n_minconfigs)
            min_ncovs_arr.append(o.min_ncovs)            

        def median_siqr((s, arr)):
            return "{} {} ({})".format(s, median(arr), siqr(arr))


        nruns = len(strens_arr)
        nruns_f = float(nruns)
        logger.info("*** Analysis over {} runs ***".format(nruns))

        rs = [("iter", niters_arr),
              ("ints", ncores_arr),
              ("time", nitime_arr),
              ("xtime", nxtime_arr),
              ("configs", nconfigs_arr),
              ("covs", ncovs_arr),
              ("nminconfigs", nminconfigs_arr),
              ("nmincovs", min_ncovs_arr)]
        logger.info(', '.join(median_siqr(r) for r in rs))

        #vtyps_arr= [(c,d,m), ... ]
        conjs, disjs, mixs = zip(*vtyps_arr)
        rs = [("conjs", conjs),("disjs", disjs), ("mixed", mixs) ]
        logger.info("Int types: {}".format(', '.join(median_siqr(r) for r in rs)))
        
        sres = {}
        for i,(strens,strens_str) in enumerate(zip(strens_arr,strens_str_arr)):
            logger.debug("run {}: {}".format(i+1,strens_str))
            for strength,ninters,ncov in strens:
                if strength not in sres:
                    sres[strength] = ([],[])

                inters,covs = sres[strength]
                inters.append(ninters)
                covs.append(ncov)


        rs = []
        for strength in sorted(sres):
            inters,covs = sres[strength]
            assert len(inters) == len(covs)
            ndiffs = nruns - len(inters)
            if ndiffs:
                inters.extend([0,] * ndiffs)
                covs.extend([0,] * ndiffs)

            rs.append("({}, {} ({}), {} ({}))"
                      .format(strength,
                              median(inters), siqr(inters),
                              median(covs), siqr(covs)))

        logger.info("Int strens: {}".format(', '.join(rs)))
        

        #fscores
        fscores = [-1 if s is None else s for s in fscores]
        rfscores = [-1 if s is None else s for s in rfscores]        
        rs = [("gt", fscores), ("rand", rfscores)]
        logger.info("fscores: {}".format(', '.join(median_siqr(r) for r in rs)))

        
    @staticmethod
    def debug_find_configs(sid, configs_d, find_in):
        if find_in:
            cconfigs_d = dict((c,cov) for c,cov in configs_d.iteritems()
                              if sid in cov)
        else:
            cconfigs_d = dict((c,cov) for c,cov in configs_d.iteritems()
                              if sid not in cov)

        logger.info(cconfigs_d)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
