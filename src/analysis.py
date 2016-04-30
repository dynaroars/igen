import os.path
import numpy
import vu_common as CM
import config_common as CC
import alg as IA

logger = CM.VLog('analysis')
logger.level = CC.logger_level
CM.VLog.PRINT_TIME = True

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
            self._z3db = self.dom.z3db
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
    
        
    #class methods
    @classmethod
    def load_dir(cls, dir_):
        assert dir_
        if dir_ in cls.data:
            return cls.data[dir_]
        else:
            seed, dom, dts, pp_cores_d, itime_total = IA.DTrace.load_dir(dir_)
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
          'n_min_configs',
          'min_ncovs',
          'vscores',
          'fscores',
          'r_fvscores',
          'influence_scores',
          'm_strens',
          'm_strens_str',
          'm_vtyps']
AnalysisResults = namedtuple("AnalysisResults",' '.join(fields))

class Analysis(object):
    
    @classmethod
    def is_run_dir(cls, d):
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
            if (ds and
                all(os.path.isdir(d_) and cls.is_run_dir(d_) for d_ in ds)):
                return False
            else:
                return None

    @classmethod
    def replay(cls, dir_, show_iters, do_min_configs, cmp_gt, cmp_rand):
        """
        Replay and analyze execution info from saved info in dir_

        """
        assert isinstance(dir_,str), dir_
        assert isinstance(show_iters, bool), show_iters
        assert cmp_gt is None or isinstance(cmp_gt, str), cmp_gt
        assert cmp_rand is None or callable(cmp_rand), cmp_rand

        logger.info("replay dir: '{}'".format(dir_))
        ld = LoadData.load_dir(dir_)
        z3db = ld.dom.z3db
        
        logger.info('seed: {}'.format(ld.seed))
        logger.debug(ld.dom.__str__())

        dts = sorted(ld.dts, key=lambda dt: dt.citer)        
        if show_iters:
            for dt in dts:
                dt.show(z3db, ld.dom)

        if not hasattr(ld.pp_cores_d.values()[0], 'vstr'):
            logger.warn("Old format, has no vstr .. re-analyze")
            ld.pp_cores_d = ld.pp_cores_d.analyze(ld.dom, covs_d=None)

        ld.mcores_d = ld.pp_cores_d.merge(z3db, ld.dom)
        ld.mcores_d.show_results()
        
        #print summary
        xtime_total = ld.itime_total - sum(dt.xtime for dt in dts)
        last_dt = max(ld.dts, key=lambda dt: dt.citer) #last iteration
        nconfigs = last_dt.nconfigs
        ncovs = last_dt.ncovs
        
        logger.info(IA.DTrace.str_of_summary(
            ld.seed,
            len(ld.dts),
            ld.itime_total,
            xtime_total,
            nconfigs,
            ncovs,
            dir_))

        logger.info("*** Additional analysis ***")

        # do_min_configs has 3 possible values
        # 1. None: don't find min configs
        # 2. f: (callable(f)) find min configs using f
        # 3. anything else: find min configs using existing configs
        if do_min_configs is None:
            min_configs = []
            min_ncovs = 0
        else:
            logger.info("Finding Min Configs")
            from alg_miscs import MinConfigs
            mc = MinConfigs(ld)
            if callable(do_min_configs):
                min_configs, min_ncovs = mc.search_f(f=do_min_configs)
            else:
                min_configs, min_ncovs = mc.search_existing()
            
        logger.info("Influential Options")
        from alg_miscs import Influence
        influence_scores = Influence(ld).search(ncovs)

        logger.info("Check Precision")
        from alg_miscs import Precision
        ud = Precision(ld)
        strong_mcores_d = {}
        ok_mcores_d, weak_mcores_d = ud.check_existing()
        if cmp_gt: #compare against ground truths
            logger.info("against results in '{}'".format(cmp_gt))
            equivs, weaks, strongs, nones = ud.check_gt(cmp_dir=cmp_gt)
            
        logger.info("Measure Similarity")
        from alg_miscs import Similarity
        sl = Similarity(ld)
        vscores = sl.get_vscores()
        fscores, gt_pp_cores_d = None, None
        if cmp_gt: #compare against ground truths
            logger.info("against results in '{}'".format(cmp_gt))
            fscores, gt_pp_cores_d = sl.get_fscores(cmp_dir=cmp_gt)
            
        #compare against rand search
        r_f = cmp_rand
        if callable(r_f):
            r_pp_cores_d,r_cores_d,r_configs_d,r_covs_d,_ = r_f(nconfigs)
            if gt_pp_cores_d:
                r_fscore = sl.fscore_cores_d(r_pp_cores_d, gt_pp_cores_d)
            else:
                r_fscore = None
                
            r_vscore = sl.vscore_cores_d(r_cores_d)
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


        #return analyzed results
        rs = AnalysisResults(niters=len(dts),
                             ncores=len(ld.mcores_d),
                             itime=ld.itime_total,
                             xtime=xtime_total,
                             ncovs=ncovs,
                             nconfigs=nconfigs,
                             n_min_configs=len(min_configs),
                             min_ncovs=min_ncovs,
                             vscores=vscores,
                             fscores=fscores,
                             r_fvscores=r_fvscores,
                             influence_scores=influence_scores,
                             m_strens=ld.mcores_d.strens,
                             m_strens_str=ld.mcores_d.strens_str,
                             m_vtyps=ld.mcores_d.vtyps)
        return rs

    @classmethod
    def replay_dirs(cls, dir_,show_iters,do_min_configs,cmp_gt,cmp_rand):
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
                rdir, show_iters, do_min_configs, cmp_gt, cmp_rand)
            
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

        nruns_total = len(strens_s)
        nruns_total_f = float(nruns_total)

        ss = ["iter {}".format(niters_total/nruns_total_f),
              "results {}".format(ncores_total/nruns_total_f),
              "time {}".format(nitime_total/nruns_total_f),
              "xtime {}".format(nxtime_total/nruns_total_f),
              "configs {}".format(nconfigs_total/nruns_total_f),
              "covs {}".format(ncovs_total/nruns_total_f),
              "minconfigs {}".format(nminconfigs_total/nruns_total_f),
              "nmincovs {}".format(min_ncovs_total/nruns_total_f)]
        
        logger.info("STAT of {} runs (avg): {}"
                    .format(nruns_total, ', '.join(ss)))
        
        ssMed = ["iter {}".format(numpy.median(niters_arr)),
                 "results {}".format(numpy.median(ncores_arr)),
                 "time {}".format(numpy.median(nitime_arr)),
                 "xtime {}".format(numpy.median(nxtime_arr)),
                 "configs {}".format(numpy.median(nconfigs_arr)),
                 "nminconfigs {}".format(numpy.median(nminconfigs_arr)),
                 "covs {}".format(numpy.median(ncovs_arr)),
                 "nmincovs {}".format(numpy.median(min_ncovs_arr))]
        logger.info("STAT of {} runs (median): {}"
                    .format(nruns_total, ', '.join(ssMed)))
        
        ssSIQR = ["iter {}".format(Analysis.siqr(niters_arr)),
                  "results {}".format(Analysis.siqr(ncores_arr)),
                  "time {}".format(Analysis.siqr(nitime_arr)),
                  "xtime {}".format(Analysis.siqr(nxtime_arr)),
                  "configs {}".format(Analysis.siqr(nconfigs_arr)),
                  "nminconfigs {}".format(Analysis.siqr(nminconfigs_arr)),                  
                  "covs {}".format(Analysis.siqr(ncovs_arr)),
                  "nmincovs {}".format(Analysis.siqr(min_ncovs_arr))]
        
        logger.info("STATS of {} runs (SIQR): {}"
                    .format(nruns_total_f,', '.join(ssSIQR)))

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
            for num in range(length,int(nruns_total_f)):
                inters.append(0)
                covs.append(0)
            ss.append("({}, {}, {})"
                      .format(strength,
                              sum(inters)/nruns_total_f,
                              sum(covs)/nruns_total_f))
            medians.append("({}, {}, {})"
                           .format(strength, numpy.median(inters), numpy.median(covs)))
            siqrs.append("({}, {}, {})"
                         .format(strength, Analysis.siqr(inters), Analysis.siqr(covs)))
            tmp.append("{},{})"
                       .format(strength,','.join(map(str, inters))))
            tex_table4.append("{} \\mso{{{}}}{{{}}}"
                              .format(strength,numpy.median(inters),Analysis.siqr(inters)))
            tex_table5.append("{} \\mso{{{}}}{{{}}}"
                              .format(strength,numpy.median(covs),Analysis.siqr(covs)))
        
        logger.info("interaction strens averages: {}".format(', '.join(ss)))
        logger.info("interaction strens medians : {}".format(', '.join(medians)))
        logger.info("interaction strens SIQRs   : {}".format(', '.join(siqrs)))
        #logger.info("interactions arrays   : {}".format('\n'.join(tmp)))
        
        conjs = [c for c,_,_ in vtyps_s]
        disjs = [d for _,d,_ in vtyps_s]
        mixs = [m for _,_,m in vtyps_s]
        
        length=len(conjs)
        for num in range(length,int(nruns_total_f)):
            conjs.append(0)
        
        length=len(disjs)
        for num in range(length,int(nruns_total_f)):
            disjs.append(0)
        
        length=len(mixs)
        for num in range(length,int(nruns_total_f)):
            mixs.append(0)
        
        #logger.info("conjs array: {}".format(', '.join(map(str, conjs))))
        #logger.info("disjs array: {}".format(', '.join(map(str, disjs))))
        #logger.info("mixs  array: {}".format(', '.join(map(str, mixs))))

        nconjs = sum(conjs)/nruns_total_f
        ndisjs = sum(disjs)/nruns_total_f
        nmixs  = sum(mixs)/nruns_total_f
        
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
    
