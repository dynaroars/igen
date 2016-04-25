import os.path
import numpy
import vu_common as CM
import config_common as CC

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

        import alg as IA
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
        from alg_miscs import HighCov
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

        logger.info("Additional analysis")
        from alg_miscs import Influence
        influence_scores = Influence.get_influence(mcores_d,ncovs,dom)

        from alg_miscs import Metrics
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
            ss.append("({}, {}, {})".format(strength,sum(inters)/nruns_total_f,sum(covs)/nruns_total_f))
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
    def debug_find_configs(sid,configs_d,find_in):
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
    
