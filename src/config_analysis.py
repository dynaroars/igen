import os.path
import numpy
import vu_common as CM

logger = CM.VLog('analysis')
logger.level = CM.VLog.DEBUG
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
    def replay(dir_,show_iters,do_min_configs):
        """
        Replay execution info from saved info in dir_
        """
        seed,dom,dts,pp_cores_d,itime_total = Analysis.load_dir(dir_)

        #print info
        logger.info("replay dir: '{}'".format(dir_))
        logger.info('seed: {}'.format(seed))
        logger.debug(dom.__str__())
        
        if show_iters:
            for dt in sorted(dts,key=lambda dt: dt.citer):
                dt.show()

        #print postprocess results
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

            mcores_d.get_min_configs(covs,configs_d,dom)
        
        return (len(dts),len(mcores_d),
                itime_total,xtime_total,nconfigs,ncovs,
                mcores_d.strens,mcores_d.strens_str,mcores_d.vtyps)

    @staticmethod
    def replay_dirs(dir_,show_iters,do_min_configs):
        dir_ = CM.getpath(dir_)
        logger.info("replay_dirs '{}'".format(dir_))
        
        niters_total = 0
        nresults_total = 0        
        nitime_total = 0
        nxtime_total = 0    
        nconfigs_total = 0
        ncovs_total = 0
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
        counter = 0
        csv_arr = []

        for rdir in sorted(os.listdir(dir_)):
            rdir = os.path.join(dir_,rdir)
            (niters,nresults,itime,xtime,nconfigs,ncovs,
             strens,strens_str,ntyps) = Analysis.replay(rdir,
                                                        show_iters,
                                                        do_min_configs)
            niters_total += niters
            nresults_total += nresults
            nitime_total += itime
            nxtime_total += xtime
            nconfigs_total += nconfigs
            ncovs_total += ncovs
            strens_s.append(strens)
            strens_str_s.append(strens_str)
            ntyps_s.append(ntyps)

            niters_arr.append(niters)
            nresults_arr.append(nresults)
            nitime_arr.append(itime)
            nxtime_arr.append(xtime)
            nconfigs_arr.append(nconfigs)
            ncovs_arr.append(ncovs)
            csv_arr.append("{},{},{},{},{},{},{},{},{}".format(
                counter,niters,nresults,itime,xtime,nconfigs,ncovs,
                ','.join(map(str, ntyps)),','.join(map(str, strens))))
            counter += 1

        nruns_total = float(len(strens_s))

        ss = ["iter {}".format(niters_total/nruns_total),
              "results {}".format(nresults_total/nruns_total),
              "time {}".format(nitime_total/nruns_total),
              "xtime {}".format(nxtime_total/nruns_total),
              "configs {}".format(nconfigs_total/nruns_total),
              "covs {}".format(ncovs_total/nruns_total)]
        logger.info("STATS of {} runs (averages): {}".format(nruns_total,', '.join(ss)))
        
        ssMed = ["iter {}".format(numpy.median(niters_arr)),
              "results {}".format(numpy.median(nresults_arr)),
              "time {}".format(numpy.median(nitime_arr)),
              "xtime {}".format(numpy.median(nxtime_arr)),
              "configs {}".format(numpy.median(nconfigs_arr)),
              "covs {}".format(numpy.median(ncovs_arr))]
        logger.info("STATS of {} runs (medians) : {}".format(nruns_total,', '.join(ssMed)))
        
        ssSIQR = ["iter {}".format(Analysis.siqr(niters_arr)),
              "results {}".format(Analysis.siqr(nresults_arr)),
              "time {}".format(Analysis.siqr(nitime_arr)),
              "xtime {}".format(Analysis.siqr(nxtime_arr)),
              "configs {}".format(Analysis.siqr(nconfigs_arr)),
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
