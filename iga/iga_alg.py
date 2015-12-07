from collections import OrderedDict
from time import time
import os.path
import random
import math

import vu_common as CM
import config_common as CC
import cfg as CFG

logger = CM.VLog('GA')
logger.level = CC.logger_level
CM.VLog.PRINT_TIME = True

class Dom(CC.Dom):
    def gen_configs_cex(self, config, existing_configs):

        new_configs = []
        for k in config:
            vs = self[k] - set(config[k])
            for v in vs:
                new_config = Config(config)
                new_config[k] = v
                if new_config not in existing_configs:
                    new_configs.append(new_config)
        return new_configs
                
class Config(CC.Config):
    pass

class Configs_d(CC.Configs_d):
    pass

class Fits_d(CC.CustDict):
    def __setitem__(self, config, fits):
        assert isinstance(config,Config),config
        assert config not in self.__dict__,config
        assert all((sid is None or isinstance(sid,str))
                   and isinstance(fit,(int,float))
                   for sid,fit in fits.iteritems()), fits
        
        self.__dict__[config] = fits

    def __str__(self, sid):
        cs = (c for c,fits in self.__dict__.iteritems() if sid in fits)
        cs = sorted(cs,key=lambda c: self.__dict__[c][sid],reverse=True)
        return '\n'.join("{}. {} fit {}"
                         .format(i+1,c,self.__dict__[c][sid])
                         for i,c in enumerate(cs))

class IGa(object):
    """
    Main algorithm
    """
    def __init__(self, dom, cfg, get_cov):
        assert isinstance(dom, CC.Dom), dom
        assert isinstance(cfg, CFG.CFG), cfg
        assert callable(get_cov), get_cov
        
        self.dom = Dom(dom)
        self.z3db = self.dom.z3db        
        self.cfg = cfg
        self.sids = self.cfg.__sids__
        logger.debug("cfg {}".format(len(self.cfg)))
        self.get_cov = get_cov

    def go(self, seed, sids, econfigs=None, tmpdir=None):
        assert isinstance(seed, float), seed
        assert (sids and isinstance(sids, set) and
                all(sid in self.sids for sid in sids)), sids
        assert not econfigs or isinstance(econfigs, list), econfigs
        assert isinstance(tmpdir, str) and os.path.isdir(tmpdir), tmpdir
        assert isinstance(tmpdir, str), tmpdir
        
        random.seed(seed)

        covs, found = set(), set()        
        configs_d = Configs_d()
        fits_d = Fits_d()
        
        logger.debug("search for {} sids".format(len(sids)))
        logger.debug(str(sids))
        
        remains = list(sorted(sids))
        while remains:            
            covs_ = self.go_sid(remains.pop(), configs_d, fits_d)
            for sid in covs_:
                covs.add(sid)

            remains = [sid for sid in remains if sid not in covs]
            found = set(sid for sid in sids if sid in covs)
            logger.debug("found {}/{}, remains {}".format(
                len(found), len(sids), len(remains)))
            
        is_success = found == sids
        notfound = set(sid for sid in sids if sid not in covs)
        if not is_success:
            logger.debug("not found {}/{}".format(len(notfound), len(sids)))
            logger.detail("{}".format(", ".join(sorted(notfound))))

        logger.info("{}, seed {}, found {}/{}, covs {}, configs {}/{}"
                    .format('success' if is_success else 'fail',
                            seed,
                            len(found), len(sids), len(covs), len(configs_d),
                            self.dom.siz))
        return is_success, found, configs_d
        

    def go_sid(self, sid, configs_d, fits_d):
        """
        Use GA to find sid.
        Also update information from configs_d and fits_d
        """
        assert isinstance(sid, str), sid
        assert isinstance(configs_d, Configs_d), configs_d
        assert isinstance(fits_d, Fits_d), fits_d

        found = False
        max_stuck = 3
        cur_stuck = 0
        xtime_total = 0.0
        covs = set()
        bests = set()
        
        cur_iter = 0        
        req_s = os.path.basename(sid).split(':')[0]
        pop_siz = max(len(self.dom), self.dom.max_fsiz)
        
        while True:
            cur_iter += 1
            logger.debug("sid '{}': iter {} covs {}"
                         .format(sid, cur_iter, len(covs)))

            #init configs (to quickly find easy locations)
            if cur_iter == 1: 
                pop = self.dom.gen_configs_tcover1(config_cls=Config)
                logger.debug("create {} init pop".format(len(pop)))

                found, xtime = IGa.eval_configs(
                    sid, pop, req_s, self.get_cov, configs_d, covs)
                    
                xtime_total += xtime
                if found:
                    break
                
                continue
            
            if cur_iter == 2: #compute fitness of init configs
                paths = self.cfg.get_paths(sid)
                logger.debug("{} paths: {}".format(
                    len(paths), CM.str_of_list(map(len,paths))
                ))
                logger.debug("\n" + '\n'.join("{}. {}".format(i, CM.str_of_list(p))
                                               for i,p in enumerate (paths)))
                if max(map(len,paths)) <= 1:
                    logger.warn("something wrong with cfg for sid '{}' .. skip"
                                .format(sid))
                    break
                
                IGa.compute_fits(sid, paths, pop, configs_d, fits_d)

            #cur stats
            cur_avg_fit, cur_best_fit, cur_best = IGa.get_avg_best(
                sid, pop, fits_d)
            pop_bests = [c for c in pop if fits_d[c][sid] == cur_best_fit]
            for c in pop_bests: bests.add(c)
            cur_nbests = len(bests)
            cur_ncovs = len(covs)
            
            logger.debug("pop {} (total {}) fit best {} avg {}"
                         .format(len(pop), len(configs_d),
                                 cur_best_fit, cur_avg_fit))
            logger.debug(str(cur_best))

            #create new configs
            pop = IGa.gen_pop(sid, pop, pop_siz, configs_d, fits_d, self.dom)
            
            # evaluate & compute fitness
            found, xtime = IGa.eval_configs(sid, pop, req_s, self.get_cov, configs_d, covs)
            if found:
                break
            xtime_total += xtime

            IGa.compute_fits(sid, paths, pop, configs_d, fits_d)
            avg_fit, best_fit, best = IGa.get_avg_best(sid, pop, fits_d)
            pop_bests = [c for c in pop if fits_d[c][sid] == best_fit]
            for c in pop_bests: bests.add(c)

            #elitism
            if best_fit < cur_best_fit:
                pop.append(cur_best)
            
            #progress made ?
            better_cov = len(covs) > cur_ncovs
            #better_avg = avg_fit > cur_avg_fit
            better_avg = False
            better_best = best_fit > cur_best_fit
            better_bests = len(bests) > cur_nbests
            if better_cov or better_avg or better_best or better_bests:
                cur_stuck = 0
            else:
                cur_stuck += 1
                if cur_stuck > max_stuck:
                    break

        if found:
            config = CM.find_first(configs_d, lambda c: sid in configs_d[c])
            logger.debug("sol for '{}' : {}".format(sid, config))
            
        return covs
    
    @staticmethod
    def eval_configs(sid, pop, req_s, get_cov_f, configs_d, covs_s):
        """
        Compute cov for pop and Update configs_d, configs, covs
        """
        assert isinstance(sid, str), sid
        assert (isinstance(pop, list) and pop and 
                all(isinstance(c, Config) for c in pop)), pop
        assert all(c not in configs_d for c in pop), pop
        assert isinstance(req_s, str), req_s
        assert callable(get_cov_f), get_cov_f
        assert isinstance(configs_d, Configs_d), configs_d
        assert CC.is_cov(covs_s), covs_s

        cache = set()
        st = time()
        
        for c in pop:
            if c in cache or c in configs_d:
                continue

            cache.add(c)
            sids, _ = get_cov_f(c)
            sids = set(s for s in sids if req_s in s)
            if not sids:
                logger.warn("config {} produces nothing".format(c))
            
            configs_d[c] = sids
            for s in sids:
                covs_s.add(s)
            
            if sid in sids:
                return True, time() - st
            
        return False, time() - st
    
    @staticmethod
    def get_best(sid, configs, fits_d):
        assert isinstance(sid, str), sid
        assert (configs is None or
                (configs and
                all(isinstance(c, Config) for c in configs))), configs
        assert fits_d and isinstance(fits_d, Fits_d), fits_d

        if configs is None:
            cs = [c for c in fits_d if sid in fits_d[c]]
        else:
            cs = configs
        best =  max(cs, key = lambda c: fits_d[c][sid])
        best_fit = fits_d[best][sid]
        return best_fit, best
    
    @staticmethod
    def get_avg(sid, configs, fits_d):
        assert isinstance(sid,str), sid
        assert (configs and
                all(isinstance(c, Config) for c in configs)), configs
        assert fits_d and isinstance(fits_d, Fits_d), fits_d

        if configs is None:
            cs = [c for c in fits_d if sid in fits_d[c]]
        else:
            cs = configs

        return sum(fits_d[c][sid] for c in cs) / float(len(cs))

    @staticmethod
    def get_avg_best(sids, configs, fits_d):
        avg_fit = IGa.get_avg(sids, configs, fits_d)
        best_fit, best = IGa.get_best(sids, configs, fits_d)
        return avg_fit, best_fit, best
    

    @staticmethod
    def get_fscore(cov, path):
        #TOFIX, these parms should be set
        assert isinstance(cov, set), cov
        assert isinstance(path, set), path
        
        if cov == path:
            return 1.0
        else:
            tp = len([s for s in cov if s in path])
            fp = len([s for s in cov if s not in path])
            fn = len([s for s in path if s not in cov])
            if fp == fn == 0:  #identical
                return 1.0
            else:
                p=0.0 if (tp+fp)==0 else (float(tp)/(tp+fp))
                r=0.0 if (tp+fn)==0 else (float(tp)/(tp+fn))
                f=0.0 if (r+p)==0 else float(2*r*p)/(r+p)
                return f

    @staticmethod
    def get_vscore(cov, path):
        if cov == path:
            return 1.0
        else:
            tp = len([s for s in cov if s in path])
            r = float(tp) / len(path)
            return r


    @staticmethod
    def compute_fits(sid, paths, configs, configs_d, fits_d):
        """
        Compute fitness (store results in fits_d)
        it's OK for duplicatins in configs (similar individuals in pop)
        """
        if __debug__:
            assert isinstance(sid,str), sid
            assert paths, paths
            assert (configs and
                    all(isinstance(c,Config) for c in configs)), configs
            assert configs_d and isinstance(configs_d,  Configs_d), configs
            assert isinstance(fits_d, Fits_d), fits_d

        for c in configs:
            if c in fits_d and sid in fits_d[c]:
                continue
            cov = configs_d[c]
            fit = max(IGa.get_fscore(cov, path) for path in paths)
            if c not in fits_d:
                fits_d[c] = {sid:fit}
            else:
                if __debug__:
                    assert sid not in fits_d[c], (sid, fits_d[c])
                fits_d[c][sid] = fit


    @staticmethod
    def gen_pop(sid, old_pop, pop_siz, configs_d, fits_d, dom):
        """
        Pick a config with the highest fitness and modify it
        """
        assert isinstance(sid, str), sid
        assert old_pop, old_pop
        assert pop_siz > 0, pop_siz
        assert isinstance(dom, Dom), dom
        
        best_fit = max(fits_d[c][sid] for c in fits_d if sid in fits_d[c])
        pop_bests = [c for c in old_pop if fits_d[c][sid] == best_fit]
        best = random.choice(pop_bests)
        pop = dom.gen_configs_cex(best, configs_d)
        rand_n = pop_siz - len(pop)
        if rand_n > 0:
            pop_ = dom.gen_configs_rand_smt(
                rand_n, existing_configs=configs_d.keys(), config_cls=Config)
            pop.extend(pop_)
        return pop
                
if __name__ == "__main__":
    import doctest
    doctest.testmod()
