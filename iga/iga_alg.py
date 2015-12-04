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


class GA(object):
    
    @staticmethod
    def mutate_p(config, dom, p):
        """
        Randomly change values of config using probability p
        """
        assert isinstance(config, Config), config
        assert isinstance(dom, Dom), dom
        assert len(config) == len(dom), (len(config), len(dom))
        assert isinstance(p, float) and 0.0 <= p <= 1.0, p

        settings = []
        for k, v in config.iteritems():
            if random.random() >= p:
                v = random.choice(list(dom[k] - set([v])))
            settings.append((k,v))
        return Config(settings)

    
    @staticmethod
    def mutate_n(config, dom, n):
        """
        Create a new candidate by suffling n values in config
        
        >>> ks = 'a b c d e f'.split()
        >>> vs = ['0 1 3 4', '0 1', '0 1', '0 1', '0 1', '0 1 2']
        >>> dom = Dom(zip(ks, [frozenset(v.split()) for v in vs]))
        >>> c = Config(zip(ks, '0 1 0 1 0 1'.split()))
        >>> print c
        a=0 b=1 c=0 d=1 e=0 f=1

        >>> random.seed(0)
        >>> configs = [GA.mutate_n(c, dom, 1) for _ in range(3)]
        >>> for c in configs: print c
        a=0 b=1 c=0 d=1 e=0 f=2
        a=0 b=1 c=1 d=1 e=0 f=1
        a=0 b=1 c=0 d=0 e=0 f=1

        >>> random.seed(0)
        >>> print GA.mutate_n(c, dom, 4)
        a=3 b=0 c=0 d=1 e=0 f=0
        """
        assert isinstance(config, Config), config
        assert isinstance(dom, Dom), dom
        assert len(config) == len(dom), (len(config), len(dom))
        assert 0 < n < len(dom), n

        ks = set(random.sample(dom.keys(), n))
        settings = []
        for k, v in config.iteritems():
            if k in ks:
                v = random.choice(list(dom[k] - set([v])))
            settings.append((k,v))
        return Config(settings)

    @staticmethod
    def mutate(config, dom, p_or_n):
        """
        Call mutate_p if p_or_n is a float and call mutate_n if p_or_n is an int
        """
        f = GA.mutate_p if isinstance(p_or_n, float) else GA.mutate_n
        return f(config, dom, p_or_n)
        

    @staticmethod
    def get_fitness(pop):
        pass
    
    @staticmethod
    def gen_pop_tourn_sel(sid, old_pop, pop_siz, cur_exploit, configs_d, fits_d, dom):
        #create new configs
        freqs = IGa.get_freqs(sid, old_pop, fits_d, dom)
        pop = [IGa.tourn_select(freqs, cur_exploit) for _ in range(pop_siz)]

        #mutation
        pop_ = []
        for c in pop:
            if random.random() > cur_exploit:
                c = GA.mutate(c, dom, 1)
            pop_.append(c)

        pop = pop_

        #random
        uniqs = set(c for c in pop if c not in configs_d)
        if not uniqs:
            #introduce some varieties
            pass

        return pop

    #Steady state
    @staticmethod
    def gen_pop_steady_state(sid, old_pop, pop_siz, cur_exploit, configs_d, fits_d, dom):
        """
        Pick a config with the highest fitness and modify it
        """
        assert isinstance(sid, str), sid
        assert old_pop, old_pop
        assert pop_siz > 0, pop_siz
        assert isinstance(dom, Dom), dom
        
        best_fit = max(fits_d[c][sid] for c in fits_d)
        pop_bests = [c for c in old_pop if fits_d[c][sid] == best_fit]
        best = random.choice(pop_bests)
        pop = dom.gen_configs_cex(best, configs_d)
        rand_n = pop_siz - len(pop)
        if rand_n > 0:
            pop_ = dom.gen_configs_rand_smt(
                rand_n, existing_configs=configs_d.keys(), config_cls=Config)
            pop.extend(pop_)
        return pop
    
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
        
    def eval_configs(self, configs):
        if __debug__:
            assert (isinstance(configs, list) and
                    all(isinstance(c, Config) for c in configs)
                    and configs), configs
        st = time()
        results = CC.eval_configs(configs, self.get_cov)
        cconfigs_d = Configs_d()
        for c, rs in results:
            cconfigs_d[c] = rs
        return cconfigs_d, time() - st

    @staticmethod
    def rm_sids(cconfigs_d, s):
        """
        Only keep statements containing string s
        """
        for c in cconfigs_d:
            cconfigs_d[c] = set(sid for sid in cconfigs_d[c] if s in sid)

    def eval_update_configs(self, pop, s, eval_configs_f, configs_d, covs):
        cconfigs_d, xtime = eval_configs_f(pop)
        IGa.rm_sids(cconfigs_d, s)
        IGa.update_caches(cconfigs_d, configs_d, covs)
        return xtime

    def go(self, seed, sids, econfigs=None, tmpdir=None):
        if __debug__:
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

    @staticmethod
    def update_caches(cconfigs_d, configs_d, covs):
        """
        Update configs_d, configs, covs with the cconfigs_d
        """
        if __debug__:
            assert isinstance(cconfigs_d, Configs_d), cconfigs_d
            assert CC.is_cov(covs), covs

        for c in cconfigs_d:
            if c not in configs_d:
                configs_d[c] = cconfigs_d[c]

            for s in configs_d[c]:
                covs.add(s)

    def go_sid(self, sid, configs_d, fits_d):
        """
        Use GA to find sid.
        Also update information from configs_d and fits_d
        """
        if __debug__:
            assert isinstance(sid, str), sid
            assert isinstance(configs_d, Configs_d), configs_d
            assert isinstance(fits_d, Fits_d), fits_d

        max_stuck = 10
        cur_stuck = 0
        max_exploit = 0.8
        cur_exploit = max_exploit
        xtime_total = 0.0
        covs = set()
        bests = set()
        
        cur_iter = 0        
        req_s = os.path.basename(sid).split(':')[0]
        pop_siz = max(len(self.dom), self.dom.max_fsiz) * 2
        
        while sid not in covs and cur_stuck <= max_stuck:
            cur_iter += 1
            logger.debug("sid {}, iter {} covs {} stuck {}/{}, exploit {}"
                         .format(sid, cur_iter, len(covs),
                                 cur_stuck, max_stuck, cur_exploit))

            #init configs (to find easy locations quickly)
            if cur_iter == 1: 
                pop = self.dom.gen_configs_tcover1(config_cls=Config)
                logger.debug("create {} init pop".format(len(pop)))

                xtime = self.eval_update_configs(
                    pop, req_s, self.eval_configs, configs_d, covs)
                xtime_total += xtime
                continue
            
            if cur_iter == 2: #compute fitness of init configs
                paths = self.cfg.get_paths(sid)
                logger.debug("{} paths: {}".format(
                    len(paths), CM.str_of_list(map(len,paths))
                ))
                logger.debug("\n" + '\n'.join("{}. {}".format(i, CM.str_of_list(p))
                                              for i,p in enumerate (paths)))
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
            # pop = GA.gen_pop_tourn_sel(sid, pop, pop_siz, cur_exploit,
            #                            configs_d, fits_d, self.dom)

            pop = GA.gen_pop_steady_state(sid, pop, pop_siz, cur_exploit,
                                          configs_d, fits_d, self.dom)
            
            # evaluate & compute fitness
            xtime = self.eval_update_configs(
                pop, req_s, self.eval_configs, configs_d, covs)
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
                cur_exploit = max_exploit
            else:
                cur_stuck += 1
                cur_exploit -= max_exploit / max_stuck
                if cur_exploit < 0.01: cur_exploit = 0.01

        if sid in covs:
            config = CM.find_first(configs_d, lambda c: sid in configs_d[c])
            logger.debug("sol for '{}' : {}".format(sid, config))
            
        return covs
                
    @staticmethod
    def tourn_sel(configs, sid, fits_d, tourn_siz):
        if __debug__:
            assert configs, configs
            assert isinstance(fits_d, Fits_d), fits_d
            assert tourn_siz >= 2, tourn_siz
            
        sample = (random.choice(configs) for _ in range(tourn_siz))
        best = max(sample, key=lambda c: fits_d[c][sid])
        return best
            
    @staticmethod
    def get_best(sid, configs, fits_d):
        if __debug__:
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
        if __debug__:
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
    def get_freqs(sid, configs, fits_d, dom):
        """
        Returns a list of variable names and their fitness scores and occurrences.
        Used later to create new configuration (i.e., those with high scoes and 
        appear frequently are likely chosen).

        >>> ks = 'a b c d e f'.split()
        >>> vs = ['0 1 3 4', '0 1', '0 1', '0 1', '0 1', '0 1 2']
        >>> dom = Dom(zip(ks, [frozenset(v.split()) for v in vs]))
        >>> cs = [\
        ('0 1 0 1 0 0',2),\
        ('1 1 0 1 0 0',1),\
        ('1 0 1 0 1 0',1),\
        ('1 0 1 0 0 1',0),\
        ('0 1 0 1 0 1',2),\
        ('0 0 1 0 1 1',8)]

        >>> configs = [Config(zip(ks, c.split())) for c,f in cs]
        >>> fits = [{'s':f} for _, f in cs]
        >>> fits_d = Fits_d()
        >>> for c,f in zip(configs, fits): fits_d[c]=f

        >>> print fits_d.__str__('s')
        1. a=0 b=0 c=1 d=0 e=1 f=1 fit 8
        2. a=0 b=1 c=0 d=1 e=0 f=0 fit 2
        3. a=0 b=1 c=0 d=1 e=0 f=1 fit 2
        4. a=1 b=1 c=0 d=1 e=0 f=0 fit 1
        5. a=1 b=0 c=1 d=0 e=1 f=0 fit 1
        6. a=1 b=0 c=1 d=0 e=0 f=1 fit 0

        >>> freqs = IGa.get_freqs('s', configs, fits_d, dom)
        >>> print freqs
        [('a', [('0', 2), ('1', 1), ('1', 1), ('1', 0), ('0', 2), ('0', 8), ('3', -0.1), ('4', -0.1)]), ('b', [('1', 2), ('1', 1), ('0', 1), ('0', 0), ('1', 2), ('0', 8)]), ('c', [('0', 2), ('0', 1), ('1', 1), ('1', 0), ('0', 2), ('1', 8)]), ('d', [('1', 2), ('1', 1), ('0', 1), ('0', 0), ('1', 2), ('0', 8)]), ('e', [('0', 2), ('0', 1), ('1', 1), ('0', 0), ('0', 2), ('1', 8)]), ('f', [('0', 2), ('0', 1), ('0', 1), ('1', 0), ('1', 2), ('1', 8), ('2', -0.1)])]

        #high exploit rates result in values with high fits (e.g., a=0)
        >>> random.seed(0)
        >>> configs = Config.mk(10, lambda: IGa.tourn_select(freqs, 0.9))
        >>> for c in configs: print c
        a=0 b=0 c=1 d=0 e=1 f=1
        a=0 b=0 c=1 d=0 e=0 f=1
        a=0 b=0 c=1 d=1 e=1 f=1
        a=0 b=0 c=1 d=0 e=0 f=0
        a=0 b=1 c=1 d=0 e=1 f=1
        a=0 b=0 c=0 d=0 e=0 f=1
        a=0 b=0 c=1 d=1 e=1 f=0
        a=0 b=1 c=1 d=1 e=1 f=0
        a=0 b=1 c=0 d=1 e=1 f=1
        a=0 b=1 c=1 d=0 e=1 f=0

        #with low exploit rate, also consider value with low fits
        #or even values that do not appear in the configs (e.g., a=3)
        >>> random.seed(0)
        >>> configs = Config.mk(10, lambda: IGa.tourn_select(freqs, 0.1))
        >>> for c in configs: print c
        a=3 b=1 c=1 d=1 e=0 f=0
        a=3 b=1 c=1 d=0 e=1 f=1
        a=1 b=1 c=1 d=1 e=1 f=2
        a=3 b=0 c=0 d=1 e=1 f=1
        a=1 b=1 c=1 d=0 e=1 f=2
        a=1 b=0 c=0 d=1 e=0 f=0
        a=0 b=0 c=0 d=1 e=0 f=1
        a=3 b=1 c=0 d=0 e=0 f=1
        a=1 b=0 c=0 d=0 e=0 f=0
        a=0 b=0 c=0 d=0 e=0 f=1

        """
        if __debug__:
            assert isinstance(sid, str), sid
            assert (not configs or
                    all(isinstance(c, Config) for c in configs)), configs
            assert fits_d and isinstance(fits_d, Fits_d), fits_d
            assert isinstance(dom, Dom), dom
            

        fits = [fits_d[config][sid] for config in configs]
        #compute low_fit wrt to fits (instead of giving a fixed value) avoids
        #problem with fitness having negative values (not with fscore, but could
        #happen with other approach)
        low_fit = min(fits) - 0.1

        freqs = []
        # Iterate through variables.
        for k, vs in dom.iteritems():
            # For each variable, obtain its value and fitness score from configs.
            cvs = [config[k] for config in configs]
            rs = zip(cvs, fits)
            # Also give a low fitness score to values not appearring in configs
            # so that these values are also considered
            cvs = set(cvs)
            rs_ = [(v, low_fit) for v in vs if v not in cvs]
            freqs.append((k, rs + rs_))

        return freqs

    @staticmethod
    def tourn_select(freqs, exploit_rate):
        """
        Use tournament selection to create a new candidate from freqs list 
        (computed from fitness and occurences).
        """
        if __debug__:
            assert freqs and isinstance(freqs, list), freqs
            #('var', [('val', score)])
            assert all(isinstance(f,tuple) and len(f)==2 and
                       isinstance(f[0], str) and
                       isinstance(f[1], list) and f[1] and 
                       all(isinstance(vs, tuple) and len(vs)==2 and
                           isinstance(vs[0], str) and isinstance(vs[1], (int,float))
                           for vs in f[1])
                       for f in freqs), freqs
            assert (isinstance(exploit_rate, float) and
                    0.0 <= exploit_rate <= 1.0), exploit_rate

        settings = []
        for k, ls in freqs:
            if __debug__:
                assert ls, ls

            #higher exploit rate gives larger tour_siz
            if exploit_rate > 0:
                tourn_siz = int(math.ceil(len(ls) * exploit_rate))
            else:
                tourn_siz = 1
            if __debug__:
                assert 1 <= tourn_siz <= len(ls), tourn_siz
            #select randomly tourn_siz values and use the largest fitness
            sample = (random.choice(ls) for _ in range(tourn_siz))
            val, _ = max(sample, key=lambda (val, fit) : fit)
            settings.append((k, val))
            
        return Config(settings)
    
    @staticmethod
    def get_fscore(cov, path):
        # print cov
        # print path
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


                
if __name__ == "__main__":
    import doctest
    doctest.testmod()
