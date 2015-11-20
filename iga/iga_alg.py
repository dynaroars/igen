from collections import OrderedDict
from time import time
import os.path
import random
import math

import vu_common as CM
import config_common as CC

logger = CM.VLog('GA')
logger.level = CC.logger_level
CM.VLog.PRINT_TIME = True

class CFG(OrderedDict):
    """
    sage: lines = [\
    'LS',\
    'L0 LS',\
    'L1 L0',\
    'L2 L0',\
    'L3 L2',\
    'L4 L3 L2 L1',\
    'L5 L4']

    sage: c = CFG.mk_from_lines(lines)
    sage: print c
    CFG([('LS', []), ('L0', ['LS']), ('L1', ['L0']), \
    ('L2', ['L0']), ('L3', ['L2']), ('L4', ['L3', 'L2', 'L1']), \
    ('L5', ['L4'])])
    sage: c.compute_paths(max_loop=1)

    sage: print sorted(c.sids)
    ['L0', 'L1', 'L2', 'L3', 'L4', 'L5', 'LS']

    sage: CFG.get_paths('L5',c,{},10,1)
    [['LS', 'L0', 'L2', 'L3', 'L4', 'L5'], \
    ['LS', 'L0', 'L2', 'L4', 'L5'], \
    ['LS', 'L0', 'L1', 'L4', 'L5']]

    sage: c.paths['L4']
    [['LS', 'L0', 'L2', 'L3', 'L4'], \
    ['LS', 'L0', 'L2', 'L4'], ['LS', 'L0', 'L1', 'L4']]

    sage: c = CFG(OrderedDict([('LS', []), ('L0', ['LS']), \
    ('L1', ['L0']), ('L2', ['L0']), ('L3', ['L2']), ('L4', \
    ['L3', 'L2', 'L1']), ('L5', ['L4'])]))
    sage: print c
    CFG([('LS', []), ('L0', ['LS']), ('L1', ['L0']), \
    ('L2', ['L0']), ('L3', ['L2']), ('L4', ['L3', 'L2', 'L1']), \
    ('L5', ['L4'])])

    sage: d = CFG([['LS', []], ['L0', ['LS']], ['L1', ['L0']], \
    ['L2', ['L0']], ['L3', ['L2']], ['L4', ['L3', 'L2', 'L1']], \
    ['L5', ['L4']]])
    sage: assert c==d

    sage: lines = [\
    'LS',\
    'L0 LS',\
    'L4 L3',\
    'L4 L2    L1   L3',\
    'L1 L0',\
    'L2 L0',\
    'L2    L0',\
    'L3 L2',\
    'L5 L4']

    sage: c = CFG.mk_from_lines(lines)
    sage: print c
    CFG([('LS', []), ('L0', ['LS']), ('L4', ['L3', 'L2', 'L1']), \
    ('L1', ['L0']), ('L2', ['L0']), ('L3', ['L2']), ('L5', ['L4'])])

    sage: print c.__str__(as_lines=True)
    LS
    L0 LS
    L4 L3 L2 L1
    L1 L0
    L2 L0
    L3 L2
    L5 L4
    """    
    def __init__(self,d):            
        OrderedDict.__init__(self,d)
        if __debug__:
            assert self, 'empty cfg'

    def __str__(self, as_lines=False):
        if as_lines:
            lines = ['{} {}'.format(loc,CM.str_of_list(prevs,' '))
                     for loc,prevs in self.iteritems()]
            lines = [line.strip() for line in lines]
            return CM.str_of_list(lines,'\n')
        else:
            return super(CFG,self).__str__()
        
    @staticmethod
    def mk_from_lines(lines):
        """
        Make a CFG from list of strings (e.g., results from CIL)
        These lines have the form
        l prev1 prev2  where
        previ are's previous locs of l
        """
        rs = (l.split() for l in lines)
        prevs = OrderedDict()
        for l in rs:
            k = l[0]
            cs = l[1:]
            if k in prevs:
                cs = [c for c in cs if c not in prevs[k]]
                prevs[k].extend(cs)
            else:
                prevs[k]=cs

        return CFG(prevs)
    
    def compute_paths(self, max_loop=0):
        self.paths = OrderedDict()

        paths = [(sid, CFG.get_paths(sid,self,{},20,max_loop)) for sid in self]
        invalids,paths = CM.vpartition(paths,lambda (s,p): len(p)>0)
        self.paths = OrderedDict(paths)

        if invalids:
            logger.warn("ignore {} sids without paths".format(len(invalids)))
            logger.detail(CM.str_of_list(s for s,_ in invalids))

        #locations (sids) in program
        self.sids = set(self.paths)
                           

    @staticmethod
    def get_paths(sid, preds_d, visited, recur_call, max_loop):
        """
        compute coverage paths from predecessors
        'max_loop' sets the maximum number of loop unrolling.
        E.g., set to 1 goes through each loop once.

        sage: CFG.get_paths(5,OrderedDict([(5,[4]),(4,[3,5]),(3,[2]),(2,[1,3]),(1,[])]),{},10,1)
        [[1, 2, 3, 4, 5], [1, 2, 3, 2, 3, 4, 5], [1, 2, 3, 4, 5, 4, 5]]

        sage: CFG.get_paths(5,OrderedDict([(5,[4]),(4,[3,5]),(3,[2]),(2,[1,3]),(1,[])]),{},10,2)
        [[1, 2, 3, 4, 5],
        [1, 2, 3, 2, 3, 4, 5],
        [1, 2, 3, 2, 3, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 4, 5],
        [1, 2, 3, 2, 3, 4, 5, 4, 5],
        [1, 2, 3, 4, 5, 4, 5, 4, 5],
        [1, 2, 3, 2, 3, 4, 5, 4, 5, 4, 5]]
        
        sage: CFG.get_paths(5,OrderedDict([(5,[4]),(4,[3,5]),(3,[2]),(2,[1,3]),(1,[])]),{},10,0)
        [[1, 2, 3, 4, 5]]

        #this is a strange case because it doesn't lead back to the start
        #happens in vsftpd
        sage: CFG.get_paths(3,OrderedDict([(3,[2]),(2,[1]),(1,[3])]),{},10,0)
        []
        """
        if recur_call < 0:
            return [[sid]]
        
        if sid not in visited:
            visited[sid] = 0
        else:
            visited[sid] += 1
            if visited[sid] > max_loop:
                return [[None]]

        try:
            preds = preds_d[sid]
        except KeyError:
            preds = []
            
        if len(preds) == 0:
            return [[sid]]
        else:
            rs = []
            for p in preds:
                paths = CFG.get_paths(p,preds_d,visited,recur_call-1,max_loop)
                for path in paths:
                    visited[p]=0 #reset
                    if None not in path:
                        rs.append(path+[sid])
                        
            return rs

    
class Config(CC.Config):

    def get_mincover(self,sids):
        """
        Greedy method to get a subset of self that covers sids.
        Return min cover and its coverage.

        sage: random.seed(1)

        sage: p = Pop([Config.mk([('x',1)],cov=[1,4,5]), Config.mk([('x',2)],cov=[1,2,4,5]),Config.mk([('x',3)],cov=[1,2,3,4])])

        sage: m,s = p.get_mincover(set([1,2,3,4,5])); print m; print s
        1. x=1, cov (3): 1, 4, 5
        2. x=3, cov (4): 1, 2, 3, 4
        set([])

        sage: m,s = p.get_mincover(set([3,5])); print m; print s
        1. x=1, cov (3): 1, 4, 5
        2. x=3, cov (4): 1, 2, 3, 4
        set([])

        sage: m,s = p.get_mincover(set([1,2,5,7,8])); print m; print s
        1. x=2, cov (4): 1, 2, 4, 5
        set([8, 7])

        sage: m,s = p.get_mincover(set([8])); print m; print s
        None
        set([8])

        sage: p = Pop([Config.mk([('x',1)],cov=[1,2,4,5]),Config.mk([('x',2)],cov=[1,2,4,5]),Config.mk([('x',3)],cov=[1,2,3,4])])
        sage: m,s = p.get_mincover(set([1,2,3,4,5])); print m; print s
        1. x=1, cov (4): 1, 2, 4, 5
        2. x=3, cov (4): 1, 2, 3, 4
        set([])

        sage: p = Pop([Config.mk([('x',1)],cov=[1,2,3]),Config.mk([('x',2)],cov=[1,2,4,5]),Config.mk([('x',3)],cov=[1,2,3,4])])
        sage: m,s = p.get_mincover(set([1,2,3,4,5])); print m; print s
        1. x=3, cov (4): 1, 2, 3, 4
        2. x=2, cov (4): 1, 2, 4, 5
        set([])

        sage: p = Pop([Config.mk([('x',1)],cov=[1,4,5]), Config.mk([('x',2)],cov=[2,4,6]),Config.mk([('x',3)],cov=[1,3,4,7])])
        sage: m,s = p.get_mincover(set([1,2,3,4,5,6,7,8])); print m; print s
        1. x=1, cov (3): 1, 4, 5
        2. x=3, cov (4): 1, 3, 4, 7
        3. x=2, cov (3): 2, 4, 6
        set([8])
        """
        
        if __debug__:
            assert isinstance(sids,set)

        cs = set(self) #make a copy
        mincover = set()

        while sids:
            #greedy
            cs_ = [(c,len(sids-c.cov)) for c in cs]
            if len(cs_) == 0: #used everything
                break

            config,n_uncovers = min(cs_, key=lambda (_,n_uncovers): n_uncovers)

            if n_uncovers == len(sids):  #nothing contributes
                break

            mincover.add(config)
            cs.remove(config)
            sids = sids - config.cov
        
        return mincover, sids

class Configs_d(CC.Configs_d):
    pass

class Fits_d(CC.CustDict):
    def __setitem__(self, config, fits):
        if __debug__:
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
        if __debug__:
            assert isinstance(dom, CC.Dom), dom
            assert isinstance(cfg, CFG), cfg
            assert callable(get_cov), get_cov
        
        self.dom = dom
        self.cfg = cfg
        self.cfg.compute_paths()
        self.sids = self.cfg.sids
        self.get_cov = get_cov
        self.z3db = self.dom.z3db
        logger.debug("cfg {}, sids {}"
                     .format(len(self.cfg),len(self.sids)))

    def eval_configs(self, configs):
        if __debug__:
            assert (isinstance(configs, list) and
                    all(isinstance(c, Config) for c in configs)
                    and configs), configs
        st = time()
        results = CC.eval_configs(configs, self.get_cov)
        cconfigs_d = CC.Configs_d()
        for c,rs in results:
            cconfigs_d[c] = rs
        return cconfigs_d, time() - st

    @staticmethod
    def mk_configs(n, f):
        """
        Generic method to create at most n configs.
        """
        if __debug__:
            assert isinstance(n,int) and n > 0, n
            assert callable(f), f
            
        pop = OrderedDict()
        for _ in range(n):
            c = f()

            c_iter = 0
            while c in pop and c_iter < 3:
                c_iter += 1
                c = f()

            if c not in pop:
                pop[c]=None
                
        if __debug__:
            assert len(pop) <= n, (len(pop), n)

        pop = pop.keys()
        
        return pop

    @staticmethod
    def mk_config_shuffle(config, dom, n=1):
        """
        Create a new config by suffling n values in config
        
        >>> ks = 'a b c d e f'.split()
        >>> vs = ['0 1 3 4', '0 1', '0 1', '0 1', '0 1', '0 1 2']
        >>> dom = CC.Dom(zip(ks, [frozenset(v.split()) for v in vs]))
        >>> c = Config(zip(ks, '0 1 0 1 0 1'.split()))
        >>> print c
        a=0 b=1 c=0 d=1 e=0 f=1
        >>> random.seed(0)
        >>> configs = [IGa.mk_config_shuffle(c, dom, 1) for _ in range(3)]
        >>> for c in configs: print c
        a=0 b=1 c=0 d=1 e=0 f=2
        a=0 b=1 c=1 d=1 e=0 f=1
        a=0 b=1 c=0 d=0 e=0 f=1
        >>> random.seed(0)
        >>> print IGa.mk_config_shuffle(c, dom, 4)
        a=3 b=0 c=0 d=1 e=0 f=0
        """
        if __debug__:
            assert isinstance(config, Config), config
            assert isinstance(dom, CC.Dom), dom
            assert len(config) == len(dom), (len(config), len(dom))
            assert 0 < n < len(dom), n
            
        ks = set(random.sample(dom.keys(), n))
        settings = []
        for k, v in config.iteritems():
            if k in ks:
                v = random.choice(list(set(dom[k]) - set([v])))
            settings.append((k,v))
        return Config(settings)
    
    def go(self, seed, sids, tmpdir):
        if __debug__:
            assert isinstance(seed, float), seed
            assert (sids and isinstance(sids, set) and
                    all(sid in self.sids for sid in sids)), sids
            assert isinstance(tmpdir, str), tmpdir
            
        random.seed(seed)
        logger.debug("seed: {}, tmpdir: {}".format(seed,tmpdir))        

        configs_d  = Configs_d()
        fits_d = Fits_d()
        covs_s = set()

        sids = set(sids)
        remains = list(sorted(sids))
        while remains:
            sid = remains[0]
            _, covs_s_ = self.go_sid(sid, configs_d, fits_d)
            #len(configs_d) might not be the same as len(fits_d)
            #b/c sometime we found what we search for without
            #having to compute fitness
            for s in covs_s_:
                if s not in covs_s:
                    covs_s.add(s)
                
            remains = [s for s in remains[1:] if s not in covs_s]
            logger.info('found {}/{} sids, uncovered {}'
                        .format(len([s for s in covs_s if s in sids]),
                                len(sids),
                                len(remains)))
        
        solved = set(s for s in sids if s in covs_s)
        unsolved = set(s for s in sids if s not in covs_s)
        is_success = solved == sids

        if not is_success:
            logger.debug('unsolved {}/{}: {}'.format(
                len(unsolved),len(sids),', '.join(sorted(unsolved))))
            
        logger.debug('{}, seed {}, sids {}/{} (covs {}), configs {}/{}'
                     .format('success' if is_success else 'fail',
                             seed,
                             len(solved), len(sids), len(covs_s),
                             len(configs_d), self.dom.siz))

        return is_success, solved, configs_d
        
    def go_sid(self, sid, configs_d, fits_d):
        if __debug__:
            assert isinstance(sid, str), sid
            assert isinstance(configs_d, Configs_d), configs_d
            assert isinstance(fits_d, Fits_d), fits_d

        def cache(configs,config_s,cov_s):
            for c in configs:
                config_s.add(c)
                for s in configs_d[c]:
                    if s not in cov_s:
                        cov_s.add(s)

        max_stuck = 3
        cur_stuck = 0
        max_exploit = 0.7
        cur_exploit = max_exploit
        cur_iter = 0
        xtime_total = 0.0
        config_s = set()
        cov_s = set()
        config_siz = None
        paths = None
        
        #begin
        st = time()
        while sid not in cov_s and cur_stuck <= max_stuck:
            cur_iter += 1
            logger.debug("sid '{}' iter {} stuck {} exploit {}"
                         .format(sid,cur_iter,cur_stuck,cur_exploit))

            if cur_iter == 1:
                configs = self.dom.gen_configs_tcover1(config_cls=Config)
                configs_d_, xtime = self.eval_configs(configs)
                for c in configs_d_:
                    if c not in configs_d:
                        configs_d[c] = configs_d_[c]                        
                xtime_total += xtime        
                cache(configs, config_s, cov_s)
                continue

            if cur_iter == 2:
                config_siz = max(len(self.dom), self.dom.max_fsiz)
                paths = self.cfg.paths[sid]
                IGa.compute_fits(sid, paths, configs, configs_d, fits_d)
            
            cur_best_fit, cur_best = IGa.get_best(sid, configs, fits_d)
            cur_avg_fit = IGa.get_avg(sid, configs, fits_d)
            cur_ncovs = len(cov_s)
            
            logger.debug("configs {}, fit best {} avg {} "
                         .format(len(configs_d), cur_best_fit, cur_avg_fit))
            logger.debug(str(cur_best))
            
            #gen new configs
            freqs = IGa.get_freqs(sid, configs, fits_d, self.dom)
            configs = IGa.mk_configs(
                int(config_siz/2), lambda: IGa.tourn_select(freqs, cur_exploit))

            if len(configs) < config_siz:
                for _ in range(config_siz - len(configs)):
                    config = IGa.mk_config_shuffle(cur_best, self.dom, n=1)
                    configs.append(config)
                    
            if __debug__:
                assert len(configs) == config_siz, (len(configs), config_siz)

            configs_d_, xtime = self.eval_configs(configs)
            for c in configs_d_:
                if c not in configs_d:
                    configs_d[c] = configs_d_[c]
            xtime_total += xtime
            cache(configs,config_s,cov_s)            
            IGa.compute_fits(sid, paths, configs, configs_d, fits_d)
            #elitism
            if cur_best not in configs:
                configs.append(cur_best)
                
            best_fit, _ = IGa.get_best(sid, configs, fits_d)
            avg_fit = IGa.get_avg(sid, configs, fits_d)
            better_best = best_fit > cur_best_fit
            better_avg = avg_fit > cur_avg_fit
            better_cov = len(cov_s) > cur_ncovs
            
            if better_cov or better_best or better_avg:
                cur_stuck = 0
                cur_exploit = max_exploit
            else:
                cur_stuck += 1
                cur_exploit -= max_exploit / max_stuck

                if cur_exploit < 0:
                    cur_exploit = 0

        logger.debug("* sid '{}': {} ({}s), iters {}, configs {}, covs {}"
                     .format(sid,
                             'found' if sid in cov_s else 'not found',
                             time() - st,
                             cur_iter,
                             len(config_s),
                             len(cov_s)))

        if sid in cov_s:
            config = CM.find_first(config_s, lambda c: sid in configs_d[c])
            logger.debug(str(config))
            
        return config_s, cov_s

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
    def get_freqs(sid, configs, fits_d, dom):
        """
        Returns a list of variable names and their fitness scores and occurrences.
        Used later to create new configuration (i.e., those with high scoes and 
        appear frequently are likely chosen).

        >>> ks = 'a b c d e f'.split()
        >>> vs = ['0 1 3 4', '0 1', '0 1', '0 1', '0 1', '0 1 2']
        >>> dom = CC.Dom(zip(ks, [frozenset(v.split()) for v in vs]))
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
        >>> configs = IGa.mk_configs(10, lambda: IGa.tourn_select(freqs, 0.9))
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
        >>> configs = IGa.mk_configs(10, lambda: IGa.tourn_select(freqs, 0.1))
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
            assert isinstance(dom, CC.Dom), dom
            

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
            if exploit_rate:
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
