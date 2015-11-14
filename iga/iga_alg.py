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


class Dom(CC.Dom):
    pass

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
        
        if mincover:
            mincover = Pop(mincover)
            return mincover,sids
        else:
            return None,sids


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

    def __str__(self,sid):
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
            assert isinstance(dom, Dom), dom
            assert isinstance(cfg, CFG), cfg
            assert callable(get_cov), get_cov
        
        self.dom = dom
        self.cfg = cfg
        self.cfg.compute_paths()
        self.sids = self.cfg.sids
        self.get_cov = get_cov
        self.z3db = self.dom.z3db
        logger.info("cfg {}, sids {}"
                    .format(len(self.cfg),len(self.sids)))

    def compute_covs(self, configs, configs_d):
        """
        It's OK for duplicatins in configs (similar individuals in pop)
        """
        if __debug__:
            assert (configs and
                    all(isinstance(c,Config) for c in configs)), configs

        st = time()
        for c in configs:
            if c in configs_d:
                continue
            sids,_ = self.get_cov(c)
            configs_d[c]= sids
        return time() - st

    def go(self, seed, sids, tmpdir):
        if __debug__:
            assert isinstance(seed,float), seed
            assert (sids and isinstance(sids, set) and
                    all(sid in self.sids for sid in sids)), sids
            assert isinstance(tmpdir,str), tmpdir
            
        random.seed(seed)
        logger.info("seed: {}, tmpdir: {}".format(seed,tmpdir))        

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
            
        logger.info('{}, seed {}, sids {}/{} (covs {}), configs {}/{}'
                    .format('success' if is_success else 'fail',
                            seed,
                            len(solved), len(sids), len(covs_s),
                            len(configs_d), self.dom.siz))

        return is_success, solved, configs_d
        
    def go_sid(self, sid, configs_d, fits_d):
        if __debug__:
            assert isinstance(sid,str), sid
            assert isinstance(configs_d, Configs_d), configs_d
            assert isinstance(fits_d,Fits_d), fits_d

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
                xtime = self.compute_covs(configs, configs_d)
                xtime_total += xtime        
                cache(configs,config_s,cov_s)
                continue

            if cur_iter == 2:
                config_siz = max(len(self.dom),self.dom.max_fsiz)
                paths = self.cfg.paths[sid]
                IGa.compute_fits(sid, paths, configs, configs_d, fits_d)
            
            cur_best_fit, cur_best = IGa.get_best(sid, configs, fits_d)
            cur_avg_fit = IGa.get_avg(sid, configs, fits_d)
            cur_ncovs = len(cov_s)
            
            logger.debug("configs {} fit best {} avg {} "
                         .format(len(configs_d), cur_best_fit, cur_avg_fit))
            
            #gen new configs
            freqs = IGa.get_freqs(sid, configs, fits_d, self.dom)
            configs = [IGa.tourn_sel(freqs, cur_exploit)
                       for _ in range(config_siz)]
            if __debug__:
                assert len(configs) == config_siz, \
                    (len(configs), config_siz)

            xtime = self.compute_covs(configs, configs_d)
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
    def get_tourn_siz(sample_siz,exploit_rate):
        """
        >>> [IGa.get_tourn_siz(x,y) for x,y in [(10,.8),(10,.9),(3,.8),(3,.5)]]
        [8, 9, 3, 2]
        """
        if __debug__:
            assert sample_siz >= 1, sample_siz
        siz = int(math.ceil(sample_siz*exploit_rate))
        if siz > sample_siz:
            siz = sample_siz
        if siz < 1:
            siz = 1
        return siz
    
    @staticmethod
    def tourn_sel(rls,exploit_rate):
        def sample_f(rl,tourn_siz):
            if __debug__:
                assert isinstance(rl[0],tuple) and len(rl[0])==2
                #rl = [('val', score)]            
            s = (random.choice(rl) for _ in range(tourn_siz))
            return s
        
        settings = []
        for name,ranked_list in rls:
            assert len(ranked_list) >= 1
            
            tourn_siz = IGa.get_tourn_siz(len(ranked_list),exploit_rate)
            sample = sample_f(ranked_list,tourn_siz)
            val,_ = max(sample,key=lambda (val,rank):rank)
            settings.append((name,val))

        c = Config(settings)
        return c

    @staticmethod
    def get_best(sid, configs, fits_d):
        if __debug__:
            assert isinstance(sid,str), sid
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
    def get_avg(sid,configs,fits_d):
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
    def get_freqs(sid,configs,fits_d,dom):
        """
        >>> ns = 'ssl loc lis acc anon dual'.split()
        >>> vs = ['0 1 3 4', '0 1', '0 1', '0 1', '0 1', '0 1 2']
        >>> dom = Dom(zip(ns,[frozenset(v.split()) for v in vs]))

        >>> cs = [\
        ('0 1 0 1 0 0',2),\
        ('1 1 0 1 0 0',1),\
        ('1 0 1 0 1 0',1),\
        ('1 0 1 0 0 1',0),\
        ('0 1 0 1 0 1',2),\
        ('0 0 1 0 1 1',8)]

        >>> configs = [Config(zip(ns,c.split())) for c,f in cs]
        >>> fits = [{None:f} for _,f in cs]
        >>> fits_d = Fits_d()
        >>> for c,f in zip(configs,fits): fits_d[c]=f

        >>> print fits_d.__str__(None)
        1. ssl=0 loc=0 lis=1 acc=0 anon=1 dual=1 fit 8
        2. ssl=0 loc=1 lis=0 acc=1 anon=0 dual=0 fit 2
        3. ssl=0 loc=1 lis=0 acc=1 anon=0 dual=1 fit 2
        4. ssl=1 loc=0 lis=1 acc=0 anon=1 dual=0 fit 1
        5. ssl=1 loc=1 lis=0 acc=1 anon=0 dual=0 fit 1
        6. ssl=1 loc=0 lis=1 acc=0 anon=0 dual=1 fit 0

        >>> rls = IGa.analyze(None,fits_d,dom)
        >>> print rls
        [('ssl', [('3', -0.1), ('4', -0.1), ('1', 0), ('1', 1), ('1', 1), ('0', 2), ('0', 2), ('0', 8)]), ('loc', [('0', 0), ('0', 1), ('1', 1), ('1', 2), ('1', 2), ('0', 8)]), ('lis', [('1', 0), ('1', 1), ('0', 1), ('0', 2), ('0', 2), ('1', 8)]), ('acc', [('0', 0), ('0', 1), ('1', 1), ('1', 2), ('1', 2), ('0', 8)]), ('anon', [('0', 0), ('1', 1), ('0', 1), ('0', 2), ('0', 2), ('1', 8)]), ('dual', [('2', -0.1), ('1', 0), ('0', 1), ('0', 1), ('0', 2), ('1', 2), ('1', 8)])]



        # #Another example
        >>> ns = 'a b c d e'.split()
        >>> vs = ['0 1', '0 1', '0 1', '0 1', '0 1']
        >>> dom = OrderedDict(zip(ns,[v.split() for v in vs]))
        >>> cs = [\
        ('1 1 1 1 1', 3),\
        ('1 1 0 1 0', 6),\
        ('1 0 1 1 0', 6),\
        ('0 1 1 1 0', 6),\
        ('1 1 1 1 0', 6)]

        >>> configs = [Config(zip(ns,c.split())) for c,f in cs]
        >>> fits = [{'s':f} for _,f in cs]
        >>> fits_d = Fits_d()
        >>> for c,f in zip(configs,fits): fits_d[c]=f

        >>> print fits_d.__str__('s')
        1. a=0 b=1 c=1 d=1 e=0 fit 6
        2. a=1 b=0 c=1 d=1 e=0 fit 6
        3. a=1 b=1 c=1 d=1 e=0 fit 6
        4. a=1 b=1 c=0 d=1 e=0 fit 6
        5. a=1 b=1 c=1 d=1 e=1 fit 3
        
        >>> rls = IGa.analyze('s',fits_d,dom)
        >>> print rls
        [('a', [('0', 6), ('1', 6), ('1', 3), ('1', 6), ('1', 6)]), ('b', [('1', 6), ('0', 6), ('1', 3), ('1', 6), ('1', 6)]), ('c', [('1', 6), ('1', 6), ('1', 3), ('1', 6), ('0', 6)]), ('d', [('0', 2.9), ('1', 6), ('1', 6), ('1', 3), ('1', 6), ('1', 6)]), ('e', [('0', 6), ('0', 6), ('1', 3), ('0', 6), ('0', 6)])]

        # >>> random.seed(1)
        # >>> print Pop.mk_ts(rls,siz=10,tourn_siz=2,exploit_rate=1.0)
        # 1. a=1 b=1 c=1 d=1 e=1
        # 2. a=1 b=1 c=1 d=1 e=0
        # 3. a=1 b=1 c=0 d=1 e=0
        # 4. a=0 b=1 c=1 d=1 e=0
        # 5. a=1 b=0 c=1 d=1 e=0
        # 6. a=1 b=0 c=0 d=1 e=0
        # 7. a=0 b=1 c=0 d=1 e=0
        # 8. a=1 b=1 c=1 d=0 e=0

        """
        if __debug__:
            assert isinstance(sid,str), sid
            assert (configs is None or
                    (configs and
                    all(isinstance(c, Config) for c in configs))), configs
            assert fits_d and isinstance(fits_d, Fits_d), fits_d
            assert isinstance(dom, Dom), dom
            
        fits = [fits_d[c][sid] for c in configs]
        min_fit = min(fits)
        tiny = 0.1

        freqs = []
        for k,vs in dom.iteritems():
            cvs = [c[k] for c in configs]
            ts = zip(cvs,fits)
            #non-appearing values are also considered (with low fit)
            ts = [(v,min_fit-tiny) for v in vs if v not in cvs] + ts
            freqs.append((k,ts))

        return freqs

    @staticmethod
    def get_fscore(cov,path):
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
            fit = max(IGa.get_fscore(cov,path) for path in paths)
            if c not in fits_d:
                fits_d[c] = {sid:fit}
            else:
                if __debug__:
                    assert sid not in fits_d[c], (sid, fits_d[c])
                fits_d[c][sid] = fit


                
# if __name__ == "__main__":
#     import doctest
#     doctest.testmod()
    
