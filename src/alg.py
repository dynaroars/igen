from time import time
import os.path
import random

import vcommon as CM
import config_common as CC

from config import (DTrace, Dom, Config, Core, MCore, SCore, PNCore,
                Cores_d, Mcores_d)

import settings
mlog = CM.getLogger(__name__, settings.logger_level)

class IGen(object):
    """
    Main algorithm
    """
    def __init__(self, dom, get_cov, sids=None):
        assert isinstance(dom, Dom), dom
        assert callable(get_cov), get_cov
        assert not sids or CC.is_cov(sids), sids
            
        self.dom = dom
        self.get_cov = get_cov
        self.sids = sids
        self.z3db = CC.Z3DB(self.dom)

        self.kconstraints = dom.get_kconstraints(self.z3db)
        
    def go(self, seed, rand_n=None, econfigs=None, tmpdir=None):
        """
        rand_n = None: use default CEGIR mode
        rand_n = 0  : use init configs
        rand_n > 0  : use rand_n configs
        rand_n < 0  : use all possible configs
        """
        assert isinstance(seed,(float, int)), seed
        assert rand_n is None or isinstance(rand_n, int), rand_n
        assert not econfigs or isinstance(econfigs, list), econfigs
        assert isinstance(tmpdir, str) and os.path.isdir(tmpdir), tmpdir
            
        random.seed(seed)
        mlog.info("seed: {}, tmpdir: {}".format(seed, tmpdir))

        DTrace.save_pre(seed, self.dom, tmpdir)

        #some settings
        cur_iter = 1
        min_stren = 1
        cur_min_stren = min_stren
        cur_stuck = 0
        max_stuck = 3
        cores_d, configs_d, covs_d = Cores_d(), CC.Configs_d(), CC.Covs_d()
        sel_core = SCore.mk_default()
        ignore_sel_cores = set()

        #begin
        st = time()
        ct = st
        xtime_total = 0.0

        cconfigs_d = CC.Configs_d()
        configs = []
        xtime = 0.0

        #init configs
        if econfigs:
            for c, cov in econfigs:
                c = Config(c)
                if cov is None:
                    configs.append(c)
                else:
                    cconfigs_d[c] = cov
                    
        configs = [c for c in configs if c not in cconfigs_d]

        mlog.info("{} existing configs: {} evaled, {} not evaled"
                     .format(len(econfigs), len(cconfigs_d), len(configs)))
        
        if not configs:
            configs_ = self.gen_configs_init(rand_n)
            configs.extend(configs_)
            
        if configs:
            cconfigs_d_, xtime = self.eval_configs(configs)
            xtime_total += xtime
            for c in cconfigs_d_:
                assert c not in cconfigs_d
                cconfigs_d[c]  = cconfigs_d_[c]
                
        mlog.info("init configs {}".format(len(cconfigs_d)))
        #CC.pause()
        
        new_covs, new_cores = Infer.infer_covs(
            cores_d, cconfigs_d, configs_d, covs_d, self.dom, self.sids)
            
        while True:
            ct_ = time(); itime = ct_ - ct; ct = ct_
            dtrace = DTrace(
                cur_iter, itime, xtime,
                len(configs_d), len(covs_d), len(cores_d),
                cconfigs_d,
                new_covs, new_cores,
                sel_core,
                cores_d)
            dtrace.show(self.dom, self.z3db)
            DTrace.save_iter(cur_iter, dtrace, tmpdir)

            if rand_n is not None:
                break

            cur_iter += 1
            sel_core, configs = self.gen_configs_iter(
                set(cores_d.values()), ignore_sel_cores, cur_min_stren, configs_d)

            if sel_core is None:
                cur_iter -= 1
                mlog.info('done after iter {}'.format(cur_iter))
                break

            assert configs, configs
                
            cconfigs_d, xtime = self.eval_configs(configs)
            xtime_total += xtime
            new_covs, new_cores = Infer.infer_covs(
                cores_d, cconfigs_d, configs_d, covs_d, self.dom, self.sids)

            if new_covs or new_cores: #progress
                cur_stuck = 0
                cur_min_stren = min_stren
            else: #no progress
                cur_stuck += 1
                if cur_stuck > max_stuck:
                    cur_stuck = 0
                    cur_min_stren += 1
                    mlog.debug('cur_min_stren is {}'.format(cur_min_stren))

        #postprocess
        #only analyze sids
        if self.sids:
            cores_d_, covs_d_ = Cores_d(), CC.Covs_d()
            for sid in self.sids:
                if sid in cores_d:
                    cores_d_[sid] = cores_d[sid]
                    for c in covs_d[sid]:
                        covs_d_.add(sid, c)
            pp_cores_d = cores_d_.analyze(self.dom, self.z3db, covs_d_)
            _ = pp_cores_d.merge(self.dom, self.z3db, show_detail=True)
        else:
            pp_cores_d = cores_d.analyze(self.dom, self.z3db, covs_d)
            _ = pp_cores_d.merge(self.dom, self.z3db, show_detail=True)
        
        itime_total = time() - st
        mlog.info(DTrace.str_of_summary(
            seed, cur_iter, itime_total, xtime_total,
            len(configs_d), len(covs_d), tmpdir))
        mlog.info("Done (seed {}, test {})"
                    .format(seed, random.randrange(100)))
        DTrace.save_post(pp_cores_d, itime_total, tmpdir)
        
        return pp_cores_d, cores_d, configs_d, covs_d, self.dom

    #Shortcuts
    def go_full(self, tmpdir=None):
        return self.go(seed=0, rand_n=-1, tmpdir=tmpdir)
                       
    def go_rand(self,rand_n, seed=None, econfigs=None, tmpdir=None):
        return self.go(seed=seed, rand_n=rand_n, econfigs=econfigs, tmpdir=tmpdir)

    #Helper functions
    def eval_configs(self, configs):
        assert isinstance(configs, list) and configs, configs
        assert all(isinstance(c, Config) for c in configs), configs
        
        st = time()
        results = Config.eval(configs, self.get_cov, self.kconstraints,
                              self.z3db)
        cconfigs_d = CC.Configs_d()
        for c, rs in results:
            cconfigs_d[c] = rs
        return cconfigs_d, time() - st

    def gen_configs_init(self, rand_n):
        if not rand_n: #None or 0
            configs = self.dom.gen_configs_tcover1(config_cls=Config)
            mlog.info("gen {} configs using tcover 1".format(len(configs)))
        elif rand_n > 0 and rand_n < self.dom.siz:        
            configs = self.dom.gen_configs_rand_smt(
                rand_n, self.z3db, config_cls=Config)
            mlog.info("gen {} rand configs".format(len(configs)))
        else:
            configs = self.dom.gen_configs_full(config_cls=Config)
            mlog.info("gen all {} configs".format(len(configs)))

        configs = list(set(configs))
        assert configs, 'no initial configs created'
        return configs
        
    def gen_configs_iter(self, cores, ignore_sel_cores, min_stren, configs_d):
        assert (isinstance(cores, set) and 
                all(isinstance(c, PNCore) for c in cores)), cores
        assert (isinstance(ignore_sel_cores, set) and 
                all(isinstance(c, SCore) for c in ignore_sel_cores)),\
                ignore_sel_cores
        assert isinstance(configs_d, CC.Configs_d),configs_d

        configs = []
        while True:
            sel_core = self.select_core(cores, ignore_sel_cores, min_stren)
            if sel_core is None:
                break

            configs = self.dom.gen_configs_cex(sel_core, configs_d, self.z3db)
            configs = list(set(configs)) 
            if configs:
                break
            else:
                mlog.info("no cex's created for sel_core {}, try new core"
                          .format(sel_core))

        #self_core -> configs
        assert not sel_core or configs, (sel_core,configs)
        assert all(c not in configs_d for c in configs), configs

        return sel_core, configs

    @staticmethod
    def select_core(pncores, ignore_sel_cores, min_stren):
        """
        Returns either None or SCore
        """
        assert (all(isinstance(c, PNCore) for c in pncores) and
                pncores), pncores
        assert (isinstance(ignore_sel_cores, set) and
                all(isinstance(c, SCore) for c in ignore_sel_cores)),\
            ignore_sel_cores

        sel_cores = []
        for (pc,pd,nc,nd) in pncores:
            #if can add pc then don't cosider pd (i.e., refine pc first)
            if pc and (pc,None) not in ignore_sel_cores:
                sc = SCore((pc, None))
                if pd is None: sc.set_keep()
                sel_cores.append(sc)
                    
            elif pd and (pd,pc) not in ignore_sel_cores:
                sc = SCore((pd,pc))
                sel_cores.append(sc)

            if nc and (nc, None) not in ignore_sel_cores:
                sc = SCore((nc,None))
                if nd is None: sc.set_keep()
                sel_cores.append(sc)

            elif nd and (nd,nc) not in ignore_sel_cores:
                sc = SCore((nd,nc))
                sel_cores.append(sc)
                
        sel_cores = [c for c in sel_cores if c.sstren >= min_stren]

        if sel_cores:
            sel_core = max(sel_cores, key=lambda c: (c.sstren, c.vstren))
            ignore_sel_cores.add(sel_core)
        else:
            sel_core = None

        return sel_core


    
#Inference algorithm
class Infer(object):
    @classmethod
    def infer(cls, configs, core, dom):
        """
        Approximation in *conjunctive* form
        """
        assert (all(isinstance(c, Config) for c in configs)
                and configs), configs
        assert Core.maybe_core(core), core
        assert isinstance(dom, Dom), dom
        
        if core is None:  #not yet set
            core = min(configs, key=lambda c: len(c))
            core = Core((k, frozenset([v])) for k,v in core.iteritems())

        def f(k,s,ldx):
            s_ = set(s)
            for config in configs:
                if k in config:
                    s_.add(config[k])
                    if len(s_) == ldx:
                        return None
                else:
                    return None
            return s_

        vss = [f(k,vs,len(dom[k])) for k,vs in core.iteritems()]
        core = Core((k,frozenset(vs)) for k,vs in zip(core,vss) if vs)
        return core  

    @classmethod
    def infer_cache(cls, core, configs, dom, cache):
        assert core is None or isinstance(core, Core), core
        assert (configs and
                all(isinstance(c,Config) for c in configs)), configs
        assert isinstance(dom,Dom),dom
        assert isinstance(cache,dict),cache

        configs = frozenset(configs)
        key = (core,configs)
        if key not in cache:
            cache[key] = cls.infer(configs,core,dom)
        return cache[key]

    @classmethod
    def infer_sid(cls,sid,core,cconfigs_d,configs_d,covs_d,dom,cache):
        assert isinstance(sid,str),sid
        assert isinstance(core, PNCore),core
        assert (cconfigs_d and
                isinstance(cconfigs_d, CC.Configs_d)), cconfigs_d
        assert isinstance(configs_d, CC.Configs_d), configs_d
        assert isinstance(covs_d, CC.Covs_d),covs_d
        assert isinstance(dom,Dom),dom        
        assert isinstance(cache,dict),cache
        
        def _f(configs, cc, cd, _b):
            new_cc,new_cd = cc,cd
            if configs:
                new_cc = cls.infer_cache(cc,configs,dom,cache)

            #TODO: this might be a bug, if new_cc is empty,
            #then new_cd won't be updated
            if new_cc:
                configs_ = [c for c in _b() if c.c_implies(new_cc)]
                if configs_:
                    new_cd = cls.infer_cache(cd,configs_,dom,cache)
                    if new_cd:
                        new_cd = Core((k,v) for (k,v) in new_cd.iteritems()
                                      if k not in new_cc)

            return new_cc, new_cd

        pc, pd, nc, nd = core
        
        pconfigs = [c for c in cconfigs_d if sid in cconfigs_d[c]]
 
        if nc is None:
            #never done nc, so has to consider all traces
            nconfigs = [c for c in configs_d if sid not in configs_d[c]]
        else:
            #done nc, so can do incremental traces
            nconfigs = [c for c in cconfigs_d if sid not in cconfigs_d[c]]
            
        _b = lambda: [c for c in configs_d if sid not in configs_d[c]]
        pc_,pd_ = _f(pconfigs, pc, pd, _b)
        
        _b = lambda: covs_d[sid]
        nc_,nd_ = _f(nconfigs, nc, nd, _b)
        return PNCore((pc_, pd_, nc_, nd_))

    @classmethod
    def infer_covs(cls, cores_d, cconfigs_d, configs_d, covs_d, dom, sids=None):
        assert isinstance(cores_d, Cores_d), cores_d
        assert isinstance(cconfigs_d, CC.Configs_d) and cconfigs_d, cconfigs_d
        assert isinstance(configs_d, CC.Configs_d), configs_d        
        assert all(c not in configs_d for c in cconfigs_d), cconfigs_d
        assert isinstance(covs_d, CC.Covs_d), covs_d
        assert isinstance(dom, Dom), dom
        assert not sids or CC.is_cov(sids), sids
            
        sids_ = set(cores_d.keys())
        #update configs_d and covs_d
        for config in cconfigs_d:
            for sid in cconfigs_d[config]:
                sids_.add(sid)
                covs_d.add(sid, config)
                
            assert config not in configs_d, config
            configs_d[config] = cconfigs_d[config]

        #only consider interested sids
        if sids:
            sids_ = [sid for sid in sids_ if sid in sids]
            
        cache = {}
        new_covs, new_cores = set(), set()  #updated stuff
        for sid in sorted(sids_):
            if sid in cores_d:
                core = cores_d[sid]
            else:
                core = PNCore.mk_default()
                new_covs.add(sid) #progress

            core_ = cls.infer_sid(
                sid, core, cconfigs_d, configs_d, covs_d, dom, cache)
                
            if not core_ == core: #progress
                new_cores.add(sid)
                cores_d[sid] = core_

        return new_covs, new_cores

    
