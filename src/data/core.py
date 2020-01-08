import pdb
import z3
import z3util

import data.base
import data.dom
import data.config

import helpers.vcommon
import settings

DBG = pdb.set_trace
mlog = helpers.vcommon.getLogger(__name__, settings.logger_level)


class Core(data.base.HDict):
    """
    >>> print Core()
    true

    >>> c = Core([('x',frozenset(['2'])),('y',frozenset(['1'])),('z',frozenset(['0','1']))])
    >>> print c
    x=2 y=1 z=0,1

    >>> print ', '.join(map(str_of_setting,c.settings))
    x=2, y=1, z=1, z=0

    >>> dom = data.dom.Dom([('x',frozenset(['1','2'])),\
    ('y',frozenset(['1'])),\
    ('z',frozenset(['0','1','2'])),\
    ('w',frozenset(['a','b','c']))\
    ])

    >>> print c.neg(dom)
    x=1 z=2

    >>> c = Core([('x',frozenset(['2'])),('z',frozenset(['0','1'])),('w',frozenset(['a']))])
    >>> print c.neg(dom)
    x=1 z=2 w=b,c

    """

    def __init__(self, core=data.base.HDict()):
        super(Core, self).__init__(core)

        assert all(data.base.is_csetting(s) for s in self.items()), self

    def __str__(self, delim=' '):
        if self:
            return delim.join(map(data.base.str_of_csetting, self.items()))
        else:
            return 'true'

    @property
    def settings(self):
        return [(k, v) for k, vs in self.items() for v in vs]

    def neg(self, dom):
        try:
            return self._neg
        except AttributeError:
            assert isinstance(dom, data.dom.Dom), dom
            ncore = ((k, dom[k] - self[k]) for k in self)
            self._neg = Core((k, vs) for k, vs in ncore if vs)
            return self._neg

    def z3expr(self, z3db, is_and):
        assert isinstance(z3db, data.dom.Z3DB)
        return z3db.expr_of_dict_dict(self, is_and)

    @staticmethod
    def maybe_core(c): return c is None or isinstance(c, Core)


class MCore(tuple):
    """
    Multiple (tuples) cores
    """

    def __new__(cls, cores):
        assert len(cores) == 2 or len(cores) == 4, cores
        assert all(Core.maybe_core(c) for c in cores), cores
        return super(MCore, cls).__new__(cls, cores)

    @property
    def settings(self):
        core = (c for c in self if c)
        return set(s for c in core for s in c.items())

    @property
    def values(self):
        core = (c for c in self if c)
        return set(s for c in core for s in c.values())

    @property
    def sstren(self): return len(self.settings)

    @property
    def vstren(self): return sum(map(len, self.values))


class SCore(MCore):
    def __new__(cls, mcsc):
        """
        mc: main core that will generate cex's
        sc (if not None): sat core that is satisfied by all generated cex'
        """
        mc, sc = mcsc
        assert mc is None or isinstance(mc, Core) and mc, mc
        # sc is not None => ...
        assert not sc or all(k not in mc for k in sc), sc
        return super(MCore, cls).__new__(cls, mcsc)

    def __init__(self, mcsc):
        self.keep = False

    def set_keep(self):
        """
        keep: if true then generated cex's with diff settings than mc 
        and also those that have the settings of mc
        """
        self.keep = True

    @property
    def mc(self): return self[0]

    @property
    def sc(self): return self[1]

    def __str__(self):
        ss = []
        if self.mc:
            ss.append("mc(keep): {}".format(self.mc))

        if self.sc:
            ss.append("sc: {}".format(self.sc))
        return '; '.join(ss)

    @classmethod
    def mk_default(cls):
        c = cls((None, None))
        return c


class PNCore(MCore):
    """
    >>> pc = Core([('x',frozenset(['0','1'])),('y',frozenset(['1']))])
    >>> pd = None
    >>> nc = Core([('z',frozenset(['1']))])
    >>> nd = None
    >>> pncore = PNCore((pc,pd,nc,nd))
    >>> print pncore
    pc: x=0,1 y=1; nc: z=1

    >>> dom = data.dom.Dom([('x',frozenset(['0','1','2'])),\
    ('y',frozenset(['0','1'])),\
    ('z',frozenset(['0','1'])),\
    ('w',frozenset(['0','1']))\
    ])
    >>> z3db = dom.z3db

    >>> print PNCore._get_str(pc,pd,dom,is_and=True)
    (x=0,1 & y=1)
    >>> print PNCore._get_expr(pc,pd,dom,z3db,is_and=True)
    And(Or(x == 1, x == 0), y == 1)

    >>> print PNCore._get_str(nd,nc,dom,is_and=False)
    z=0
    >>> print PNCore._get_expr(nd,nc,dom,z3db,is_and=False)
    z == 0
    >>> print pncore.z3expr(dom, z3db)
    And(And(Or(x == 1, x == 0), y == 1), z == 0)

    >>> pc = Core([])
    >>> pd = None
    >>> nc = None
    >>> nd = None
    >>> pncore = PNCore((pc,pd,nc,nd))

    >>> assert PNCore._get_str(pc,pd,dom,is_and=True) == 'true'
    >>> assert PNCore._get_str(nd,nc,dom,is_and=False) == 'true'
    >>> assert PNCore._get_expr(pc,pd,dom,z3db,is_and=True) is None
    >>> assert PNCore._get_expr(nd,nc,dom,z3db,is_and=True) is None
    >>> assert pncore.z3expr(dom, z3db) is None

    """

    def __new__(cls, pcpdncnd):
        return super(PNCore, cls).__new__(cls, pcpdncnd)

    @property
    def pc(self): return self[0]

    @property
    def pd(self): return self[1]

    @property
    def nc(self): return self[2]

    @property
    def nd(self): return self[3]

    @property
    def vtyp(self): return self._vtyp

    @vtyp.setter
    def vtyp(self, vt):
        assert isinstance(vt, str) and vt in 'conj disj mix'.split(), vt

        self._vtyp = vt

    @property
    def vstr(self): return self._vstr

    @vstr.setter
    def vstr(self, vs):
        assert isinstance(vs, str) and vs, vs

        self._vstr = vs

    @classmethod
    def mk_default(cls):
        c = cls((None, None, None, None))
        return c

    def __str__(self):
        try:
            return "{} ({})".format(self.vstr, self.vtyp)
        except AttributeError:
            ss = ("{}: {}".format(s, c) for s, c in
                  zip('pc pd nc nd'.split(), self) if c is not None)
            return '; '.join(ss)

    def verify(self, configs, dom):
        assert self.pc is not None, self.pc  # this never happens
        # nc is None => pd is None
        assert self.nc is not None or self.pd is None, (self.nc, self.nd)
        # assert (all(isinstance(c, Config) #VU: temp comment
        #             for c in configs) and configs), configs
        assert isinstance(dom, data.dom.Dom), dom

        pc, pd, nc, nd = self

        # traces => pc & neg(pd)
        assert not pc or all(c.c_implies(pc) for c in configs), pc

        if pd:
            pd_n = pd.neg(dom)
            if not all(c.d_implies(pd_n) for c in configs):
                mlog.info('pd {} invalid'.format(pd))
                pd = None

        # neg traces => nc & neg(nd)
        # pos traces => neg(nc & neg(nd))
        # pos traces => nd | neg(nc)
        if nc and not nd:
            nc_n = nc.neg(dom)
            if not all(c.d_implies(nc_n) for c in configs):
                mlog.info('nc {} invalid'.format(nc))
                nc = None
        elif not nc and nd:
            if not all(c.c_implies(nd) for c in configs):
                mlog.info('nd {} invalid'.format(nd))
                nd = None
        elif nc and nd:
            nc_n = nc.neg(dom)
            if not all(c.c_implies(nd) or
                       c.d_implies(nc_n) for c in configs):
                mlog.info('nc {} & nd {} invalid').format(nc, nd)
                nc = None
                nd = None

        return PNCore((pc, pd, nc, nd))

    @staticmethod
    def _get_expr(cc, cd, dom, z3db, is_and):
        assert Core.maybe_core(cc)
        assert Core.maybe_core(cd)
        assert isinstance(dom, data.dom.Dom)
        assert isinstance(z3db, data.dom.Z3DB)

        k = (cc, cd, is_and)
        if k in z3db.cache:
            return z3db.cache[k]

        fs = []
        if cc:
            f = cc.z3expr(z3db, is_and=True)
            fs.append(f)
        if cd:
            cd_n = cd.neg(dom)
            f = cd_n.z3expr(z3db, is_and=False)
            fs.append(f)

        myf = z3util.myAnd if is_and else z3util.myOr
        expr = myf(fs)

        z3db.add(k, expr)
        return expr

    @staticmethod
    def _get_str(cc, cd, dom, is_and):
        and_delim = ' & '
        or_delim = ' | '

        def _str(core, delim):
            s = core.__str__(delim)
            if len(core) > 1:
                s = '({})'.format(s)
            return s

        ss = []
        if cc:
            s = _str(cc, and_delim)
            ss.append(s)
        if cd:
            cd_n = cd.neg(dom)
            s = _str(cd_n, or_delim)
            ss.append(s)

        if ss:
            delim = and_delim if is_and else or_delim
            return delim.join(sorted(ss))
        else:
            return 'true'

    @staticmethod
    def _get_expr_str(cc, cd, dom, z3db, is_and):
        expr = PNCore._get_expr(cc, cd, dom, z3db, is_and)
        vstr = PNCore._get_str(cc, cd, dom, is_and)
        return expr, vstr

    def simplify(self, dom, z3db, do_firsttime=True):
        """
        Compare between (pc,pd) and (nc,nd) and return the stronger one.
        This will set either (pc,pd) or (nc,nd) to (None,None)

        if do_firstime is False then don't do any checking,
        essentialy this option is used for compatibility purpose

        Assumption: all 4 cores are verified

        inv1 = pc & not(pd)
        inv2 = not(nc & not(nd)) = nd | not(nc)
        """
        assert isinstance(dom, data.dom.Dom), dom
        assert isinstance(z3db, data.dom.Z3DB), z3db
        if __debug__:
            if do_firsttime:
                assert self.pc is not None, self.pc  # this never could happen
                # nc is None => pd is None
                assert self.nc is not None or self.pd is None, (
                    self.nc, self.nd)

        # pf = pc & neg(pd)
        # nf = neg(nc & neg(nd)) = nd | neg(nc)
        pc, pd, nc, nd = self

        # remove empty ones
        if not pc:
            pc = None
        if not pd:
            pd = None
        if not nc:
            nc = None
        if not nd:
            nd = None

        if pc is None and pd is None:
            expr, vstr = PNCore._get_expr_str(nd, nc, dom, z3db, is_and=False)
        elif nc is None and nd is None:
            expr, vstr = PNCore._get_expr_str(pc, pd, dom, z3db, is_and=True)
        else:
            pexpr, pvstr = PNCore._get_expr_str(pc, pd, dom, z3db, is_and=True)
            nexpr, nvstr = PNCore._get_expr_str(
                nd, nc, dom, z3db, is_and=False)

            assert pexpr is not None
            assert nexpr is not None

            if z3util.is_tautology(z3.Implies(pexpr, nexpr), z3db.solver):
                nc = None
                nd = None
                expr = pexpr
                vstr = pvstr
            elif z3util.is_tautology(z3.Implies(nexpr, pexpr), z3db.solver):
                pc = None
                pd = None
                expr = nexpr
                vstr = nvstr
            else:  # could occur when using incomplete traces
                mlog.warn("inconsistent ? {}\npf: {} ?? nf: {}"
                          .format(PNCore((pc, pd, nc, nd)), pexpr, nexpr))

                expr = z3util.myAnd([pexpr, nexpr])
                vstr = ','.join([pvstr, nvstr]) + '***'

        def _typ(s):
            # hackish way to get type
            if ' & ' in s and ' | ' in s:
                return 'mix'
            elif ' | ' in s:
                return 'disj'
            else:
                return 'conj'

        core = PNCore((pc, pd, nc, nd))
        core.vstr = vstr
        core.vtyp = _typ(vstr)

        return core, expr

    def is_simplified(self):
        return ((self.pc is None and self.pd is None) or
                (self.nc is None and self.nd is None))

    def z3expr(self, dom, z3db):
        """
        Note: z3 expr "true" is represented (and returned) as None
        """
        assert isinstance(dom, data.dom.Dom)
        assert isinstance(z3db, data.dom.Z3DB)

        if self in z3db.cache:
            return z3db.cache[self]

        pc, pd, nc, nd = self
        if pc is None and pd is None:
            expr = PNCore._get_expr(nd, nc, dom, z3db, is_and=False)
        elif nc is None and nd is None:
            expr = PNCore._get_expr(pc, pd, dom, z3db, is_and=True)
        else:
            pexpr = PNCore._get_expr(pc, pd, dom, z3db, is_and=True)
            nexpr = PNCore._get_expr(nd, nc, dom, z3db, is_and=False)
            expr = z3util.myAnd([pexpr, nexpr])

        z3db.add(self, expr)
        return expr


class Cores_d(dict):
    """
    rare case when diff c1 and c2 became equiv after simplification
    >>> dom = data.dom.Dom([('a',frozenset(['0','1'])),('b',frozenset(['0','1']))])
    >>> z3db = dom.z3db

    c1 = a & b
    >>> pc = Core([('a',frozenset('1'))])
    >>> pd = Core([('b',frozenset('0'))])
    >>> nc = Core()
    >>> nd = Core()
    >>> c1 = PNCore((pc,pd,nc,nd))

    c2 = b & a 
    >>> pc = Core([('b',frozenset('1'))])
    >>> pd = Core([('a',frozenset('0'))])
    >>> nc = Core()
    >>> nd = Core()
    >>> c2 = PNCore((pc,pd,nc,nd))

    >>> cores_d = Cores_d()
    >>> cores_d['L1'] = c1
    >>> cores_d['L2'] = c2
    >>> print cores_d
    1. L1: pc: a=1; pd: b=0; nc: true; nd: true
    2. L2: pc: b=1; pd: a=0; nc: true; nd: true

    >>> mlog.level = VLog.WARN
    >>> print cores_d.merge(dom, z3db)
    1. (2) pc: a=1; pd: b=0; nc: true; nd: true: (2) L1,L2

    >>> covs_d = Covs_d()
    >>> config = Config([('a', '1'), ('b', '1')])
    >>> covs_d.add('L1',config)
    >>> covs_d.add('L2',config)

    >>> mlog.level = VLog.WARN
    >>> cores_d = cores_d.analyze(dom, z3db, covs_d)
    >>> print cores_d.merge(dom, z3db, show_detail=False)
    1. (2) a=1 & b=1 (conj): (2) L1,L2

    >>> cores_d = cores_d.analyze(dom, z3db, covs_d=None)
    >>> print cores_d.merge(dom, z3db, show_detail=False)
    1. (2) a=1 & b=1 (conj): (2) L1,L2

    """

    def __setitem__(self, sid, pncore):
        assert isinstance(sid, str), sid
        assert isinstance(pncore, PNCore), pncore

        dict.__setitem__(self, sid, pncore)

    def __str__(self):
        return '\n'.join("{}. {}: {}"
                         .format(i+1, sid, self[sid])
                         for i, sid in enumerate(sorted(self)))

    def merge(self, dom, z3db, show_detail=False):
        assert isinstance(dom, data.dom.Dom)
        assert isinstance(z3db, data.dom.Z3DB)

        mcores_d = Mcores_d()
        cache = {}
        for sid, core in self.items():
            try:
                key = core.vstr
            except AttributeError:
                key = core

            if key not in cache:
                cache[key] = core

            mcores_d.add(cache[key], sid)

        mcores_d = mcores_d.fix_duplicates(dom, z3db)

        if show_detail:
            mcores_d.show_results()

        return mcores_d

    def analyze(self, dom, z3db, covs_d):
        """
        Simplify cores. If covs_d then also check that cores are valid invs
        """
        assert isinstance(dom, data.dom.Dom), dom
        assert isinstance(z3db, data.dom.Z3DB)
        if __debug__:
            if covs_d is not None:
                assert isinstance(
                    covs_d, data.config.Covs_d) and covs_d, covs_d
                assert len(self) == len(covs_d), (len(self), len(covs_d))

        if not self:
            return self

        def show_compare(sid, old_c, new_c):
            if old_c != new_c:
                mlog.info("sid {}: {} ~~> {}".
                          format(sid, old_c, new_c))
        mlog.info("analyze results for {} sids".format(len(self)))
        cores_d = Cores_d()

        if covs_d:
            mlog.info("verify ...")
            cache = {}
            for sid, core in self.items():
                configs = frozenset(covs_d[sid])
                key = (core, configs)
                if key not in cache:
                    core_ = core.verify(configs, dom)
                    cache[key] = core_
                    show_compare(sid, core, core_)
                else:
                    core_ = cache[key]

                cores_d[sid] = core
        else:
            cores_d = self

        mlog.info("simplify ...")
        cache = {}
        for sid in cores_d:
            core = cores_d[sid]
            if core not in cache:
                core_, expr = core.simplify(
                    dom, z3db, do_firsttime=(covs_d is not None))
                cache[core] = core_
                show_compare(sid, core, core_)
            else:
                core_ = cache[core]
            cores_d[sid] = core_

        return cores_d


class Mcores_d(dict):
    """
    A mapping from core -> {sids}
    """

    def add(self, core, sid):
        assert isinstance(core, PNCore), core
        assert isinstance(sid, str), str

        # super(Mcores_d, self).add_set(core, sid)
        if core not in self:
            self[core] = set()
        self[core].add(sid)

    def __str__(self):
        def _f(corecov):
            core, cov = corecov
            return (core.sstren, core.vstren, len(cov))

        mc = sorted(self.items(), key=_f)
        ss = ("{}. ({}) {}: {}"
              .format(i+1, core.sstren, core, data.base.str_of_cov(cov))
              for i, (core, cov) in enumerate(mc))
        return '\n'.join(ss)

    def fix_duplicates(self, dom, z3db):
        assert isinstance(dom, data.dom.Dom)
        assert not isinstance(z3db, data.dom.Dom)

        def find_dup(expr, d):
            for pc in d:
                expr_ = pc.z3expr(dom, z3db)
                if ((expr is None and expr_ is None) or
                    (expr is not None and expr_ is not None and
                     z3util.is_tautology(expr == expr_, z3db.solver))):
                    return pc

            return None  # no dup

        uniqs = {}
        for pc in self:
            expr = pc.z3expr(dom, z3db)
            dup = find_dup(expr, uniqs)
            if dup:
                uniqs[dup].add(pc)
            else:
                uniqs[pc] = set()

        if len(uniqs) == len(self):  # no duplicates
            return self
        else:
            mlog.info('merge {} dups'.format(len(self) - len(uniqs)))
            mc_d = Mcores_d()
            for pc in uniqs:
                for sid in self[pc]:
                    mc_d.add(pc, sid)

                for dup in uniqs[pc]:
                    for sid in self[dup]:
                        mc_d.add(pc, sid)
            return mc_d

    @property
    def vtyps(self):
        try:
            return self._vtyps
        except AttributeError:
            d = {'conj': 0, 'disj': 0, 'mix': 0}
            for core in self:
                vtyp = core.vtyp
                d[vtyp] += 1

            self._vtyps = (d['conj'], d['disj'], d['mix'])
            return self._vtyps

    @property
    def strens(self):
        """
        (strength,cores,sids)
        """
        strens = set(core.sstren for core in self)

        rs = []
        for stren in sorted(strens):
            cores = [c for c in self if c.sstren == stren]
            cov = set(sid for core in cores for sid in self[core])
            rs.append((stren, len(cores), len(cov)))
        return rs

    @property
    def strens_str(self): return self.str_of_strens(self.strens)

    def show_results(self):
        print("inferred results ({}):\n{}".format(len(self), self))
        mlog.info("strens (stren, nresults, nsids): {}"
                  .format(self.strens_str))

    @classmethod
    def str_of_strens(cls, strens):
        return ', '.join("({}, {}, {})".format(siz, ncores, ncov)
                         for siz, ncores, ncov in strens)
