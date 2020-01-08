import pdb
import random
from collections import OrderedDict
from pathlib import Path
import os

import z3
import z3util

import helpers.vcommon
import settings

import data.config
import data.core

mlog = helpers.vcommon.getLogger(__name__, settings.logger_level)
DBG = pdb.set_trace


class Dom(OrderedDict):
    """
    >>> dom = Dom([('x',frozenset(['1','2'])),\
    ('y',frozenset(['1'])),\
    ('z',frozenset(['0','1','2'])),\
    ('w',frozenset(['a','b','c']))\
    ])

    >>> print dom
    4 vars and 18 pos configs
    1. x: (2) 1,2
    2. y: (1) 1
    3. z: (3) 0,1,2
    4. w: (3) a,b,c

    >>> assert dom.siz == len(dom.gen_configs_full()) == 18
    >>> assert dom.max_fsiz ==  3


    >>> random.seed(0)
    >>> configs = dom.gen_configs_rand(5)
    >>> print "\\n".join(map(str,configs))
    x=2 y=1 z=2 w=a
    x=2 y=1 z=2 w=b
    x=2 y=1 z=0 w=a
    x=1 y=1 z=0 w=a
    x=1 y=1 z=2 w=c

    >>> random.seed(0)
    >>> configs = dom.gen_configs_tcover1()
    >>> print "\\n".join(map(str,configs))
    x=2 y=1 z=0 w=a
    x=1 y=1 z=2 w=c
    x=1 y=1 z=1 w=b

   >>> z3db = Z3DB(dom)
    >>> assert len(z3db) == len(dom) and set(z3db) == set(dom)

    >>> random.seed(0)
    >>> configs = dom.gen_configs_rand_smt(5, z3db)
    >>> print "\\n".join(map(str,configs))
    x=2 y=1 z=0 w=a
    x=1 y=1 z=2 w=b
    x=2 y=1 z=1 w=c
    x=1 y=1 z=0 w=a
    x=2 y=1 z=2 w=c


    >>> random.seed(0)
    >>> configs = dom.gen_configs_rand_smt(5, z3db, configs+configs)
    >>> print "\\n".join(map(str, configs))
    x=2 y=1 z=2 w=b
    x=1 y=1 z=2 w=c
    x=2 y=1 z=0 w=c
    x=1 y=1 z=0 w=c
    x=1 y=1 z=1 w=c

    >>> new_configs = dom.gen_configs_rand_smt(dom.siz, z3db, configs)
    >>> assert len(new_configs) == dom.siz - len(configs), (len(new_configs), dom.siz, len(configs))

    >>> configs = dom.gen_configs_rand_smt(dom.siz, z3db)
    >>> assert len(configs) == dom.siz

    >>> configs = dom.gen_configs_rand_smt(dom.siz, z3db, configs)
    >>> assert not configs

    """

    def __str__(self):
        """
        """
        s = "{} vars and {} pos configs".format(len(self), self.siz)
        s_detail = '\n'.join("{}. {}: ({}) {}".format(
            i+1, k, len(vs), ','.join(sorted(vs)))
            for i, (k, vs) in enumerate(self.items()))
        s = "{}\n{}".format(s, s_detail)
        return s

    @property
    def sortedself(self):
        """
        use sorted for deterministic behavior (Python 3)
        """

        return dict((k, list(sorted(set(self[k])))) for k in self)

    @property
    def siz(self): return helpers.vcommon.vmul(len(vs) for vs in self.values())

    @property
    def max_fsiz(self):
        """
        Size of the largest finite domain
        """
        return max(len(vs) for vs in self.values())

    # Methods to generate configurations
    def gen_configs_full(self):  # TODO kconfig_contraint
        ns, vs = zip(*self.items())
        configs = [data.config.Config(zip(ns, c))
                   for c in itertools.product(*vs)]
        return configs

    def gen_configs_tcover1(self):
        """
        Return a set of tcover array of stren 1
        """
        unused = dict((k, list(sorted(set(self[k])))) for k in self)

        def mk():
            config = []
            for k in self:
                if k in unused:
                    v = random.choice(unused[k])
                    unused[k].remove(v)
                    if not unused[k]:
                        unused.pop(k)
                else:
                    v = random.choice(self.sortedself[k])

                config.append((k, v))
            return data.config.Config(config)

        configs = []
        while unused:
            configs.append(mk())
        return configs

    def gen_configs_rand(self, rand_n, config_cls=None):  # TODO kconfig_contraint
        assert 0 < rand_n <= self.siz, (rand_n, self.siz)

        if config_cls is None:
            config_cls = data.config.Config

        def rgen(): return [(k, random.choice(self.sortedself[k]))
                            for k in self]
        configs = list(set(config_cls(rgen()) for _ in range(rand_n)))
        return configs

    # generate configs using an SMT solver
    def config_of_model(self, model, config_cls):
        """
        Ret a config from a model
        """
        assert isinstance(model, dict), model
        assert config_cls, config_cls

        def _f(k): return (model[k] if k in model
                           else random.choice(self.sortedself[k]))
        config = config_cls((k, _f(k)) for k in self)
        return config

    def gen_configs_expr(self, expr, k, config_cls):
        """
        Return at most k configs satisfying expr
        """
        assert z3.is_expr(expr), expr
        assert k > 0, k
        assert config_cls, config_cls

        def _f(m):
            m = dict((str(v), str(m[v])) for v in m)
            return None if not m else self.config_of_model(m, config_cls)

        models = z3util.get_models(expr, k)
        assert models is not None, models  # z3 cannot solve this
        if not models:  # not satisfy
            return []
        else:
            assert len(models) >= 1, models
            configs = [_f(m) for m in models]
            return configs

    def gen_configs_exprs(self, yexprs, nexprs, k, config_cls):
        """
        Return a config satisfying yexprs but not nexprs
        """
        assert all(Z3DB.maybe_expr(e) for e in yexprs), yexprs
        assert all(Z3DB.maybe_expr(e) for e in nexprs), nexprs
        assert k > 0, k
        assert config_cls, config_cls

        # if z3.solve:
        #     pass
        yexprs = [e for e in yexprs if e is not None]
        nexprs = [z3.Not(e) for e in nexprs if e is not None]
        exprs = yexprs + nexprs
        assert exprs, 'empty exprs'

        expr = exprs[0] if len(exprs) == 1 else z3util.myAnd(exprs)
        return self.gen_configs_expr(expr, k, config_cls)

    def gen_configs_rand_smt(self, rand_n, z3db, existing_configs=[],
                             config_cls=None):
        """
        Create rand_n uniq configs
        """
        assert 0 < rand_n <= self.siz, (rand_n, self.siz)
        assert isinstance(z3db, Z3DB)
        assert isinstance(existing_configs, list), existing_configs

        if config_cls is None:
            config_cls = Config

        exprs = []

        existing_configs = set(existing_configs)
        if existing_configs:
            exprs = [c.z3expr(z3db) for c in existing_configs]
            nexpr = z3util.myOr(exprs)
            configs = self.gen_configs_exprs([], [nexpr], 1, config_cls)
            if not configs:
                return []
            else:
                config = configs[0]
        else:
            configs = self.gen_configs_rand(1, config_cls)
            assert len(configs) == 1, configs
            config = configs[0]

        for _ in range(rand_n - 1):
            exprs.append(config.z3expr(z3db))
            nexpr = z3util.myOr(exprs)
            configs_ = self.gen_configs_exprs([], [nexpr], 1, config_cls)
            if not configs_:
                break
            config = configs_[0]
            configs.append(config)

        return configs

    def gen_configs_cex(self, sel_core, existing_configs, z3db):
        """
        >>> dom = Dom([('a', frozenset(['1', '0'])), \
        ('b', frozenset(['1', '0'])), ('c', frozenset(['1', '0', '2']))])
        >>> z3db = dom.z3db

        >>> c1 = Config([('a', '0'), ('b', '0'), ('c', '0')])
        >>> c2 = Config([('a', '0'), ('b', '0'), ('c', '1')])
        >>> c3 = Config([('a', '0'), ('b', '0'), ('c', '2')])

        >>> c4 = Config([('a', '0'), ('b', '1'), ('c', '0')])
        >>> c5 = Config([('a', '0'), ('b', '1'), ('c', '1')])
        >>> c6 = Config([('a', '0'), ('b', '1'), ('c', '2')])

        >>> c7 = Config([('a', '1'), ('b', '0'), ('c', '0')])
        >>> c8 = Config([('a', '1'), ('b', '0'), ('c', '1')])
        >>> c9 = Config([('a', '1'), ('b', '0'), ('c', '2')])

        >>> c10 = Config([('a', '1'), ('b', '1'), ('c', '0')])
        >>> c11 = Config([('a', '1'), ('b', '1'), ('c', '1')])
        >>> c12 = Config([('a', '1'), ('b', '1'), ('c', '2')])

        >>> configs = [c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11]
        >>> nexpr = z3util.myOr([c.z3expr(z3db) for c in configs])
        >>> assert dom.gen_configs_exprs([None],[nexpr],k=1,config_cls=Config)[0] == c12

        >>> core = Core([('a',frozenset(['1']))])
        >>> core_expr = core.z3expr(z3db,z3util.myAnd)
        >>> assert dom.gen_configs_exprs([None],[nexpr],k=1, config_cls=Config)[0] == c12

        >>> core = Core([('a',frozenset(['0']))])
        >>> core_expr = core.z3expr(z3db,z3util.myAnd)
        >>> assert not dom.gen_configs_exprs([core_expr],[nexpr],k=1, config_cls=Config)

        >>> core = Core([('c',frozenset(['0','1']))])
        >>> core_expr = core.z3expr(z3db,z3util.myAnd)
        >>> assert not dom.gen_configs_exprs([core_expr],[nexpr],k=1, config_cls=Config)

        >>> core = Core([('c',frozenset(['0','2']))])
        >>> core_expr = core.z3expr(z3db,z3util.myAnd)
        >>> config = dom.gen_configs_exprs([core_expr],[nexpr],k=1, config_cls=Config)[0]
        >>> print config
        a=1 b=1 c=2


        sel_core = (c_core,s_core)
        create counterexample configs by changing settings in c_core,
        but these configs must satisfy s_core
        x=0,y=1  =>  [x=0,y=0,z=rand;x=0,y=2,z=rand;x=1,y=1;z=rand]
        """

        assert isinstance(sel_core, data.core.SCore), sel_core
        assert isinstance(z3db, Z3DB)

        configs = []
        c_core, s_core = sel_core

        # keep
        changes = []
        if sel_core.keep and (len(self) - len(c_core)):
            changes.append(c_core)

        # change
        def _new(): return data.core.Core((k, c_core[k]) for k in c_core)
        for k in c_core:
            vs = self[k] - c_core[k]
            for v in vs:
                new_core = _new()
                new_core[k] = frozenset([v])
                if s_core:
                    for sk, sv in s_core.items():
                        assert sk not in new_core, sk
                        new_core[sk] = sv
                changes.append(new_core)

        e_configs = [c.z3expr(z3db) for c in existing_configs]
        for changed_core in changes:
            yexpr = changed_core.z3expr(z3db, z3util.myAnd)
            # yexpr = z3.simplify(z3.And(yexpr, constrains))
            nexpr = z3util.myOr(e_configs)
            # logger.debug('constraint: {}, changed_core: {}, yexpr: {} '.format(constrains, changed_core, yexpr))
            configs_ = self.gen_configs_exprs(
                [yexpr], [nexpr], k=1, config_cls=data.config.Config)
            if not configs_:
                continue
            config = configs_[0]

            assert config.c_implies(changed_core)
            assert config not in existing_configs, \
                ("ERR: gen existing config {}".format(config))

            configs.append(config)
            e_configs.append(config.z3expr(z3db))

        return configs

    @classmethod
    def parse(cls, dom_dir):
        """
        Read domain info from a file.
        Also *extract* default configs (.default*)
        And *save*
        - kconstraints (.dimacs) if given
        - runscript (.run) if given
        """
        assert dom_dir.is_dir(), dom_dir

        def check(f, must_exist):
            if f.is_file():
                mlog.info("using '{}'".format(f))
                return f
            else:
                assert not must_exist, "'{}' does not exist".format(f)
                return None

        def get_lines(lines):
            rs = (line.split() for line in lines)
            rs_ = []
            for parts in rs:
                var = parts[0]
                vals = frozenset(parts[1:])
                rs_.append((var, vals))
            return rs_

        # domain file
        dom_file = check(dom_dir / 'dom', must_exist=True)
        dom = get_lines(helpers.vcommon.iread_strip(dom_file))
        dom = cls(dom)

        # run script
        dom.run_script = check(dom_dir / 'run', must_exist=True)
        mlog.info("using run script from '{}'".format(dom.run_script))

        dom.kconstraints_file = check(dom_dir / 'dimacs', must_exist=False)

        # potentially multiple default configs  (*.default*)
        configs = [dict(get_lines(helpers.vcommon.iread_strip(c)))
                   for c in os.scandir(dom_dir)
                   if c.is_file() and c.name.startswith("default")]
        configs = [[(k, list(c[k])[0]) for k in dom] for c in configs]

        return dom, configs

    def get_kconstraints(self, z3db):
        """
        parse k-constraint file
        """
        assert isinstance(z3db, Z3DB), z3db

        if self.kconstraints_file is None:
            return []

        lines = [l.strip()
                 for l in helpers.vcommon.iread(self.kconstraints_file)]
        lines = [l for l in lines]

        symbols = {}
        clauses = []
        for l in lines:
            if l.startswith('c'):
                if l.startswith('c kconfig_variable'):
                    pair = l.split(' ')
                    sidx, symbol = pair[2], pair[3]  # '2', 'y'
                    symbols[sidx] = symbol

            elif l.startswith("p"):  # p cnf 3 4
                _, _, nvars, nclauses = l.split(" ")
                nsymbols = int(nvars)
                assert nsymbols == len(symbols)
                nclauses = int(nclauses)
            else:  # clause,  -1 2
                clause = l.split()  # -2 11 0
                clause = frozenset(clause[: -1])  # remove the last 0
                clauses.append(clause)

        # kconstraint_symbols = set(symbols.values())
        # z3db_symbols = set(z3db.keys())

        clauses = frozenset(clauses)
        assert len(clauses) <= nclauses

        def _f(sidx):
            isNot = sidx.startswith('-')
            return isNot, sidx[1:] if isNot else sidx

        my01 = ['0', '1']
        myny = ['n', 'y']
        myStr = ['\'\'', '\'non_empty\'']

        def _g(isNot, sidx):
            s = symbols[sidx]
            try:
                s, d = z3db[s]
            except KeyError:
                return None

            if my01[0] in d and my01[1] in d:
                f, t = my01
            elif myny[0] in d and myny[1] in d:
                f, t = myny
            elif myStr[0] in d and myStr[1] in d:
                f, t = myStr
            else:
                assert False, d

            rs = s == d[f if isNot else t]
            return rs

        rs = []
        for clause in clauses:
            ss = [_f(sidx) for sidx in clause]  # -3
            ss = [_g(isNot, sidx) for isNot, sidx in ss]
            if all(s is not None for s in ss) and ss:
                rs.append(z3.simplify(z3util.Or(ss)))

        return rs


class Z3DB(dict):
    def __init__(self, dom):
        assert isinstance(dom, Dom), dom
        db = {}
        for k, vs in dom.items():
            vs = sorted(list(vs))
            ttyp, tvals = z3.EnumSort(k, vs)
            rs = [vv for vv in zip(vs, tvals)]
            rs.append(('typ', ttyp))
            db[k] = (z3.Const(k, ttyp), dict(rs))
        dict.__init__(self, db)

    @property
    def cache(self):
        try:
            return self._exprs_cache
        except AttributeError:
            self._exprs_cache = {}
            return self._exprs_cache

    @property
    def solver(self):
        try:
            solver = self._solver
            return solver
        except AttributeError:
            self._solver = z3.Solver()
            return self._solver

    def add(self, k, v):
        assert self.maybe_expr(v), v
        self.cache[k] = v

    def expr_of_dict(self, d):
        # s=1 t=1 u=1 v=1 x=0 y=0 z=4
        assert all(k in self for k in d), (d, self)

        if d in self.cache:
            return self.cache[d]

        rs = [self.get_eq_expr(k, v) for k, v in d.items()]
        expr = z3util.myAnd(rs)
        self.add(d, expr)
        return expr

    def expr_of_dict_dict(self, d, is_and):
        # s=0 t=0 u=0 v=0 x=1 y=1 z=0,1,2  =>  ... (z=0 or z=1 or z=2)
        assert all(k in self for k in d), (d, self)

        key = (d, is_and)
        if key in self.cache:
            return self.cache[key]

        myf = z3util.myAnd if is_and else z3util.myOr
        rs = [self.get_eq_expr(k, vs) for k, vs in d.items()]
        expr = myf(rs)

        self.add(key, expr)
        return expr

    def get_eq_expr(self, k, v):
        """
        e.g., x == 0
        x == 0 or x == 1
        """
        s, d = self[k]
        if isinstance(v, frozenset):
            return z3util.myOr([s == d[v_] for v_ in v])
        else:
            return s == d[v]

    @staticmethod
    def maybe_expr(expr):
        # not None => z3expr
        return expr is None or z3.is_expr(expr)
