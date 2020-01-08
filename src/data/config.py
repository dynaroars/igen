import pdb
import z3

import helpers.vcommon
import settings

import data.base
import data.dom
import data.core

DBG = pdb.set_trace
mlog = helpers.vcommon.getLogger(__name__, settings.logger_level)


class Config(data.base.HDict):
    """
    Hashable dictionary

    >>> c = Config([('a', '1'), ('b', '0'), ('c', '1')])
    >>> print c
    a=1 b=0 c=1

    >>> dom = data.dom.Dom([('a',frozenset(['1','2'])),\
    ('b',frozenset(['0','1'])),\
    ('c',frozenset(['0','1','2']))])
    >>> c.z3expr(Z3DB(dom))
    And(a == 1, b == 0, c == 1)
    """

    def __init__(self, config=data.base.HDict()):
        super(Config, self).__init__(config)

        assert all(data.base.is_setting(s) for s in self.items()), self

    def __str__(self, cov=None):
        assert cov is None or data.base.is_cov(cov), cov

        s = ' '.join(map(data.base.str_of_setting, self.items()))
        if cov:
            s = "{}: {}".format(s, data.base.str_of_cov(cov))
        return s

    def z3expr(self, z3db):
        assert isinstance(z3db, data.dom.Z3DB), z3db
        return z3db.expr_of_dict(self)

    @staticmethod
    def mk(n, f):
        """
        Try to create at most n configs by calling f().
        """
        assert isinstance(n, int) and n > 0, n
        assert callable(f), f

        pop = OrderedDict()
        for _ in range(n):
            c = f()
            c_iter = 0
            while c in pop and c_iter < 5:
                c_iter += 1
                c = f()

            if c not in pop:
                pop[c] = None

        assert len(pop) <= n, (len(pop), n)
        pop = pop.keys()
        return pop

    def c_implies(self, core):
        """
        self => conj core
        x=0&y=1 => x=0,y=1
        not(x=0&z=1 => x=0,y=1)
        """
        assert isinstance(core, data.core.Core), core

        return (not core or
                all(k in self and self[k] in core[k] for k in core))

    def d_implies(self, core):
        """
        self => disj core
        """
        assert isinstance(core, data.core.Core), core

        return (not core or
                any(k in self and self[k] in core[k] for k in core))

    @classmethod
    def eval(cls, configs, get_cov_f, kconstraints, z3db):
        """
        Eval (e.g., get coverage) configurations using function get_cov_f
        Ret a list of configs and their results
        """
        assert (isinstance(configs, list) and
                all(isinstance(c, (cls, Config)) for c in configs)
                and configs), configs
        assert callable(get_cov_f), get_cov_f
        assert all(z3.is_expr(constraint)
                   for constraint in kconstraints), kconstraints
        assert isinstance(z3db, data.dom.Z3DB), z3db

        def eval_get_cov(c):
            rs = get_cov_f(c)
            if not rs:
                mlog.warning("'{}' produces nothing".format(c))
            return rs

        def eval_f(c):
            if kconstraints:
                # assert not Z3.is_unsat(kconstraint)
                exprs = kconstraints + [c.z3expr(z3db)]
                isunsat = Z3.is_unsat(exprs, print_unsat_core=False)

                if isunsat:
                    print('invalid config')
                    return set()

            return eval_get_cov(c)

        def wprocess(tasks, Q):
            rs = [(c, eval_f(c)) for c in tasks]
            if Q is None:
                return rs
            else:
                Q.put(rs)

        tasks = list(set(configs))
        wrs = data.base.runMP("eval", tasks, wprocess,
                              chunksiz=1, doMP=settings.doMP and len(tasks) >= 2)

        return wrs


class Covs_d(dict):
    """
    A mapping from sid -> {configs}

    >>> c1 = Config([('a', '0'), ('b', '0'), ('c', '0')])
    >>> c2 = Config([('a', '0'), ('b', '0'), ('c', '1')])
    >>> covs_d = Covs_d()
    >>> assert 'l1' not in covs_d
    >>> covs_d.add('l1',c1)
    >>> covs_d.add('l1',c2)
    >>> assert 'l1' in covs_d
    >>> assert covs_d['l1'] == set([c1,c2])
    """

    def add(self, sid, config):
        assert isinstance(sid, str), sid
        assert isinstance(config, Config), config

        if sid not in self:
            self[sid] = set()
        self[sid].add(config)


class Configs_d(dict):
    """
    A mapping from config -> {covs}
    """

    def __setitem__(self, config, cov):
        assert isinstance(config, Config), config
        assert data.base.is_cov(cov), cov

        dict.__setitem__(self, config, cov)

    def __str__(self):
        ss = (c.__str__(self[c]) for c in self)
        return '\n'.join("{}. {}".format(i+1, s) for i, s in enumerate(ss))
